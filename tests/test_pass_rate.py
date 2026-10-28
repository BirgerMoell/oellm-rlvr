import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oellm_rlvr import pass_rate


def _pool(tmp_path, n=6):
    rows = [{"id": f"p{i}", "messages": [{"role": "user", "content": f"q{i}"}], "ground_truth": str(i),
             "dataset": "math" if i % 2 else "eu_qa"} for i in range(n)]
    path = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _samples(tmp_path, rewards_by_id, finished=True):
    path = tmp_path / "samples.jsonl"
    with path.open("w") as f:
        for pid, rewards in rewards_by_id.items():
            for k, r in enumerate(rewards):
                f.write(json.dumps({"id": pid, "dataset": "x", "sample_index": k, "text": str(r),
                                    "token_ids": [1, 2, 3], "finished": finished}) + "\n")
    return path


class _FakeVerifier:
    """Reward = the float written in the completion text (stands in for the training verifier)."""

    async def async_call(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        return SimpleNamespace(score=float(prediction))


@pytest.fixture
def fake_verifiers(monkeypatch):
    monkeypatch.setattr(pass_rate, "build_verifiers", lambda *a, **k: {"math": _FakeVerifier(), "eu_qa": _FakeVerifier()})


def test_shards_partition_the_pool(tmp_path):
    rows = pass_rate.read_pool(_pool(tmp_path, 40))
    parts = [pass_rate.shard_rows(rows, s, 3) for s in range(3)]
    assert sorted(r["id"] for part in parts for r in part) == sorted(r["id"] for r in rows)
    with pytest.raises(ValueError):
        pass_rate.shard_rows(rows, 3, 3)


def test_score_profile_select_keeps_only_prompts_with_signal(tmp_path, fake_verifiers):
    pool = _pool(tmp_path)
    rewards = {"p0": [1, 1, 1, 1], "p1": [0, 0, 0, 0], "p2": [1, 0, 0, 0], "p3": [1, 1, 0, 1],
               "p4": [-1, -1, -1, -1], "p5": [0.5, 0.5, 1, 0.5]}
    samples = _samples(tmp_path, rewards)
    scores = tmp_path / "scores.jsonl"
    assert pass_rate.score(pool, [samples], scores)["errors"] == 0
    assert pass_rate.score(pool, [samples], scores)["new"] == 0  # resumable: nothing rescored
    out = tmp_path / "filtered.parquet"
    summary = pass_rate.select(pool, scores, out, samples=4, model="ckpt")
    kept = {r["id"] for r in pq.read_table(out).to_pylist()}
    assert kept == {"p2", "p3", "p5"}
    assert summary["kept"] == 3 and summary["model"] == "ckpt"
    bands = {p["id"]: p["band"] for p in json.loads(out.with_suffix(".profile.json").read_text())["prompts"]}
    assert bands["p0"] == "no_signal_solved" and bands["p1"] == "no_signal_unsolved"
    assert bands["p4"] == "no_signal_unsolved" and bands["p2"] == "hard" and bands["p3"] == "easy"


def test_select_band_cap_and_truncation(tmp_path, fake_verifiers):
    pool = _pool(tmp_path)
    rewards = {pid: [1, 0, 0, 0] if i % 2 else [1, 1, 1, 0] for i, pid in enumerate(f"p{i}" for i in range(6))}
    scores = tmp_path / "scores.jsonl"
    pass_rate.score(pool, [_samples(tmp_path, rewards)], scores)
    summary = pass_rate.select(pool, scores, tmp_path / "a.parquet", samples=4, min_solve=0.5)
    assert summary["kept"] == 3  # only the 3/4-solved prompts
    summary = pass_rate.select(pool, scores, tmp_path / "b.parquet", samples=4, cap_per_dataset={"math": 1})
    assert summary["kept_by_dataset"]["math"] == 1
    trunc = tmp_path / "t"
    trunc.mkdir()
    tscores = trunc / "scores.jsonl"
    pass_rate.score(pool, [_samples(trunc, rewards, finished=False)], tscores)
    assert pass_rate.select(pool, tscores, trunc / "c.parquet", samples=4, max_truncated=0.5)["kept"] == 0


def test_incomplete_or_errored_groups_are_never_selected(tmp_path, monkeypatch):
    pool = _pool(tmp_path, 2)

    class Broken:
        async def async_call(self, *args, **kwargs):
            raise RuntimeError("code api down")

    monkeypatch.setattr(pass_rate, "build_verifiers", lambda *a, **k: {"math": Broken(), "eu_qa": _FakeVerifier()})
    scores = tmp_path / "scores.jsonl"
    report = pass_rate.score(pool, [_samples(tmp_path, {"p0": [1, 0], "p1": [1, 0]})], scores)
    assert report["errors"] == 2
    summary = pass_rate.select(pool, scores, tmp_path / "f.parquet", samples=2)
    assert summary["kept"] == 1 and summary["bands_by_dataset"]["math"] == {"incomplete": 1}


def test_unknown_dataset_is_an_error(tmp_path, monkeypatch):
    pool = _pool(tmp_path, 2)
    monkeypatch.setattr(pass_rate, "build_verifiers", lambda *a, **k: {"math": _FakeVerifier()})
    with pytest.raises(ValueError, match="no verifier"):
        pass_rate.score(pool, [_samples(tmp_path, {"p0": [1], "p1": [0]})], tmp_path / "s.jsonl")
