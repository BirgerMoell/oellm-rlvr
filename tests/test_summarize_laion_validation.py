import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("summarize_laion_validation", Path(__file__).parents[1] / "scripts" / "summarize_laion_validation.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def trial(jobs, arm, n, task_dir, toml_name, reward, exception=None):
    d = jobs / arm / f"{task_dir}__{n}"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "task_name": toml_name, "trial_name": f"{task_dir}__{n}", "task_id": {"path": f"/pack/{task_dir}"},
        "verifier_result": {"rewards": {"reward": reward}}, "exception_info": exception}))


def test_tasks_are_matched_by_pack_directory_not_toml_name(tmp_path, capsys, monkeypatch):
    jobs, pack = tmp_path / "jobs", tmp_path / "pack"
    for name in ("good", "always_passes", "oracle_fails"):
        (pack / name).mkdir(parents=True)
    for n in range(2):
        trial(jobs, "oracle", n, "good", "bench/some-name", 1.0)
        trial(jobs, "nop", n, "good", "bench/some-name", 0.0)
        trial(jobs, "oracle", n, "always_passes", "bench/other", 1.0)
        trial(jobs, "nop", n, "always_passes", "bench/other", 1.0)        # nop must fail
        trial(jobs, "oracle", n, "oracle_fails", "bench/third", 0.0)
        trial(jobs, "nop", n, "oracle_fails", "bench/third", 0.0)
    monkeypatch.setattr(sys, "argv", ["x", "--jobs", str(jobs), "--tasks-dir", str(pack), "--attempts", "2",
                                      "--compatible-out", str(tmp_path / "ok.jsonl"), "--report-out", str(tmp_path / "r.json")])
    assert S.main() == 0
    assert [json.loads(line)["task_id"] for line in (tmp_path / "ok.jsonl").read_text().splitlines()] == ["good"]
    report = json.loads((tmp_path / "r.json").read_text())
    reasons = {f["task_id"]: (f["oracle"], f["nop"]) for f in report["failures"]}
    assert reasons == {"always_passes": (None, "reward=1.0"), "oracle_fails": ("reward=0.0", None)}


def test_exception_or_too_few_trials_disqualify():
    ok = {"reward": 1.0, "exception": None}
    assert S.verdict([ok, ok], 1.0, 2) is None
    assert S.verdict([ok], 1.0, 2) == "only 1/2 trials"
    assert S.verdict([ok, {"reward": 1.0, "exception": {"x": 1}}], 1.0, 2) == "exception"
