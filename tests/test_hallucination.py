import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oellm_rlvr import eu_tasks as E
from oellm_rlvr import hallucination as H


def test_fabricated_rows_grade_abstention_and_any_answer_as_wrong():
    rows = H.fabricated_qa(6, langs=["sv", "de"])
    assert len(rows) == 12 and len({r["id"] for r in rows}) == 12
    assert {r["subset"] for r in rows} == set(H.FABRICATED_TYPES)
    for r in rows:
        assert r["subject"] in r["prompt"] and r["prompt"].endswith(E.QA_INSTRUCTION[r["language"]])
        label = {**r["label"], "mode": "default"}
        assert E.grade_qa("ANSWER: UNKNOWN", label) == "abstain"
        assert E.grade_qa("ANSWER: vet inte", label) == "abstain"
        assert E.grade_qa("ANSWER: 1957", label) == "wrong"
        assert E.score("ANSWER: Stockholm", label) == -1.0
    assert H.fabricated_qa(6, langs=["sv"]) == H.fabricated_qa(6, langs=["sv"])  # deterministic


def test_threshold_prompt_states_target_and_penalty():
    prompt = E.qa_prompt("birth_year", "sv", {"s": "Selma Lagerlöf", "d": "författare, Sverige"})
    out = H.threshold_prompt(prompt, "sv", 0.75)
    assert out.startswith("Vilket år föddes Selma Lagerlöf") and "75 %" in out and "kostar 3 poäng" in out
    assert "kostar 9 poäng" in H.threshold_prompt(prompt, "sv", 0.9)
    assert set(H.THRESHOLD_INSTRUCTION) == set(E.QA_INSTRUCTION)
    with pytest.raises(ValueError):
        H.threshold_prompt("no instruction", "sv", 0.5)


def test_consistency_policy_trades_wrong_answers_for_abstentions(tmp_path):
    label = {"suite": "qa", "kind": "year", "answers": ["1858"], "mode": "default"}
    rows = [{"id": "eu_qa-birth_year-Q1-sv", "messages": [{"role": "user", "content": "q"}],
             "ground_truth": json.dumps(label), "dataset": "eu_qa"},
            {"id": "eu_qa-birth_year-Q2-sv", "messages": [{"role": "user", "content": "q"}],
             "ground_truth": json.dumps(label), "dataset": "eu_qa"}]
    pool = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pool)
    # Q1: consistent and right; Q2: scattered guesses, all wrong.
    texts = {"eu_qa-birth_year-Q1-sv": ["ANSWER: 1858"] * 4,
             "eu_qa-birth_year-Q2-sv": ["ANSWER: 1901", "ANSWER: 1850", "ANSWER: 1777", "ANSWER: UNKNOWN"]}
    samples = tmp_path / "s.jsonl"
    with samples.open("w") as f:
        for pid, ts in texts.items():
            for i, t in enumerate(ts):
                f.write(json.dumps({"id": pid, "dataset": "eu_qa", "sample_index": i, "text": t, "token_ids": [1],
                                    "finished": True}) + "\n")
    report = H.consistency(pool, [samples])
    assert report["as_sampled"]["wrong"] == 3 / 8 and report["auroc_agreement_vs_correct"] == 1.0
    half = next(p for p in report["policy"] if p["tau"] == 0.5)
    assert half["correct"] == 0.5 and half["wrong"] == 0.0 and half["abstain"] == 0.5
