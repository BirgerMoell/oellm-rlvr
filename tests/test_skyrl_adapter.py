from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from oellm_rlvr.skyrl_adapter import export_skyrl_math

ROOT = Path(__file__).parents[1]


def test_export_skyrl_math_builds_grouped_reward_schema(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "m1",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "ground_truth": "4",
                "oellm_source_dataset": "birgermoell/oellm-math-rlvr",
            }
        )
        + "\n"
    )
    output = tmp_path / "train.parquet"

    report = export_skyrl_math(source, output, copies=4)

    rows = pq.read_table(output).to_pylist()
    assert report["rows"] == 4
    assert rows[0]["env_class"] == "gsm8k"
    assert rows[0]["reward_spec"]["ground_truth"] == "4"
    assert rows[0]["prompt"][-1]["content"].endswith("numeric answer.")
    assert {row["extra_info"]["copy_index"] for row in rows} == {0, 1, 2, 3}


def test_lumi_skyrl_math_smoke_is_consistently_text_only() -> None:
    script = (ROOT / "scripts/lumi_skyrl_amd_smoke.sbatch").read_text()
    assert "trainer.policy.language_model_only=true" in script
    assert "trainer.ref.language_model_only=true" in script
    assert "generator.inference_engine.language_model_only=true" in script
