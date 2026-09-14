import json
from pathlib import Path

import pyarrow.parquet as pq

from oellm_rlvr.datasets import write_rows


def test_reasoning_bridge_filters_contract_and_adds_replay(tmp_path: Path) -> None:
    import importlib.util

    script = Path(__file__).parents[1] / "scripts/build_multilingual_reasoning_sft_bridge.py"
    spec = importlib.util.spec_from_file_location("reasoning_bridge_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    contract = tmp_path / "languages"
    contract.write_text("eng: English: eng_Latn\nswe: Swedish: swe_Latn\nkat: Georgian: kat_Geor\n")
    multilingual = tmp_path / "multilingual.parquet"
    write_rows(
        [
            {
                "language": "sv",
                "messages": [
                    {"role": "user", "content": "Räkna 1+1."},
                    {"role": "assistant", "content": "<think>1+1=2</think>\\boxed{2}"},
                ],
                "source_id": "source-1",
                "source_license": "cc-by-4.0",
                "quality": {"accepted": True},
            },
            {
                "language": "ru",
                "messages": [
                    {"role": "user", "content": "1+1"},
                    {"role": "assistant", "content": "<think>2</think>\\boxed{2}"},
                ],
                "source_id": "source-2",
                "source_license": "cc-by-4.0",
                "quality": {"accepted": True},
            },
        ],
        multilingual,
    )
    english = tmp_path / "english.parquet"
    write_rows(
        [
            {
                "messages": [
                    {"role": "user", "content": "Compute 2+2."},
                    {"role": "assistant", "content": "<think>2+2=4</think>\\boxed{4}"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Compute 3+3."},
                    {"role": "assistant", "content": "<think>3+3=6</think>\\boxed{6}"},
                ]
            },
        ],
        english,
    )
    general = tmp_path / "general.parquet"
    write_rows(
        [
            {
                "messages": [
                    {"role": "user", "content": "Say hello."},
                    {"role": "assistant", "content": "Hello."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Say goodbye."},
                    {"role": "assistant", "content": "Goodbye."},
                ]
            },
        ],
        general,
    )

    output = tmp_path / "bridge"
    manifest = module.build_bridge(
        multilingual=multilingual,
        english_reasoning=english,
        general_replay=general,
        language_contract=contract,
        output=output,
        replay_rows=1,
        multilingual_revision="abc",
        contract_revision="def",
    )
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert len(rows) == 3
    assert manifest["rows_by_role"] == {
        "english_reasoning_replay": 1,
        "general_instruction_replay": 1,
        "multilingual_reasoning": 1,
    }
    assert manifest["dropped"] == {"outside_target_contract": 1}
    assert json.loads((output / "manifest.json").read_text()) == manifest
