from __future__ import annotations

import json
from pathlib import Path

from oellm_rlvr.checkpoint import build_checkpoint_manifest, write_checkpoint_manifest
from oellm_rlvr.config import RunConfig, load_config
from oellm_rlvr.opd import preflight_opd, prepare_opd_dataset
from oellm_rlvr.opd_reward import compute_math_score, compute_score

ROOT = Path(__file__).parents[1]


def _checkpoint(root: Path, *, tokenizer: str = "{}", chat_template: str = "{{ messages }}") -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text(tokenizer)
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))
    (root / "generation_config.json").write_text("{}")


def _manifest(model: Path, output: Path, *, model_id: str, revision: str) -> None:
    manifest = build_checkpoint_manifest(model, model_id=model_id, revision=revision)
    write_checkpoint_manifest(manifest, output)


def _opd_config(
    tmp_path: Path,
    *,
    mismatched_teacher_tokenizer: bool = False,
    mismatched_teacher_template: bool = False,
) -> RunConfig:
    revision = "a" * 40
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    _checkpoint(student)
    _checkpoint(
        teacher,
        tokenizer='{"different":true}' if mismatched_teacher_tokenizer else "{}",
        chat_template="different template" if mismatched_teacher_template else "{{ messages }}",
    )
    student_manifest = tmp_path / "student-manifest.json"
    teacher_manifest = tmp_path / "teacher-manifest.json"
    _manifest(student, student_manifest, model_id="org/student", revision=revision)
    _manifest(teacher, teacher_manifest, model_id="org/teacher", revision=revision)

    raw = load_config(ROOT / "configs/lumi-opd-qwen35-2b-contract-smoke.yaml").model_dump()
    raw["model"].update(
        name_or_path="org/student",
        local_path=str(student),
        revision=revision,
        manifest_path=str(student_manifest),
    )
    raw["distillation"]["teachers"][0].update(
        name_or_path="org/teacher",
        local_path=str(teacher),
        revision=revision,
        manifest_path=str(teacher_manifest),
    )
    return RunConfig.model_validate(raw)


def test_opd_preflight_accepts_same_tokenizer_and_pinned_manifests(tmp_path: Path) -> None:
    report = preflight_opd(_opd_config(tmp_path))

    assert report["ok"] is True
    assert report["student"]["tokenizer_sha256"] == report["teachers"][0]["tokenizer_sha256"]


def test_opd_preflight_rejects_tokenizer_mismatch(tmp_path: Path) -> None:
    report = preflight_opd(_opd_config(tmp_path, mismatched_teacher_tokenizer=True))

    assert report["ok"] is False
    assert any("tokenizer does not match" in error for error in report["errors"])


def test_opd_preflight_warns_on_chat_template_mismatch(tmp_path: Path) -> None:
    report = preflight_opd(_opd_config(tmp_path, mismatched_teacher_template=True))

    assert report["ok"] is True
    assert any("chat template differs" in warning for warning in report["warnings"])


def test_prepare_opd_dataset_converts_prompt_only_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "prompt-1",
                "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
            }
        )
        + "\n"
    )
    output = tmp_path / "opd.jsonl"

    report = prepare_opd_dataset(source, output, data_source="general-teacher")
    row = json.loads(output.read_text())

    assert report["rows"] == 1
    assert row["data_source"] == "general-teacher"
    assert row["prompt"][0]["role"] == "user"
    assert row["reward_model"]["ground_truth"] == ""
    assert row["extra_info"]["id"] == "prompt-1"


def test_opd_reward_functions_separate_pure_and_hybrid_signals() -> None:
    assert compute_score(solution_str="anything", ground_truth="5") == 0
    assert compute_math_score(solution_str="work\n\\boxed{5}", ground_truth="5") == 1
    assert compute_math_score(solution_str="work\n\\boxed{6}", ground_truth="5") == 0
