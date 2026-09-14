from __future__ import annotations

import pytest

from oellm_rlvr.checkpoint_repair import (
    is_qwen35_inference_metadata,
    qwen35_text_key,
    repair_qwen35_text_checkpoint,
)


def test_qwen35_text_key_strips_only_language_model_prefix() -> None:
    assert qwen35_text_key("model.language_model.layers.0.mlp.up_proj.weight") == (
        "model.layers.0.mlp.up_proj.weight"
    )
    assert qwen35_text_key("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert qwen35_text_key("lm_head.weight") == "lm_head.weight"


def test_qwen35_text_key_rejects_unexpected_tensors() -> None:
    with pytest.raises(ValueError, match="unexpected Qwen3.5"):
        qwen35_text_key("model.visual.patch_embed.weight")


def test_inference_metadata_filter_excludes_training_checkpoints(tmp_path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}\n")
    templates = tmp_path / "chat_templates"
    templates.mkdir()
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    training_args = tmp_path / "training_args.bin"
    training_args.write_bytes(b"training-only")

    assert is_qwen35_inference_metadata(tokenizer)
    assert is_qwen35_inference_metadata(templates)
    assert not is_qwen35_inference_metadata(checkpoint)
    assert not is_qwen35_inference_metadata(training_args)


def test_repair_copies_only_inference_metadata(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        '{"architectures":["Qwen3_5ForCausalLM"],"dtype":"float32",'
        '"model_type":"qwen3_5_text"}\n'
    )
    (source / "tokenizer.json").write_text("{}\n")
    (source / "chat_template.jinja").write_text("{{ messages }}\n")
    (source / "README.md").write_text("training model card\n")
    (source / "training_args.bin").write_bytes(b"training-only")
    checkpoint = source / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"must not be copied")
    safetensors.save_file(
        {
            "model.language_model.layers.0.weight": torch.ones(2, dtype=torch.float32),
            "lm_head.weight": torch.zeros(2, dtype=torch.float32),
        },
        source / "model.safetensors",
    )

    manifest = repair_qwen35_text_checkpoint(
        source,
        output,
        max_shard_bytes=1024,
        cast_dtype="bfloat16",
    )

    assert (output / "tokenizer.json").is_file()
    assert (output / "chat_template.jinja").is_file()
    assert not (output / "README.md").exists()
    assert not (output / "training_args.bin").exists()
    assert not (output / "checkpoint-1000").exists()
    assert manifest["copied_inference_metadata"] == ["chat_template.jinja", "tokenizer.json"]
    assert manifest["output_tensor_dtypes"] == {"torch.bfloat16": 2}
