from __future__ import annotations

import pytest

from oellm_rlvr.checkpoint_repair import qwen35_text_key


def test_qwen35_text_key_strips_only_language_model_prefix() -> None:
    assert qwen35_text_key("model.language_model.layers.0.mlp.up_proj.weight") == (
        "model.layers.0.mlp.up_proj.weight"
    )
    assert qwen35_text_key("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert qwen35_text_key("lm_head.weight") == "lm_head.weight"


def test_qwen35_text_key_rejects_unexpected_tensors() -> None:
    with pytest.raises(ValueError, match="unexpected Qwen3.5"):
        qwen35_text_key("model.visual.patch_embed.weight")
