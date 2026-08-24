from __future__ import annotations

import os

if os.environ.get("OELLM_PATCH_VLLM_MAMBA_ENUM") == "1":
    from oellm_rlvr.compat import patch_vllm_mamba_enum

    patch_vllm_mamba_enum()

if os.environ.get("OELLM_PATCH_VLLM_WEIGHT_UPDATE") == "1":
    from oellm_rlvr.compat import patch_vllm_weight_update

    patch_vllm_weight_update()
