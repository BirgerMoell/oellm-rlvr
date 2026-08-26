from __future__ import annotations

import os

from oellm_rlvr.compat import install_post_import_patch

if os.environ.get("OELLM_PATCH_VLLM_MAMBA_ENUM") == "1":
    from oellm_rlvr.compat import patch_vllm_mamba_module

    install_post_import_patch(
        "vllm.v1.attention.backends.registry",
        patch_vllm_mamba_module,
    )

if os.environ.get("OELLM_PATCH_VLLM_WEIGHT_UPDATE") == "1":
    from oellm_rlvr.compat import patch_vllm_weight_module

    install_post_import_patch(
        "vllm.v1.engine.async_llm",
        patch_vllm_weight_module,
    )

if os.environ.get("OELLM_PATCH_MATH_EQUIV_THREADS") == "1":
    from oellm_rlvr.compat import patch_math_equivalence_module

    install_post_import_patch(
        "open_instruct.math_utils",
        patch_math_equivalence_module,
    )

if os.environ.get("OELLM_PATCH_SWERL_APPTAINER_KWARGS") == "1":
    from oellm_rlvr.compat import wrap_swerl_create_backend

    install_post_import_patch(
        "open_instruct.environments.swerl_vanillux_sandbox",
        wrap_swerl_create_backend,
    )
