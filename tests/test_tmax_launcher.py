from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import oellm_rlvr.tmax_launcher as launcher
from oellm_rlvr.tmax_launcher import (
    MI250X_GCD_SPEC,
    _install_local_snapshot_alias,
    _rewrite_model_arg,
    _run_backend_entrypoint,
    _run_grpo_fast,
)


def test_mi250x_spec_is_per_gcd() -> None:
    assert MI250X_GCD_SPEC == {
        "flops": 191.5e12,
        "memory_size": 64e9,
        "memory_bandwidth": 1.6e12,
    }


def test_rewrite_model_arg_uses_local_path() -> None:
    args = ["--foo", "bar", "--model_name_or_path", "owner/model", "--baz", "qux"]
    assert _rewrite_model_arg(args, "owner/model", "/models/model") == [
        "--foo",
        "bar",
        "--model_name_or_path",
        "/models/model",
        "--baz",
        "qux",
    ]
    assert args[3] == "owner/model"


def test_local_model_alias_intercepts_snapshot_download(tmp_path: Path) -> None:
    hub = pytest.importorskip("huggingface_hub")
    (tmp_path / "config.json").write_text("{}")
    original = hub.snapshot_download
    try:
        resolved = _install_local_snapshot_alias("owner/model", str(tmp_path))
        assert resolved == str(tmp_path.resolve())
        assert hub.snapshot_download("owner/model") == str(tmp_path.resolve())
        assert hub.snapshot_download(str(tmp_path.resolve())) == str(tmp_path.resolve())
    finally:
        hub.snapshot_download = original


def _fake_grpo_module() -> tuple[ModuleType, list[object], dict[str, object]]:
    module = ModuleType("open_instruct.grpo_fast")
    parsed = [object() for _ in range(6)]
    observed: dict[str, object] = {}

    class FakeParser:
        def __init__(self, config_types):
            observed["config_types"] = config_types

        def set_defaults(self, **kwargs):
            observed["defaults"] = kwargs

        def parse_args_into_dataclasses(self):
            return parsed

    module.ArgumentParserPlus = FakeParser
    module.grpo_utils = SimpleNamespace(GRPOExperimentConfig="experiment")
    module.TokenizerConfig = "tokenizer"
    module.ModelConfig = "model"
    module.data_loader_lib = SimpleNamespace(
        StreamingDataLoaderConfig="streaming",
        VLLMConfig="vllm",
    )
    module.EnvsConfig = "envs"
    module.main = lambda *values: observed.setdefault("main", values)
    return module, parsed, observed


def test_hierarchical_grpo_entrypoint_parses_and_calls_pinned_main() -> None:
    module, parsed, observed = _fake_grpo_module()

    _run_grpo_fast(module)

    assert observed["config_types"] == (
        "experiment",
        "tokenizer",
        "model",
        "streaming",
        "vllm",
        "envs",
    )
    assert observed["defaults"] == {
        "exp_name": "grpo",
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 1,
    }
    assert observed["main"] == tuple(parsed)
    assert module.args is parsed[0]
    assert module.tokenizer_config is parsed[1]
    assert module.model_config is parsed[2]
    assert module.streaming_config is parsed[3]
    assert module.vllm_config is parsed[4]
    assert module.tools_config is parsed[5]


def test_hierarchical_backend_imports_canonical_grpo_module(monkeypatch) -> None:
    module, _, observed = _fake_grpo_module()
    imports: list[str] = []
    patched: list[ModuleType] = []
    monkeypatch.setenv("OELLM_WEIGHT_TRANSFER", "hierarchical")
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: imports.append(name) or module)
    monkeypatch.setattr(launcher, "patch_open_instruct_grpo_module", lambda value: patched.append(value))
    monkeypatch.setattr(
        launcher.runpy,
        "run_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("run_path must not run")),
    )

    _run_backend_entrypoint("open_instruct/grpo_fast.py")

    assert imports == ["open_instruct.grpo_fast"]
    assert patched == [module]
    assert "main" in observed
