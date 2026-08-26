from pathlib import Path

import pytest

from oellm_rlvr.tmax_launcher import MI250X_GCD_SPEC, _install_local_snapshot_alias, _rewrite_model_arg


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
