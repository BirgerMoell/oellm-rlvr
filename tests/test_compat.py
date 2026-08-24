import asyncio
import importlib
import sys
from enum import Enum
from types import ModuleType

from oellm_rlvr.compat import (
    install_post_import_patch,
    patch_vllm_mamba_module,
    replace_none_enum_value,
    wrap_async_weight_update,
)


class MixedEnum(Enum):
    VALUE = "value"
    CUSTOM = None


def test_replace_none_enum_value_makes_values_homogeneous() -> None:
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is True
    assert [member.value for member in MixedEnum] == ["value", ""]
    assert MixedEnum("") is MixedEnum.CUSTOM
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is False


def test_mamba_module_patch() -> None:
    class FakeMambaEnum(Enum):
        VALUE = "value"
        CUSTOM = None

    module = ModuleType("fake_vllm_registry")
    module.MambaAttentionBackendEnum = FakeMambaEnum
    assert patch_vllm_mamba_module(module) is True
    assert FakeMambaEnum.CUSTOM.value == ""


def test_post_import_patch_is_lazy_and_one_shot(tmp_path, monkeypatch) -> None:
    module_name = "oellm_post_import_target"
    (tmp_path / f"{module_name}.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    calls: list[str] = []

    assert install_post_import_patch(module_name, lambda module: calls.append(module.__name__)) is True
    assert module_name not in sys.modules
    imported = importlib.import_module(module_name)
    assert imported.VALUE == 1
    assert calls == [module_name]


def test_post_import_patch_handles_loaded_module(monkeypatch) -> None:
    module_name = "oellm_already_loaded_target"
    module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    calls: list[ModuleType] = []

    assert install_post_import_patch(module_name, calls.append) is False
    assert calls == [module]


def test_weight_update_is_wrapped_in_transaction() -> None:
    class FakeAsyncLLM:
        def __init__(self) -> None:
            self.calls: list[object] = []

        async def start_weight_update(self, is_checkpoint_format: bool) -> None:
            self.calls.append(("start", is_checkpoint_format))

        async def update_weights(self, request: object) -> str:
            self.calls.append(("update", request))
            return "done"

        async def finish_weight_update(self) -> None:
            self.calls.append(("finish",))

    assert wrap_async_weight_update(FakeAsyncLLM) is True
    assert wrap_async_weight_update(FakeAsyncLLM) is False
    engine = FakeAsyncLLM()
    assert asyncio.run(engine.update_weights("request")) == "done"
    assert engine.calls == [("start", True), ("update", "request"), ("finish",)]
