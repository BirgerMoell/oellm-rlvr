from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable
from enum import Enum
from functools import wraps
from types import ModuleType
from typing import Any

ModulePatch = Callable[[ModuleType], object]


class _PostImportLoader(importlib.abc.Loader):
    def __init__(
        self,
        original: importlib.abc.Loader,
        callback: ModulePatch,
        finder: _PostImportFinder,
    ) -> None:
        self.original = original
        self.callback = callback
        self.finder = finder

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self.original, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        try:
            self.original.exec_module(module)
            self.callback(module)
        finally:
            if self.finder in sys.meta_path:
                sys.meta_path.remove(self.finder)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_name: str, callback: ModulePatch) -> None:
        self.module_name = module_name
        self.callback = callback

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None) -> Any:
        if fullname != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostImportLoader(spec.loader, self.callback, self)
        return spec


def install_post_import_patch(module_name: str, callback: ModulePatch) -> bool:
    """Patch an imported module, or register a one-shot hook without importing it."""
    module = sys.modules.get(module_name)
    if module is not None:
        callback(module)
        return False
    sys.meta_path.insert(0, _PostImportFinder(module_name, callback))
    return True


def replace_none_enum_value(enum_type: type[Enum], member_name: str, replacement: str) -> bool:
    member = enum_type.__members__[member_name]
    if member.value is not None:
        return False
    enum_type._value2member_map_.pop(None, None)
    member._value_ = replacement
    enum_type._value2member_map_[replacement] = member
    return True


def patch_vllm_mamba_module(module: ModuleType) -> bool:
    return replace_none_enum_value(module.MambaAttentionBackendEnum, "CUSTOM", "")


def patch_vllm_mamba_enum() -> bool:
    from vllm.v1.attention.backends import registry

    # vLLM 0.22.1 defines five string values and CUSTOM=None. msgspec refuses
    # to decode any value of an enum with mixed string/None member types.
    return patch_vllm_mamba_module(registry)


def wrap_async_weight_update(async_llm_type: type[Any]) -> bool:
    original = async_llm_type.update_weights
    if getattr(original, "_oellm_transactional_update", False):
        return False

    @wraps(original)
    async def transactional_update(self: Any, request: Any) -> Any:
        await self.start_weight_update(is_checkpoint_format=True)
        try:
            return await original(self, request)
        finally:
            await self.finish_weight_update()

    transactional_update._oellm_transactional_update = True
    async_llm_type.update_weights = transactional_update
    return True


def patch_vllm_weight_module(module: ModuleType) -> bool:
    return wrap_async_weight_update(module.AsyncLLM)


def patch_vllm_weight_update() -> bool:
    from vllm.v1.engine import async_llm

    # Pinned TMAX sends one packed update per broadcast. vLLM 0.22.1 made the
    # surrounding start/finish transaction mandatory after that TMAX revision.
    return patch_vllm_weight_module(async_llm)
