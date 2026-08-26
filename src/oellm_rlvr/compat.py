from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
from collections.abc import Callable
from enum import Enum
from functools import wraps
from types import ModuleType
from typing import Any

ModulePatch = Callable[[ModuleType], object]
PREPARED_ONLY_BACKEND_KWARGS = {
    "prepared_root",
    "prepared_cache_dir",
    "prepared_state_cache_key",
    "prepared_scratch_root",
    "prepared_copy_method",
}


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


def patch_math_equivalence_module(module: ModuleType) -> bool:
    """Keep the pinned math verifier's signal timeout out of executor threads."""
    original_is_equiv = module.is_equiv
    if getattr(original_is_equiv, "_oellm_thread_safe_math_equiv", False):
        return False
    original_timeout = module.timeout

    class ThreadAwareTimeout(original_timeout):
        def __enter__(self: Any) -> Any:
            self._oellm_timeout_active = threading.current_thread() is threading.main_thread()
            if self._oellm_timeout_active:
                return super().__enter__()
            return self

        def __exit__(self: Any, *args: object) -> Any:
            if self._oellm_timeout_active:
                return super().__exit__(*args)
            return None

    @wraps(original_is_equiv)
    def bounded_is_equiv(x1: str, x2: str) -> bool:
        # Executor threads cannot install SIGALRM handlers. Keep symbolic work
        # bounded by rejecting implausibly large extracted answers before the
        # thread-aware timeout delegates to the pinned implementation.
        if not isinstance(x1, str) or not isinstance(x2, str) or max(len(x1), len(x2)) > 512:
            return False
        return bool(original_is_equiv(x1, x2))

    bounded_is_equiv._oellm_thread_safe_math_equiv = True
    module.timeout = ThreadAwareTimeout
    module.is_equiv = bounded_is_equiv
    return True


def wrap_swerl_create_backend(module: ModuleType) -> bool:
    """Keep prepared-Apptainer defaults away from the plain backend."""
    original = module.create_backend
    if getattr(original, "_oellm_filters_prepared_kwargs", False):
        return False

    @wraps(original)
    def compatible_create_backend(backend_type: str, *args: Any, **kwargs: Any) -> Any:
        if backend_type == "slurm_apptainer":
            from oellm_rlvr.slurm_sandbox import SlurmApptainerBackend

            return SlurmApptainerBackend(*args, **kwargs)
        if backend_type != "prepared_apptainer":
            kwargs = {key: value for key, value in kwargs.items() if key not in PREPARED_ONLY_BACKEND_KWARGS}
        return original(backend_type, *args, **kwargs)

    compatible_create_backend._oellm_filters_prepared_kwargs = True
    module.create_backend = compatible_create_backend
    return True
