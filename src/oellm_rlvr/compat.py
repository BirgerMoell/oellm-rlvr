from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Any


def replace_none_enum_value(enum_type: type[Enum], member_name: str, replacement: str) -> bool:
    member = enum_type.__members__[member_name]
    if member.value is not None:
        return False
    enum_type._value2member_map_.pop(None, None)
    member._value_ = replacement
    enum_type._value2member_map_[replacement] = member
    return True


def patch_vllm_mamba_enum() -> bool:
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    # vLLM 0.22.1 defines five string values and CUSTOM=None. msgspec refuses
    # to decode any value of an enum with mixed string/None member types.
    return replace_none_enum_value(MambaAttentionBackendEnum, "CUSTOM", "")


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


def patch_vllm_weight_update() -> bool:
    from vllm.v1.engine.async_llm import AsyncLLM

    # Pinned TMAX sends one packed update per broadcast. vLLM 0.22.1 made the
    # surrounding start/finish transaction mandatory after that TMAX revision.
    return wrap_async_weight_update(AsyncLLM)
