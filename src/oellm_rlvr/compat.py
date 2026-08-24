from __future__ import annotations

from enum import Enum


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
