from enum import Enum

from oellm_rlvr.compat import replace_none_enum_value


class MixedEnum(Enum):
    VALUE = "value"
    CUSTOM = None


def test_replace_none_enum_value_makes_values_homogeneous() -> None:
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is True
    assert [member.value for member in MixedEnum] == ["value", ""]
    assert MixedEnum("") is MixedEnum.CUSTOM
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is False
