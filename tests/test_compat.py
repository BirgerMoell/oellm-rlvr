import asyncio
from enum import Enum

from oellm_rlvr.compat import replace_none_enum_value, wrap_async_weight_update


class MixedEnum(Enum):
    VALUE = "value"
    CUSTOM = None


def test_replace_none_enum_value_makes_values_homogeneous() -> None:
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is True
    assert [member.value for member in MixedEnum] == ["value", ""]
    assert MixedEnum("") is MixedEnum.CUSTOM
    assert replace_none_enum_value(MixedEnum, "CUSTOM", "") is False


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
