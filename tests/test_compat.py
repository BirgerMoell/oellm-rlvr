import asyncio
import importlib
import os
import sys
import threading
from enum import Enum
from types import ModuleType, SimpleNamespace

from oellm_rlvr.compat import (
    install_post_import_patch,
    patch_math_equivalence_module,
    patch_open_instruct_grpo_module,
    patch_open_instruct_vllm_module,
    patch_vllm_mamba_module,
    patch_vllm_weight_transfer_factory,
    replace_none_enum_value,
    wrap_async_weight_update,
    wrap_open_instruct_rocm_visibility,
    wrap_open_instruct_streaming_config,
    wrap_swerl_create_backend,
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


def test_math_equivalence_timeout_is_safe_in_executor_thread() -> None:
    module = ModuleType("fake_math_utils")

    class SignalTimeout:
        def __init__(self, seconds=1):
            self.seconds = seconds

        def __enter__(self):
            if threading.current_thread() is not threading.main_thread():
                raise ValueError("signal only works in main thread")
            return self

        def __exit__(self, *_args):
            return None

    def is_equiv(left, right):
        with module.timeout(seconds=5):
            return left == right

    module.timeout = SignalTimeout
    module.is_equiv = is_equiv
    assert patch_math_equivalence_module(module) is True
    assert patch_math_equivalence_module(module) is False
    assert asyncio.run(asyncio.to_thread(module.is_equiv, "-73/20", "-73/20")) is True
    assert module.is_equiv("27", "27") is True
    assert module.is_equiv("x" * 513, "x" * 513) is False


def test_swerl_plain_apptainer_drops_prepared_kwargs() -> None:
    module = ModuleType("fake_swerl")

    def create_backend(backend_type, **kwargs):
        return backend_type, kwargs

    module.create_backend = create_backend
    assert wrap_swerl_create_backend(module) is True
    assert wrap_swerl_create_backend(module) is False
    backend_type, kwargs = module.create_backend(
        "apptainer",
        image="python.sif",
        prepared_root="/prepared",
        prepared_copy_method="reflink",
    )
    assert backend_type == "apptainer"
    assert kwargs == {"image": "python.sif"}


def test_swerl_prepared_apptainer_keeps_prepared_kwargs() -> None:
    module = ModuleType("fake_swerl")

    def create_backend(backend_type, **kwargs):
        return backend_type, kwargs

    module.create_backend = create_backend
    wrap_swerl_create_backend(module)
    _, kwargs = module.create_backend("prepared_apptainer", prepared_root="/prepared")
    assert kwargs == {"prepared_root": "/prepared"}


def test_hierarchical_factory_replaces_nccl_receiver(monkeypatch) -> None:
    class FakeHierarchicalEngine:
        pass

    hierarchical_module = ModuleType("oellm_rlvr.hierarchical_weight_transfer")
    hierarchical_module.HierarchicalNCCLWeightTransferEngine = FakeHierarchicalEngine
    monkeypatch.setitem(sys.modules, hierarchical_module.__name__, hierarchical_module)

    class FakeFactory:
        pass

    FakeFactory._registry = {"nccl": lambda: object}

    module = ModuleType("fake_vllm_factory")
    module.WeightTransferEngineFactory = FakeFactory
    assert patch_vllm_weight_transfer_factory(module) is True
    assert patch_vllm_weight_transfer_factory(module) is False
    assert FakeFactory._registry["nccl"]() is FakeHierarchicalEngine


def test_rollout_actor_gets_hierarchical_endpoint_methods(monkeypatch) -> None:
    class FakeActor:
        pass

    module = ModuleType("fake_vllm_utils")
    module.LLMRayActor = FakeActor
    module.ray = SimpleNamespace(
        _private=SimpleNamespace(services=SimpleNamespace(get_node_ip_address=lambda: "[10.0.0.8]"))
    )
    monkeypatch.setattr("oellm_rlvr.compat._reserve_local_ports", lambda count: list(range(24000, 24000 + count)))
    assert patch_open_instruct_vllm_module(module) is True
    assert patch_open_instruct_vllm_module(module) is False
    actor = FakeActor()
    endpoint = actor.oellm_hierarchical_endpoint(3)
    assert endpoint["address"] == "10.0.0.8"
    assert len(endpoint["ports"]) == 3
    assert len(set(endpoint["ports"])) == 3


def test_rocm_rollout_actor_resynchronizes_vllm_cuda_alias(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeActor:
        def _setup_gpu_visibility(self, *args, **_kwargs):
            calls.append(args)
            os.environ["HIP_VISIBLE_DEVICES"] = "6"

    module = ModuleType("fake_vllm_utils")
    module.LLMRayActor = FakeActor
    module.torch = SimpleNamespace(version=SimpleNamespace(hip="7.0"))
    module.logger = SimpleNamespace(info=lambda *_args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6")

    assert wrap_open_instruct_rocm_visibility(module) is True
    assert wrap_open_instruct_rocm_visibility(module) is False
    FakeActor()._setup_gpu_visibility(False, "uni")

    assert calls == [(False, "uni")]
    assert os.environ["HIP_VISIBLE_DEVICES"] == "6"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_cuda_rollout_actor_keeps_cuda_visibility(monkeypatch) -> None:
    class FakeActor:
        def _setup_gpu_visibility(self, *_args, **_kwargs):
            os.environ["HIP_VISIBLE_DEVICES"] = "2"

    module = ModuleType("fake_vllm_utils")
    module.LLMRayActor = FakeActor
    module.torch = SimpleNamespace(version=SimpleNamespace(hip=None))
    module.logger = SimpleNamespace(info=lambda *_args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    wrap_open_instruct_rocm_visibility(module)
    FakeActor()._setup_gpu_visibility(False, "uni")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_nonzero_trainer_rank_joins_hierarchical_barrier() -> None:
    class FakePolicyActor:
        def from_pretrained(self):
            return "loaded"

        def setup_model_update_group(self, vllm_engines):
            raise AssertionError("native setup should not run")

    barrier_calls: list[bool] = []
    module = ModuleType("fake_grpo_fast")
    module.PolicyTrainerRayProcess = FakePolicyActor
    module.torch = SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: barrier_calls.append(True)))
    assert patch_open_instruct_grpo_module(module) is True
    assert patch_open_instruct_grpo_module(module) is False

    actor = FakePolicyActor()
    actor.args = SimpleNamespace(single_gpu_mode=False)
    actor.vllm_config = SimpleNamespace(vllm_tensor_parallel_size=1)
    actor.rank = 1
    actor.vllm_engines = None
    actor.model_update_group = "unset"
    actor.setup_model_update_group([object(), object()])
    assert actor.vllm_engines is not None
    assert actor.model_update_group is None
    assert barrier_calls == [True]


def test_hierarchical_setup_patches_ray_decorated_worker_class() -> None:
    class OriginalPolicyActor:
        def from_pretrained(self):
            return "loaded"

        def setup_model_update_group(self, _vllm_engines):
            raise AssertionError("native setup should not run")

    class RayModifiedPolicyActor(OriginalPolicyActor):
        __ray_actor_class__ = OriginalPolicyActor

    decorated_actor = SimpleNamespace(
        __ray_metadata__=SimpleNamespace(modified_class=RayModifiedPolicyActor)
    )
    barrier_calls: list[bool] = []
    module = ModuleType("fake_decorated_grpo_fast")
    module.PolicyTrainerRayProcess = decorated_actor
    module.torch = SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: barrier_calls.append(True)))

    assert patch_open_instruct_grpo_module(module) is True
    assert RayModifiedPolicyActor.setup_model_update_group._oellm_hierarchical_setup is True
    assert not getattr(OriginalPolicyActor.setup_model_update_group, "_oellm_hierarchical_setup", False)

    actor = RayModifiedPolicyActor()
    actor.args = SimpleNamespace(single_gpu_mode=False)
    actor.vllm_config = SimpleNamespace(vllm_tensor_parallel_size=1)
    actor.rank = 1
    actor.setup_model_update_group([object(), object()])
    assert barrier_calls == [True]


def test_trainer_actor_restores_streaming_config_global(monkeypatch) -> None:
    class FakePolicyActor:
        def from_pretrained(self):
            return globals()["streaming_config"]

    function_globals = FakePolicyActor.from_pretrained.__globals__
    monkeypatch.delitem(function_globals, "streaming_config", raising=False)
    assert wrap_open_instruct_streaming_config(FakePolicyActor) is True
    assert wrap_open_instruct_streaming_config(FakePolicyActor) is False

    actor = FakePolicyActor()
    actor.streaming_config = object()
    assert actor.from_pretrained() is actor.streaming_config
    assert function_globals["streaming_config"] is actor.streaming_config


def test_rank_zero_builds_one_trainer_relay_group_and_leaf_links() -> None:
    class FakeRef:
        def __init__(self, value):
            self.value = value

    class RemoteMethod:
        def __init__(self, fn):
            self.fn = fn

        def remote(self, *args):
            return FakeRef(self.fn(*args))

    init_requests: list[dict[str, object]] = []

    class FakeEngine:
        def __init__(self, index: int):
            self.oellm_node_ip = RemoteMethod(lambda: "10.0.0.8")
            self.oellm_hierarchical_endpoint = RemoteMethod(
                lambda count: {"address": "10.0.0.8", "ports": list(range(25001, 25001 + count))}
            )
            self.init_weight_transfer_engine = RemoteMethod(lambda request: init_requests.append(request.init_info))

    class FakePolicyActor:
        def from_pretrained(self):
            return "loaded"

        def setup_model_update_group(self, vllm_engines):
            raise AssertionError("native setup should not run")

        @staticmethod
        def get_current_node_ip():
            return "10.0.0.9"

    class FakeRequest:
        def __init__(self, init_info):
            self.init_info = init_info

    trainer_inits: list[dict[str, object]] = []
    progress_refs: list[object] = []
    barrier_calls: list[bool] = []
    module = ModuleType("fake_grpo_fast")
    module.PolicyTrainerRayProcess = FakePolicyActor
    module.WeightTransferInitRequest = FakeRequest
    module.ray = SimpleNamespace(
        get=lambda refs: [ref.value for ref in refs] if isinstance(refs, list) else refs.value
    )
    module.utils = SimpleNamespace(find_free_port=lambda: 25000)
    module.logger = SimpleNamespace(info=lambda *_args: None)
    module.NCCLWeightTransferEngine = SimpleNamespace(
        trainer_init=lambda info: trainer_inits.append(info) or "trainer-group"
    )
    module.ray_get_with_progress = lambda refs, **_kwargs: progress_refs.extend(refs)
    module.torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda _device: None),
        distributed=SimpleNamespace(barrier=lambda: barrier_calls.append(True)),
    )
    patch_open_instruct_grpo_module(module)

    actor = FakePolicyActor()
    actor.args = SimpleNamespace(single_gpu_mode=False)
    actor.vllm_config = SimpleNamespace(vllm_tensor_parallel_size=1)
    actor.rank = 0
    actor.local_rank = 0
    engines = [FakeEngine(index) for index in range(3)]
    actor.setup_model_update_group(engines)

    assert trainer_inits == [{"master_address": "10.0.0.9", "master_port": 25000, "world_size": 2}]
    assert init_requests == [
        {
            "role": "relay",
            "upstream": {"master_address": "10.0.0.9", "master_port": 25000},
            "downstream": [
                {"master_address": "10.0.0.8", "master_port": 25001},
                {"master_address": "10.0.0.8", "master_port": 25002},
            ],
        },
        {
            "role": "leaf",
            "upstream": {"master_address": "10.0.0.8", "master_port": 25001},
            "downstream": [],
        },
        {
            "role": "leaf",
            "upstream": {"master_address": "10.0.0.8", "master_port": 25002},
            "downstream": [],
        },
    ]
    assert actor.model_update_group == "trainer-group"
    assert len(progress_refs) == 3
    assert barrier_calls == [True]
