from __future__ import annotations

import importlib.abc
import importlib.machinery
import socket
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


def patch_vllm_weight_transfer_factory(module: ModuleType) -> bool:
    """Replace vLLM's NCCL receiver with the hierarchical receiver."""
    factory = module.WeightTransferEngineFactory
    if getattr(factory, "_oellm_hierarchical_nccl", False):
        return False
    from oellm_rlvr.hierarchical_weight_transfer import HierarchicalNCCLWeightTransferEngine

    factory._registry["nccl"] = lambda: HierarchicalNCCLWeightTransferEngine
    factory._oellm_hierarchical_nccl = True
    return True


def _reserve_local_ports(count: int) -> list[int]:
    """Select distinct local TCP ports while holding every socket open."""
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("", 0))
            sockets.append(listener)
        return [int(listener.getsockname()[1]) for listener in sockets]
    finally:
        for listener in sockets:
            listener.close()


def patch_open_instruct_vllm_module(module: ModuleType) -> bool:
    """Expose rollout-node endpoint discovery before Ray wraps LLMRayActor."""
    actor_type = module.LLMRayActor
    if getattr(actor_type, "_oellm_hierarchical_endpoints", False):
        return False

    def oellm_node_ip(self: Any) -> str:
        return module.ray._private.services.get_node_ip_address().strip("[]")

    def oellm_hierarchical_endpoint(self: Any, leaf_count: int) -> dict[str, object]:
        return {"address": oellm_node_ip(self), "ports": _reserve_local_ports(leaf_count)}

    actor_type.oellm_node_ip = oellm_node_ip
    actor_type.oellm_hierarchical_endpoint = oellm_hierarchical_endpoint
    actor_type._oellm_hierarchical_endpoints = True
    return True


def _ray_actor_target(actor_type: Any) -> Any:
    metadata = getattr(actor_type, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", actor_type)


def patch_open_instruct_grpo_module(module: ModuleType) -> bool:
    """Use a trainer→relay→leaf topology for native vLLM weight sync."""
    actor_type = _ray_actor_target(module.PolicyTrainerRayProcess)
    original = actor_type.setup_model_update_group
    if getattr(original, "_oellm_hierarchical_setup", False):
        return False

    @wraps(original)
    def hierarchical_setup(self: Any, vllm_engines: list[Any]) -> None:
        if self.args.single_gpu_mode or len(vllm_engines) < 2:
            return original(self, vllm_engines)
        if self.vllm_config.vllm_tensor_parallel_size != 1:
            raise ValueError("hierarchical weight transfer currently requires tensor_parallel_size=1")

        self.vllm_engines = vllm_engines
        self.model_update_group = None
        if self.rank == 0:
            engine_addresses = module.ray.get([engine.oellm_node_ip.remote() for engine in vllm_engines])
            if len(set(engine_addresses)) != 1:
                raise RuntimeError(
                    "hierarchical weight transfer requires all rollout engines on one node; "
                    f"got {engine_addresses}"
                )
            endpoint = module.ray.get(
                vllm_engines[0].oellm_hierarchical_endpoint.remote(len(vllm_engines) - 1)
            )
            relay_address = str(endpoint["address"])
            leaf_ports = [int(port) for port in endpoint["ports"]]
            trainer_address = self.get_current_node_ip()
            trainer_port = module.utils.find_free_port()
            module.logger.info(
                "Hierarchical weight transfer: trainer=%s:%d relay=%s leaves=%d",
                trainer_address,
                trainer_port,
                relay_address,
                len(vllm_engines) - 1,
            )
            trainer_link = {"master_address": trainer_address, "master_port": trainer_port}
            downstream = [
                {"master_address": relay_address, "master_port": port} for port in leaf_ports
            ]
            init_infos = [
                {"role": "relay", "upstream": trainer_link, "downstream": downstream},
                *(
                    {"role": "leaf", "upstream": link, "downstream": []}
                    for link in downstream
                ),
            ]
            refs = [
                engine.init_weight_transfer_engine.remote(
                    module.WeightTransferInitRequest(init_info=init_info)
                )
                for engine, init_info in zip(vllm_engines, init_infos)
            ]
            module.torch.cuda.set_device(self.local_rank)
            self.model_update_group = module.NCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": trainer_address,
                    "master_port": trainer_port,
                    "world_size": 2,
                }
            )
            module.ray_get_with_progress(
                refs,
                desc="Initializing hierarchical vLLM weight transfer engines",
                timeout=600,
            )
        module.torch.distributed.barrier()

    hierarchical_setup._oellm_hierarchical_setup = True
    actor_type.setup_model_update_group = hierarchical_setup
    return True


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
