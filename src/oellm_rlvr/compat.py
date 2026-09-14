from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import re
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from enum import Enum
from functools import wraps
from operator import index
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

    def get_code(self, fullname: str) -> Any:
        """Preserve loaders used by ``python -m`` and vLLM inspectors."""
        get_code = getattr(self.original, "get_code", None)
        if get_code is None:
            raise ImportError(f"loader for {fullname} does not provide get_code")
        return get_code(fullname)


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


def patch_vllm_qwen35_text_registry(module: ModuleType) -> bool:
    """Register vLLM's native text-only Qwen3.5 implementation.

    The LUMI vLLM 0.22.1 build ships ``Qwen3_5ForCausalLM`` but omits it from
    the model registry and omits its hybrid-cache and M-RoPE interfaces. Its
    architecture fallback selects the multimodal conditional-generation class,
    which expects ``vision_config``; registering the unmarked text class alone
    then leaves ``mamba_block_size`` unset during KV-cache construction.
    """
    registry = module.ModelRegistry
    qwen_module = importlib.import_module("vllm.model_executor.models.qwen3_5")
    model_type = qwen_module.Qwen3_5ForCausalLM
    conditional_type = qwen_module.Qwen3_5ForConditionalGeneration

    # vLLM 0.22.1 also omits the IsHybrid marker and GDN state helpers from
    # its text-only class.  Without them VllmConfig skips
    # HybridAttentionMambaModelConfig, leaving mamba_block_size unset and
    # aborting KV-cache construction after all weights have loaded.  The
    # conditional class implements the same text backbone and already carries
    # the correct helpers, so copy only that interface onto the native text
    # class while preserving its class name and text-only weight loader.
    model_type.is_hybrid = True
    for method_name in (
        "get_mamba_state_dtype_from_config",
        "get_mamba_state_shape_from_config",
        "get_mamba_state_copy_func",
    ):
        if method_name not in model_type.__dict__:
            setattr(model_type, method_name, conditional_type.__dict__[method_name])

    # The Qwen3.5 text config still uses M-RoPE.  Text tokens do not require
    # the multimodal class' image/video grid logic: upstream vLLM implements
    # the text-only path by broadcasting the ordinary positions over T/H/W.
    # Backport that exact behavior instead of copying a method that expects
    # ``vision_config`` from the conditional-generation class.
    model_type.supports_mrope = True
    if "get_mrope_input_positions" not in model_type.__dict__:

        def get_mrope_input_positions(
            self: Any,
            input_tokens: list[int],
            mm_features: list[object],
        ) -> tuple[Any, int]:
            del self, mm_features
            import torch

            positions = torch.arange(len(input_tokens), dtype=torch.long)
            return positions.unsqueeze(0).expand(3, -1), 0

        model_type.get_mrope_input_positions = get_mrope_input_positions

    existing = registry.models.get("Qwen3_5ForCausalLM")
    if existing is not None and getattr(existing, "model_cls", None) is model_type:
        return False
    # Register the concrete class so vLLM does not invoke its lazy model-info
    # inspector in a subprocess before the compute actor has initialized.
    registry.register_model("Qwen3_5ForCausalLM", model_type)
    return True


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


def wrap_open_instruct_rocm_visibility(module: ModuleType) -> bool:
    """Keep vLLM's CUDA alias aligned with Ray's per-actor HIP mask.

    Ray 2.54 assigns AMD GPUs through ``HIP_VISIBLE_DEVICES``. vLLM 0.22.1
    aliases that value to ``CUDA_VISIBLE_DEVICES`` when its ROCm platform is
    imported, which happens before Ray applies an actor-specific mask. Without
    resynchronizing the alias, an EngineCore child can inherit a stale mask and
    report zero visible devices.
    """
    actor_type = module.LLMRayActor
    original = actor_type._setup_gpu_visibility
    if getattr(original, "_oellm_rocm_visibility", False):
        return False

    @wraps(original)
    def compatible_visibility(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if getattr(module.torch.version, "hip", None):
            hip_devices = os.environ.get("HIP_VISIBLE_DEVICES")
            if hip_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = hip_devices
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            module.logger.info(
                "Synchronized vLLM ROCm visibility: HIP_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s",
                hip_devices,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        return result

    compatible_visibility._oellm_rocm_visibility = True
    actor_type._setup_gpu_visibility = compatible_visibility
    return True


def _ray_actor_target(actor_type: Any) -> Any:
    metadata = getattr(actor_type, "__ray_metadata__", None)
    # Ray 2.54 derives this internal subclass from the user class. The function
    # descriptor is named from ``__ray_actor_class__``, but ``ActorClass._remote``
    # serializes ``metadata.modified_class`` in ``export_actor_class``. Patch the
    # serialized class, not merely the original class used for its descriptor.
    return getattr(metadata, "modified_class", actor_type)


def wrap_open_instruct_streaming_config(actor_type: Any) -> bool:
    """Restore the script global expected by the pinned trainer actor.

    The pinned ``grpo_fast.py`` footer assigns ``streaming_config`` at module
    scope before calling ``main``.  Importing that file canonically is required
    for the hierarchical Ray patch, but then the parsed config is only a local
    ``main`` argument.  Two actor methods still read the old module global.
    Seed it inside each Ray worker from the config already stored on the actor.
    """
    original = actor_type.from_pretrained
    if getattr(original, "_oellm_streaming_config_global", False):
        return False

    @wraps(original)
    def compatible_from_pretrained(self: Any, *args: Any, **kwargs: Any) -> Any:
        original.__globals__["streaming_config"] = self.streaming_config
        return original(self, *args, **kwargs)

    compatible_from_pretrained._oellm_streaming_config_global = True
    actor_type.from_pretrained = compatible_from_pretrained
    return True


def patch_open_instruct_grpo_module(module: ModuleType) -> bool:
    """Use a trainer→relay→leaf topology for native vLLM weight sync."""
    actor_type = _ray_actor_target(module.PolicyTrainerRayProcess)
    streaming_config_patched = wrap_open_instruct_streaming_config(actor_type)
    original = actor_type.setup_model_update_group
    if getattr(original, "_oellm_hierarchical_setup", False):
        return streaming_config_patched

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


def patch_tmax_multilingual_math_verifier(module: ModuleType) -> bool:
    """Register a conjunctive exact-answer, language, and form verifier.

    The project dataset carries a JSON label containing the ordinary math
    answer and the requested ISO-639-1 language. A response earns one only if
    every condition passes; language or formatting can never rescue an
    incorrect answer.
    """
    if getattr(module, "_oellm_multilingual_math_verifier", False):
        return False

    from oellm_rlvr.language_audit import (
        _build_detector,
        _detect,
        reasoning_prose,
        target_language_matches,
    )

    class MultilingualMathVerifier(module.VerifierFunction):
        _detector: Any = None
        _detector_lock = threading.Lock()

        def __init__(self, verifier_config: Any = None) -> None:
            super().__init__("multilingual_math", verifier_config=verifier_config, weight=1.0)
            self.math_verifier = module.MathVerifier(verifier_config)

        @classmethod
        def detector(cls) -> Any:
            if cls._detector is None:
                with cls._detector_lock:
                    if cls._detector is None:
                        cls._detector = _build_detector()
            return cls._detector

        def __call__(
            self,
            tokenized_prediction: list[int],
            prediction: str,
            label: str,
            query: str | None = None,
            rollout_state: dict | None = None,
        ) -> Any:
            del query
            try:
                contract = json.loads(label)
                answer = str(contract["answer"])
                target = str(contract["target_language"])
            except (KeyError, TypeError, json.JSONDecodeError):
                return module.VerificationResult(score=0.0, reasoning="invalid multilingual math label")

            math_result = self.math_verifier(
                tokenized_prediction,
                prediction,
                answer,
                rollout_state=rollout_state,
            )
            if float(math_result.score) != 1.0:
                return module.VerificationResult(score=0.0, reasoning="incorrect answer")

            boxes = list(re.finditer(r"\\boxed\{[^{}]+\}", prediction))
            open_count = len(re.findall(r"<think>", prediction, flags=re.IGNORECASE))
            closes = list(re.finditer(r"</think>", prediction, flags=re.IGNORECASE))
            # Qwen3.5's native generation prompt supplies the opening tag, so
            # the generated suffix normally contains only the close. Also
            # accept a fully explicit pair for other compatible templates.
            channel_ok = (
                len(boxes) == 1
                and len(closes) == 1
                and closes[0].start() < boxes[0].start()
                and open_count in {0, 1}
            )
            if not channel_ok:
                return module.VerificationResult(score=0.0, reasoning="invalid think/box format")

            prose = reasoning_prose(prediction)
            detected, confidence = _detect(self.detector(), prose)
            language_ok = target_language_matches(target, detected, confidence)
            diagnostic = json.dumps(
                {
                    "target": target,
                    "detected": detected,
                    "confidence": confidence,
                    "language_ok": language_ok,
                },
                sort_keys=True,
            )
            return module.VerificationResult(score=float(language_ok), reasoning=diagnostic)

    MultilingualMathVerifier.__name__ = "MultilingualMathVerifier"
    module.MultilingualMathVerifier = MultilingualMathVerifier
    module._oellm_multilingual_math_verifier = True
    return True


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    getter = getattr(value, "get", None)
    if callable(getter):
        candidate = getter(name, None)
        if candidate is not None:
            return candidate
    candidate = getattr(value, name, None)
    if candidate is not None:
        return candidate
    # Pydantic/LiteLLM can retain OpenAI extension fields in model extras:
    # model_dump() exposes them even though ``get`` and attribute lookup do
    # not.  The real vLLM 0.22.1 response uses this layout for token IDs.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        dumped = dump(exclude_none=False)
        if isinstance(dumped, dict):
            return dumped.get(name)
    return None


def _integer_token_ids(value: Any) -> list[int] | None:
    """Normalize only lossless integer token-ID representations."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        return None

    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool):
            return None
        if isinstance(token, str):
            if not token.isdecimal():
                return None
            normalized = int(token)
        else:
            try:
                normalized = index(token)
            except TypeError:
                return None
        if normalized < 0:
            return None
        tokens.append(normalized)
    return tokens


def _token_value_summary(value: Any) -> str:
    """Describe a token-ID candidate without logging token content."""
    if value is None:
        return "None"
    try:
        length: int | str = len(value)
    except TypeError:
        length = "?"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        element_types = sorted({type(element).__name__ for element in value[:8]})
    else:
        element_types = []
    return f"{type(value).__name__}(len={length}, element_types={element_types})"


def wrap_skyrl_harbor_direct_single_engine(generator_type: type[Any]) -> bool:
    """Bypass SkyRL's HTTP router when Harbor has exactly one vLLM engine.

    This is a narrow compatibility path for the one-engine qualification
    canary. Harbor requires vLLM-specific response fields for RL, while a
    regular OpenAI proxy may preserve the response schema but replace those
    extension values with ``None``. Multi-engine campaigns must retain a
    session-aware router and are deliberately rejected by this wrapper.
    """
    original = generator_type.__init__
    if getattr(original, "_oellm_direct_single_engine", False):
        return False

    @wraps(original)
    def direct_single_engine(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        inference_client = kwargs.get("inference_engine_client")
        if inference_client is None and len(args) >= 3:
            inference_client = args[2]
        server_urls = getattr(inference_client, "server_urls", None)
        if not isinstance(server_urls, list) or len(server_urls) != 1:
            raise RuntimeError(
                "OELLM_HARBOR_DIRECT_SINGLE_ENGINE requires exactly one vLLM server URL; "
                f"got {server_urls!r}"
            )
        direct_url = str(server_urls[0]).rstrip("/")
        self.base_url = direct_url
        self._harbor_trial_config_template["agent"]["kwargs"]["api_base"] = f"{direct_url}/v1"
        module_logger = sys.modules[generator_type.__module__].logger
        module_logger.info(f"Harbor direct single-engine data plane enabled: {direct_url} (router bypassed)")

    direct_single_engine._oellm_direct_single_engine = True
    generator_type.__init__ = direct_single_engine
    return True


def patch_skyrl_harbor_generator_module(module: ModuleType) -> bool:
    return wrap_skyrl_harbor_direct_single_engine(module.HarborGenerator)


def wrap_harbor_vllm_token_extraction(llm_type: type[Any]) -> bool:
    """Preserve vLLM token IDs across LiteLLM response-layout variants.

    LiteLLM versions have represented OpenAI-compatible extension fields as
    direct attributes, choice provider fields, or message provider fields.
    Harbor v0.22.0 checks only one of those layouts. Keep the upstream method
    as the first choice, then search the other lossless locations.
    """
    original = llm_type._extract_token_ids
    if getattr(original, "_oellm_vllm_token_layouts", False):
        return False

    @wraps(original)
    def compatible_extract(self: Any, response: Any) -> tuple[list[int] | None, list[int] | None]:
        prompt_ids, completion_ids = original(self, response)
        prompt_ids = _integer_token_ids(prompt_ids)
        completion_ids = _integer_token_ids(completion_ids)

        response_provider = _field(response, "provider_specific_fields") or {}
        prompt_candidates = (
            _field(response, "prompt_token_ids"),
            _field(response_provider, "prompt_token_ids"),
        )
        if prompt_ids is None:
            prompt_ids = next(
                (tokens for value in prompt_candidates if (tokens := _integer_token_ids(value)) is not None),
                None,
            )

        choices = _field(response, "choices") or []
        choice = choices[0] if choices else None
        message = _field(choice, "message") if choice is not None else None
        candidates = (
            _field(choice, "token_ids"),
            _field(_field(choice, "provider_specific_fields") or {}, "token_ids"),
            _field(message, "token_ids"),
            _field(_field(message, "provider_specific_fields") or {}, "token_ids"),
        )
        if completion_ids is None:
            completion_ids = next(
                (tokens for value in candidates if (tokens := _integer_token_ids(value)) is not None),
                None,
            )

        if prompt_ids is None or completion_ids is None:
            def keys(value: Any) -> list[str]:
                if isinstance(value, dict):
                    return sorted(str(key) for key in value)
                dump = getattr(value, "model_dump", None)
                if callable(dump):
                    dumped = dump(exclude_none=False)
                    if isinstance(dumped, dict):
                        return sorted(str(key) for key in dumped)
                return []

            self._logger.warning(
                "vLLM rollout token IDs missing after LiteLLM parsing: "
                "prompt=%s completion=%s response_keys=%s response_provider_keys=%s "
                "choice_keys=%s choice_provider_keys=%s message_keys=%s message_provider_keys=%s "
                "prompt_candidates=%s completion_candidates=%s",
                prompt_ids is not None,
                completion_ids is not None,
                keys(response),
                keys(response_provider),
                keys(choice),
                keys(_field(choice, "provider_specific_fields") or {}),
                keys(message),
                keys(_field(message, "provider_specific_fields") or {}),
                [_token_value_summary(value) for value in prompt_candidates],
                [_token_value_summary(value) for value in candidates],
            )
        return prompt_ids, completion_ids

    compatible_extract._oellm_vllm_token_layouts = True
    llm_type._extract_token_ids = compatible_extract
    return True


def patch_harbor_litellm_module(module: ModuleType) -> bool:
    return wrap_harbor_vllm_token_extraction(module.LiteLLM)


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
