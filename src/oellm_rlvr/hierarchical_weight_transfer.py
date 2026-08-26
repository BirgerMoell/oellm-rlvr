"""Hierarchical NCCL/RCCL weight transfer for one-node rollout fleets.

The trainer and the first rollout engine form the only cross-node
communicator.  The first engine relays every received tensor through one
independent two-rank communicator per local leaf engine.  Keeping every
communicator at world size two avoids the multi-process P2P/IPC group setup
that stalls on LUMI while ensuring that model weights cross Slingshot once.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from vllm.distributed.weight_transfer.base import WeightTransferInitInfo
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import unpack_tensor

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalWeightTransferInitInfo(WeightTransferInitInfo):
    """A receiver's upstream link and, for the relay, its downstream links."""

    role: Literal["relay", "leaf"]
    upstream: dict[str, Any]
    downstream: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in {"relay", "leaf"}:
            raise ValueError(f"unsupported hierarchical receiver role: {self.role}")
        required = {"master_address", "master_port"}
        if not required.issubset(self.upstream):
            raise ValueError("upstream requires master_address and master_port")
        if self.role == "leaf" and self.downstream:
            raise ValueError("leaf receivers cannot have downstream links")
        for link in self.downstream:
            if not required.issubset(link):
                raise ValueError("every downstream link requires master_address and master_port")


class HierarchicalNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """Receive once from the trainer and fan weights out from a local relay."""

    init_info_cls = HierarchicalWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo

    def __init__(self, config: Any, parallel_config: Any, model: torch.nn.Module) -> None:
        super().__init__(config, parallel_config, model)
        self.role: Literal["relay", "leaf"] | None = None
        self.upstream_group: Any | None = None
        self.downstream_groups: list[Any] = []

    @staticmethod
    def _group(link: dict[str, Any], *, rank: int) -> Any:
        return NCCLWeightTransferEngine._stateless_init_process_group(
            link["master_address"],
            int(link["master_port"]),
            rank,
            2,
            device=torch.accelerator.current_device_index(),
        )

    def init_transfer_engine(self, init_info: HierarchicalWeightTransferInitInfo) -> None:
        if self.parallel_config.world_size != 1:
            raise ValueError("hierarchical weight transfer currently requires tensor_parallel_size=1")
        self.role = init_info.role
        # The relay first joins the trainer so trainer_init can complete, then
        # joins its already-waiting leaf peers one pair at a time.
        self.upstream_group = self._group(init_info.upstream, rank=1)
        if self.role == "relay":
            self.downstream_groups = [self._group(link, rank=0) for link in init_info.downstream]
        logger.info(
            "Initialized hierarchical weight receiver role=%s downstream_links=%d",
            self.role,
            len(self.downstream_groups),
        )

    def _broadcast_from_upstream(self, tensor: torch.Tensor) -> None:
        if self.upstream_group is None:
            raise RuntimeError("hierarchical weight transfer was not initialized")
        stream = torch.cuda.current_stream()
        self.upstream_group.broadcast(tensor, src=0, stream=stream)
        for group in self.downstream_groups:
            group.broadcast(tensor, src=0, stream=stream)

    @staticmethod
    def _metadata_chunks(
        update_info: NCCLWeightTransferUpdateInfo,
    ) -> list[list[tuple[str, list[int], torch.dtype, int]]]:
        chunks: list[list[tuple[str, list[int], torch.dtype, int]]] = []
        current: list[tuple[str, list[int], torch.dtype, int]] = []
        current_size = 0
        for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
            dtype = getattr(torch, dtype_name)
            size = math.prod(shape) * dtype.itemsize
            current.append((name, shape, dtype, size))
            current_size += size
            # Match vLLM 0.22.1's packed producer/consumer boundary: the
            # tensor that crosses the target size remains in this chunk.
            if current_size > update_info.packed_buffer_size_bytes:
                chunks.append(current)
                current = []
                current_size = 0
        if current:
            chunks.append(current)
        return chunks

    def _receive_packed(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Any,
    ) -> None:
        # A single conservative stream keeps upstream receive, local fanout,
        # and model loading ordered. The transfer path is startup-critical and
        # correctness is more important than overlapping two 1 GiB buffers.
        for metadata in self._metadata_chunks(update_info):
            names, shapes, dtypes, sizes = zip(*metadata)
            packed = torch.empty(sum(sizes), dtype=torch.uint8, device="cuda")
            self._broadcast_from_upstream(packed)
            load_weights(
                unpack_tensor(
                    packed,
                    list(names),
                    list(shapes),
                    list(dtypes),
                    list(sizes),
                )
            )

    def _receive_unpacked(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Any,
    ) -> None:
        for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
            weight = torch.empty(shape, dtype=getattr(torch, dtype_name), device="cuda")
            self._broadcast_from_upstream(weight)
            load_weights([(name, weight)])

    def receive_weights(self, update_info: NCCLWeightTransferUpdateInfo, load_weights: Any) -> None:
        if update_info.packed:
            self._receive_packed(update_info, load_weights)
        else:
            self._receive_unpacked(update_info, load_weights)

    def shutdown(self) -> None:
        self.upstream_group = None
        self.downstream_groups = []
        self.model_update_group = None
