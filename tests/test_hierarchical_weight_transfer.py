from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar


def _load_module(monkeypatch):
    torch = ModuleType("torch")
    torch.accelerator = SimpleNamespace(current_device_index=lambda: 0)
    torch.cuda = SimpleNamespace(current_stream=lambda: "stream")
    torch.uint8 = SimpleNamespace(itemsize=1)
    torch.bfloat16 = SimpleNamespace(itemsize=2)
    torch.empty = lambda shape, **kwargs: SimpleNamespace(shape=shape, kwargs=kwargs)

    base = ModuleType("vllm.distributed.weight_transfer.base")

    @dataclass
    class WeightTransferInitInfo:
        pass

    base.WeightTransferInitInfo = WeightTransferInitInfo

    nccl = ModuleType("vllm.distributed.weight_transfer.nccl_engine")

    @dataclass
    class NCCLWeightTransferUpdateInfo:
        names: list[str]
        dtype_names: list[str]
        shapes: list[list[int]]
        packed: bool = False
        packed_buffer_size_bytes: int = 1024
        packed_num_buffers: int = 2

    class FakeNCCLEngine:
        calls: ClassVar[list[tuple[object, ...]]] = []

        def __init__(self, config, parallel_config, model):
            self.config = config
            self.parallel_config = parallel_config
            self.model = model
            self.model_update_group = "native"

        @staticmethod
        def _stateless_init_process_group(address, port, rank, world_size, device):
            result = (address, port, rank, world_size, device)
            FakeNCCLEngine.calls.append(result)
            return result

    nccl.NCCLWeightTransferEngine = FakeNCCLEngine
    nccl.NCCLWeightTransferUpdateInfo = NCCLWeightTransferUpdateInfo

    packed = ModuleType("vllm.distributed.weight_transfer.packed_tensor")
    packed.unpack_tensor = lambda tensor, names, shapes, dtypes, sizes: list(zip(names, shapes))

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer", ModuleType("vllm.distributed.weight_transfer"))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, nccl.__name__, nccl)
    monkeypatch.setitem(sys.modules, packed.__name__, packed)

    path = Path(__file__).parents[1] / "src" / "oellm_rlvr" / "hierarchical_weight_transfer.py"
    spec = importlib.util.spec_from_file_location("oellm_hierarchical_weight_transfer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, FakeNCCLEngine, NCCLWeightTransferUpdateInfo


def test_relay_initializes_one_upstream_and_pairwise_downstreams(monkeypatch) -> None:
    module, fake_engine, _ = _load_module(monkeypatch)
    fake_engine.calls.clear()
    engine = module.HierarchicalNCCLWeightTransferEngine(
        None,
        SimpleNamespace(world_size=1),
        None,
    )
    info = module.HierarchicalWeightTransferInitInfo(
        role="relay",
        upstream={"master_address": "trainer", "master_port": 24000},
        downstream=[
            {"master_address": "relay", "master_port": 24001},
            {"master_address": "relay", "master_port": 24002},
        ],
    )
    engine.init_transfer_engine(info)
    assert fake_engine.calls == [
        ("trainer", 24000, 1, 2, 0),
        ("relay", 24001, 0, 2, 0),
        ("relay", 24002, 0, 2, 0),
    ]


def test_packed_chunk_boundaries_match_vllm_producer(monkeypatch) -> None:
    module, _, update_type = _load_module(monkeypatch)
    update = update_type(
        names=["a", "b", "c"],
        dtype_names=["bfloat16", "bfloat16", "bfloat16"],
        shapes=[[2], [3], [1]],
        packed=True,
        packed_buffer_size_bytes=8,
    )
    chunks = module.HierarchicalNCCLWeightTransferEngine._metadata_chunks(update)
    assert [[item[0] for item in chunk] for chunk in chunks] == [["a", "b"], ["c"]]


def test_leaf_rejects_downstream_links(monkeypatch) -> None:
    module, _, _ = _load_module(monkeypatch)
    try:
        module.HierarchicalWeightTransferInitInfo(
            role="leaf",
            upstream={"master_address": "relay", "master_port": 24001},
            downstream=[{"master_address": "other", "master_port": 24002}],
        )
    except ValueError as error:
        assert "leaf receivers" in str(error)
    else:
        raise AssertionError("leaf with downstream links must be rejected")
