from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_nccl_weight_transfer.py"
SPEC = importlib.util.spec_from_file_location("probe_nccl_weight_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trainer_gets_rank_zero_on_device_zero() -> None:
    assert MODULE.local_rank_assignments("learner", "learner", 9) == [(0, 0)]


def test_rollout_node_gets_eight_unique_ranks_and_devices() -> None:
    assert MODULE.local_rank_assignments("rollout", "learner", 9) == [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (7, 6),
        (8, 7),
    ]


def test_two_rank_probe_uses_one_worker_device() -> None:
    assert MODULE.local_rank_assignments("rollout", "learner", 2) == [(1, 0)]


def test_child_environment_isolates_one_gpu(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    env = MODULE.child_environment(4)
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert env["HIP_VISIBLE_DEVICES"] == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
