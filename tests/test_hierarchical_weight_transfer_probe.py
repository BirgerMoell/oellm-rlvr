from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_hierarchical_weight_transfer.py"
SPEC = importlib.util.spec_from_file_location("probe_hierarchical_weight_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_trainer_node_gets_one_trainer_role() -> None:
    assert MODULE.local_roles("trainer", "trainer", 7) == [("trainer", None, 0)]


def test_rollout_node_gets_relay_and_seven_leaves() -> None:
    assert MODULE.local_roles("rollout", "trainer", 7) == [
        ("relay", None, 0),
        ("leaf", 1, 1),
        ("leaf", 2, 2),
        ("leaf", 3, 3),
        ("leaf", 4, 4),
        ("leaf", 5, 5),
        ("leaf", 6, 6),
        ("leaf", 7, 7),
    ]


def test_child_environment_isolates_each_role_gpu(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    env = MODULE.child_environment(6)
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert env["HIP_VISIBLE_DEVICES"] == "6"
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
