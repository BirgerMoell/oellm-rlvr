from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_assemble_preflight_requires_every_gate(tmp_path: Path) -> None:
    for index in range(2):
        (tmp_path / f"node-n{index}.json").write_text(
            json.dumps({"ok": True, "details": {"visible_gpus": 8}})
        )
    (tmp_path / "ray-cluster.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "model-generation.json").write_text(json.dumps({"ok": True}))
    transfer = [
        *['{"event": "hierarchical_weight_transfer_probe_rank", "received": 9173}' for _ in range(9)],
        *['{"event": "hierarchical_weight_transfer_probe_node_complete"}' for _ in range(2)],
    ]
    (tmp_path / "hierarchical-transfer.log").write_text("\n".join(transfer))
    (tmp_path / "package-lock.txt").write_text("ray==2.54.0\n")
    (tmp_path / "container-and-driver-digests.txt").write_text("digest\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/assemble_lumi_preflight.py"), "--results-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["ok"] is True
    assert all(report["gates"].values())
