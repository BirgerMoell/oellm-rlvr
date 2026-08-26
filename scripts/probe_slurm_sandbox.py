#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile

from oellm_rlvr.slurm_sandbox import SlurmApptainerBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the LUMI same-node Slurm/Apptainer sandbox proxy")
    parser.add_argument("--image", required=True)
    parser.add_argument("--tmp-dir", required=True)
    args = parser.parse_args()

    backend = SlurmApptainerBackend(image=args.image, tmp_dir=args.tmp_dir, timeout=30)
    backend.start()
    try:
        backend.write_file(
            "/workspace/solution.py",
            "value = int(input())\nprint(value * 2)\n",
        )
        direct = backend.run_command("printf '21\\n' | python /workspace/solution.py")
        if direct.exit_code != 0 or direct.stdout.strip() != "42":
            raise RuntimeError(f"sandbox command failed: {direct}")

        test_script = b"#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\ntest \"$(printf '2\\n' | python /workspace/solution.py)\" = 4\nprintf '1.0\\n' > /logs/verifier/reward.txt\n"
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            member = tarfile.TarInfo("tests/test.sh")
            member.mode = 0o755
            member.size = len(test_script)
            archive.addfile(member, io.BytesIO(test_script))
        backend.put_archive("/", stream.getvalue())
        tests = backend.run_command("bash /tests/test.sh")
        reward = backend.read_file("/logs/verifier/reward.txt").strip()
        if tests.exit_code != 0 or reward != "1.0":
            raise RuntimeError(f"sandbox verifier failed: result={tests}, reward={reward!r}")
        print(json.dumps({"ok": True, "direct_stdout": direct.stdout.strip(), "reward": reward}))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
