from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
from typing import Any

FUNCTION_TASKS = [
    ("convert-temperature", "convert_temperature", {"value": 21, "from": "celsius", "to": "fahrenheit"}),
    ("scale-recipe", "scale_recipe", {"servings_from": 3, "servings_to": 8}),
    ("lookup-timezone", "lookup_timezone", {"city": "Helsinki"}),
    ("compute-vat", "compute_vat", {"net": 80, "country": "SE"}),
]
STATEFUL_TASKS = [
    ("stock-a12", "A12", 7, 5),
    ("stock-b07", "B07", 3, -2),
    ("stock-c99", "C99", 12, 4),
    ("stock-d21", "D21", 9, -3),
]
EDIT_TASKS = [
    ("edit-workers", {"workers": 2, "timeout": 30}, {"workers": 6, "timeout": 30}),
    ("edit-timeout", {"workers": 4, "timeout": 20}, {"workers": 4, "timeout": 75}),
    ("edit-feature", {"feature_enabled": False, "mode": "safe"}, {"feature_enabled": True, "mode": "safe"}),
    ("edit-region", {"region": "us", "retries": 2}, {"region": "eu", "retries": 2}),
]
REPAIR_TASKS = [
    (
        "repair-clamp",
        "def clamp(value, low, high):\n    return min(low, max(value, high))\n",
        "def clamp(value, low, high):\n    return max(low, min(value, high))\n",
        "from app import clamp\nassert clamp(5, 0, 10) == 5\nassert clamp(-2, 0, 10) == 0\nassert clamp(12, 0, 10) == 10\n",
    ),
    (
        "repair-slug",
        "def slug(text):\n    return text.strip().replace(' ', '_')\n",
        "def slug(text):\n    return '-'.join(text.strip().lower().split())\n",
        "from app import slug\nassert slug(' Hello World ') == 'hello-world'\nassert slug('A   B') == 'a-b'\n",
    ),
    (
        "repair-retries",
        "def attempts(retries):\n    return range(retries)\n",
        "def attempts(retries):\n    return range(retries + 1)\n",
        "from app import attempts\nassert list(attempts(0)) == [0]\nassert list(attempts(2)) == [0, 1, 2]\n",
    ),
    (
        "repair-bool",
        "def parse_bool(value):\n    return bool(value)\n",
        "def parse_bool(value):\n    return value.strip().lower() in {'1', 'true', 'yes', 'on'}\n",
        "from app import parse_bool\nassert parse_bool('true') is True\nassert parse_bool('false') is False\n",
    ),
]


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def _task_toml(name: str, category: str, sif: str) -> str:
    return f'''schema_version = "1.1"

[task]
name = "openeurollm/{name}"
description = "Deterministic OpenEuroLLM RL dry-run task"
authors = [{{ name = "OpenEuroLLM contributors" }}]
keywords = ["rlvr", "dry-run", "{category}"]

[metadata]
difficulty = "easy"
category = "{category}"
tags = ["project-authored", "deterministic", "lumi"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 180.0

[environment]
docker_image = {json.dumps(sif)}
workdir = "/tmp/oellm-task"
build_timeout_sec = 300.0
cpus = 1
memory_mb = 2048
storage_mb = 2048
gpus = 0
mcp_servers = []

[verifier.env]

[environment.env]

[solution.env]
'''


def _verifier(assertions: str, marker: str) -> str:
    return f'''from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

MARKER = {marker!r}
parser = argparse.ArgumentParser()
parser.add_argument("--root", default="/tmp/oellm-task")
args = parser.parse_args()
root = Path(args.root)
sys.path.insert(0, str(root))
try:
{''.join('    ' + line + chr(10) for line in assertions.splitlines())}    result = {{"reward": 1, "marker": MARKER}}
except Exception as error:
    result = {{"reward": 0, "marker": MARKER, "error": f"{{type(error).__name__}}: {{error}}"}}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["reward"] == 1 else 1)
'''


def _test_sh() -> str:
    return '''#!/bin/bash
set +e
python3 /tests/verify.py --root /tmp/oellm-task > /logs/verifier/result.json
status=$?
if [[ $status -eq 0 ]]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0
'''


def _function_script() -> str:
    return '''from __future__ import annotations
import json, sys
from pathlib import Path

call = json.loads(sys.argv[1])
name, args = call.get("name"), call.get("arguments")
if name == "convert_temperature": result = args["value"] * 9 / 5 + 32
elif name == "scale_recipe": result = args["servings_to"] / args["servings_from"]
elif name == "lookup_timezone": result = {"Helsinki": "Europe/Helsinki"}[args["city"]]
elif name == "compute_vat": result = args["net"] * {"SE": 0.25}[args["country"]]
else: raise SystemExit("unknown tool")
Path("audit.json").write_text(json.dumps([call], sort_keys=True))
Path("result.json").write_text(json.dumps({"result": result}, sort_keys=True))
print(json.dumps({"result": result}, sort_keys=True))
'''


def _write_common(task: Path, name: str, category: str, sif: str, instruction: str, verifier: str) -> None:
    _write(task / "task.toml", _task_toml(name, category, sif))
    _write(task / "instruction.md", instruction.rstrip() + "\n")
    # Harbor v0.22.0 derives the initial shell directory from an environment
    # Dockerfile even when docker_image points at a prebuilt SIF. Without this
    # declaration it silently defaults to /app and the agent cannot see the
    # task payload mounted at /tmp/oellm-task.
    _write(task / "environment/Dockerfile", "FROM scratch\nWORKDIR /tmp/oellm-task\n")
    _write(task / "tests/test.sh", _test_sh(), executable=True)
    _write(task / "tests/verify.py", verifier)


def _stage_sif_payload(task: Path) -> None:
    """Stage task files for Harbor's prebuilt-SIF bootstrap.

    Harbor uses ``environment/Dockerfile`` to resolve the shell workdir, but a
    prebuilt SIF is not rebuilt from that context.  Its Singularity backend
    only mounts ``environment/files`` at ``/staging/env_files`` and sources a
    ``setup.sh`` found there.  Mirror the policy-visible task payload into that
    directory and copy it into the resolved workdir at container startup.
    """

    environment = task / "environment"
    staged = environment / "files"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    _write(
        staged / "setup.sh",
        r'''#!/bin/bash
set -euo pipefail
find "$HARBOR_STAGING" -mindepth 1 -maxdepth 1 ! -name setup.sh -exec cp -a '{}' "$WORKDIR"/ \;
''',
        executable=True,
    )
    for source in sorted(environment.iterdir()):
        if source.name in {"Dockerfile", "files"}:
            continue
        target = staged / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def build_harbor_dryrun_pack(output: str | Path, sif: str) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for name, function, arguments in FUNCTION_TASKS:
        task = root / f"function-{name}"
        marker = f"private-function-{sha256(name.encode()).hexdigest()[:12]}"
        expected = {"name": function, "arguments": arguments}
        assertions = f"audit = json.loads((root / 'audit.json').read_text())\nassert audit == [{expected!r}]"
        _write_common(
            task,
            task.name,
            "function-calling",
            sif,
            "Use `python3 tool_api.py '<JSON call>'` exactly once. Send this exact semantic request "
            f"(JSON key order and whitespace may differ): tool={function}, arguments={json.dumps(arguments, sort_keys=True)}. "
            "Do not create or edit `audit.json` or `result.json` directly.",
            _verifier(assertions, marker),
        )
        _write(task / "environment/tool_api.py", _function_script())
        call = json.dumps(expected, separators=(",", ":"))
        _write(task / "solution/solve.sh", f"#!/bin/bash\nset -euo pipefail\ncd \"${{OELLM_TASK_ROOT:-/tmp/oellm-task}}\"\npython3 tool_api.py '{call}'\n", executable=True)

    for name, sku, quantity, delta in STATEFUL_TASKS:
        task = root / f"stateful-{name}"
        marker = f"private-stateful-{sha256(name.encode()).hexdigest()[:12]}"
        tool = '''from __future__ import annotations
import json, sys
from pathlib import Path
state_path, audit_path = Path("inventory.json"), Path("audit.json")
state = json.loads(state_path.read_text()); audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
command, sku = sys.argv[1], sys.argv[2]
if command == "get": result = state[sku]
elif command == "update":
    delta = int(sys.argv[3]); state[sku] += delta; state_path.write_text(json.dumps(state, sort_keys=True)); result = state[sku]
else: raise SystemExit("unknown command")
audit.append({"command": command, "sku": sku, "delta": int(sys.argv[3]) if command == "update" else None})
audit_path.write_text(json.dumps(audit, sort_keys=True)); print(result)
'''
        expected_audit = [
            {"command": "get", "sku": sku, "delta": None},
            {"command": "update", "sku": sku, "delta": delta},
        ]
        assertions = (
            f"state = json.loads((root / 'inventory.json').read_text())\nassert state == {{{sku!r}: {quantity + delta}}}\n"
            f"audit = json.loads((root / 'audit.json').read_text())\nassert audit == {expected_audit!r}"
        )
        _write_common(
            task,
            task.name,
            "stateful-tools",
            sif,
            f"Use `python3 inventory_tool.py get {sku}` first. Then update {sku} by {delta:+d} with the same tool. "
            "Do not directly edit the inventory or audit files.",
            _verifier(assertions, marker),
        )
        _write(task / "environment/inventory.json", json.dumps({sku: quantity}, sort_keys=True) + "\n")
        _write(task / "environment/inventory_tool.py", tool)
        _write(
            task / "solution/solve.sh",
            f"#!/bin/bash\nset -euo pipefail\ncd \"${{OELLM_TASK_ROOT:-/tmp/oellm-task}}\"\npython3 inventory_tool.py get {sku}\npython3 inventory_tool.py update {sku} {delta}\n",
            executable=True,
        )

    for name, initial, expected in EDIT_TASKS:
        task = root / f"terminal-{name}"
        marker = f"private-edit-{sha256(name.encode()).hexdigest()[:12]}"
        changed = next(key for key in expected if expected[key] != initial.get(key))
        assertions = f"value = json.loads((root / 'config.json').read_text())\nassert value == {expected!r}"
        _write_common(
            task,
            task.name,
            "terminal-edit",
            sif,
            f"Edit `config.json`: set `{changed}` to `{expected[changed]}` and preserve every other value.",
            _verifier(assertions, marker),
        )
        _write(task / "environment/config.json", json.dumps(initial, indent=2, sort_keys=True) + "\n")
        script = (
            "#!/bin/bash\nset -euo pipefail\ncd \"${OELLM_TASK_ROOT:-/tmp/oellm-task}\"\n"
            f"python3 -c \"import json; p='config.json'; d=json.load(open(p)); d[{changed!r}]={expected[changed]!r}; "
            "open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\"\n"
        )
        _write(task / "solution/solve.sh", script, executable=True)

    for name, broken, fixed, assertions in REPAIR_TASKS:
        task = root / f"repo-{name}"
        marker = f"private-repair-{sha256(name.encode()).hexdigest()[:12]}"
        _write_common(
            task,
            task.name,
            "repository-repair",
            sif,
            "Repair the bug in `app.py`. Keep the public function name and signature unchanged. "
            "Work concisely: inspect the file, make the smallest correct edit, verify it, and finish.",
            _verifier(assertions, marker),
        )
        _write(task / "environment/app.py", broken)
        encoded = b64encode(fixed.encode()).decode()
        _write(
            task / "solution/solve.sh",
            "#!/bin/bash\nset -euo pipefail\ncd \"${OELLM_TASK_ROOT:-/tmp/oellm-task}\"\n"
            f"python3 -c 'import base64; open(\"app.py\",\"wb\").write(base64.b64decode(\"{encoded}\"))'\n",
            executable=True,
        )

    for task in sorted(path for path in root.iterdir() if path.is_dir()):
        _stage_sif_payload(task)

    return validate_harbor_dryrun_pack(root)


def _run_verifier(task: Path, work: Path) -> bool:
    result = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(task / "tests/verify.py"), "--root", str(work)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def validate_harbor_dryrun_pack(root: str | Path) -> dict[str, Any]:
    pack = Path(root)
    tasks = sorted(path for path in pack.iterdir() if path.is_dir())
    if len(tasks) != 16:
        raise ValueError(f"Harbor dry-run pack must contain 16 tasks, found {len(tasks)}")
    records = []
    for task in tasks:
        required = [
            task / "task.toml",
            task / "instruction.md",
            task / "environment",
            task / "environment/Dockerfile",
            task / "environment/files/setup.sh",
            task / "solution/solve.sh",
            task / "tests/test.sh",
            task / "tests/verify.py",
        ]
        missing = [str(path.relative_to(pack)) for path in required if not path.exists()]
        if missing:
            raise ValueError(f"task {task.name} is incomplete: {missing}")
        if (task / "environment/Dockerfile").read_text() != "FROM scratch\nWORKDIR /tmp/oellm-task\n":
            raise ValueError(f"task {task.name} does not declare Harbor's expected workdir")
        for source in sorted((task / "environment").iterdir()):
            if source.name in {"Dockerfile", "files"}:
                continue
            staged = task / "environment/files" / source.name
            if not staged.exists():
                raise ValueError(f"task {task.name} does not stage {source.name} for its prebuilt SIF")
            if source.is_file() and source.read_bytes() != staged.read_bytes():
                raise ValueError(f"task {task.name} stages a stale copy of {source.name}")
        policy_text = (task / "instruction.md").read_text() + "\n" + "\n".join(
            path.read_text(errors="replace") for path in sorted((task / "environment").rglob("*")) if path.is_file()
        )
        private_marker = next(
            line for line in (task / "tests/verify.py").read_text().splitlines() if line.startswith("MARKER = ")
        ).split(" = ", 1)[1].strip("'\"")
        if private_marker in policy_text:
            raise ValueError(f"private verifier marker leaked into policy-visible task {task.name}")
        with tempfile.TemporaryDirectory(prefix=f"{task.name}-wrong-") as temporary:
            wrong = Path(temporary)
            shutil.copytree(task / "environment", wrong, dirs_exist_ok=True)
            wrong_fails = not _run_verifier(task, wrong)
        oracle_results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory(prefix=f"{task.name}-oracle-") as temporary:
                work = Path(temporary)
                shutil.copytree(task / "environment", work, dirs_exist_ok=True)
                environment = {**os.environ, "OELLM_TASK_ROOT": str(work)}
                solution = subprocess.run(
                    ["bash", str(task / "solution/solve.sh")],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                oracle_results.append(solution.returncode == 0 and _run_verifier(task, work))
        records.append(
            {
                "task": task.name,
                "oracle_passes_twice": oracle_results == [True, True],
                "unchanged_environment_fails": wrong_fails,
                "private_marker_absent_from_policy_surface": True,
            }
        )
    report = {
        "ok": all(all(value for key, value in row.items() if key != "task") for row in records),
        "tasks": len(records),
        "by_category": {
            "function-calling": sum(row["task"].startswith("function-") for row in records),
            "stateful-tools": sum(row["task"].startswith("stateful-") for row in records),
            "terminal-edit": sum(row["task"].startswith("terminal-") for row in records),
            "repository-repair": sum(row["task"].startswith("repo-") for row in records),
        },
        "records": records,
    }
    (pack / "validation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
