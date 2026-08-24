from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .backend import build_backend_argv, shell_command
from .config import load_config
from .datasets import make_math_smoke, pack_code_dataset, sample_code_dataset, sample_math_dataset
from .gates import evaluate_gates
from .schemas import TaskSpec
from .slurm import render_slurm
from .store import JsonlTrajectoryStore
from .topology import build_topology
from .verifiers import ApptainerRunner, CodeVerifier, LocalRunner, MathVerifier


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _task(path: str) -> TaskSpec:
    source = Path(path)
    raw = yaml.safe_load(source.read_text()) if source.suffix in {".yaml", ".yml"} else json.loads(source.read_text())
    return TaskSpec.model_validate(raw)


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _json({"valid": True, "name": config.name, "task": config.task.kind, "topology": build_topology(config).as_dict()})
    return 0


def command_topology(args: argparse.Namespace) -> int:
    _json(build_topology(load_config(args.config)).as_dict())
    return 0


def command_backend_command(args: argparse.Namespace) -> int:
    print(shell_command(load_config(args.config)))
    return 0


def command_render_slurm(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rendered = render_slurm(config, args.config)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        output.chmod(0o750)
        print(output)
    else:
        print(rendered, end="")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checks: dict[str, object] = {}
    checks["container"] = Path(config.platform.container).is_file()
    backend = Path(config.backend.repo_path)
    checks["backend"] = (backend / ".git").exists()
    if checks["backend"]:
        result = subprocess.run(
            ["git", "-C", str(backend), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        actual = result.stdout.strip()
        checks["backend_commit"] = actual
        checks["backend_commit_matches"] = actual.startswith(config.backend.commit)
    if config.task.sandbox:
        # Singularity/Apptainer accepts both immutable SIF files and unpacked
        # sandbox directories as execution images.
        checks["sandbox_image"] = Path(config.task.sandbox.image).exists()
    ok = all(value for key, value in checks.items() if isinstance(value, bool))
    _json({"ok": ok, "checks": checks})
    return 0 if ok else 1


def command_run_backend(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    backend = Path(config.backend.repo_path)
    result = subprocess.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect backend checkout: {result.stderr.strip()}")
    actual = result.stdout.strip()
    if not actual.startswith(config.backend.commit):
        raise RuntimeError(f"backend revision mismatch: expected {config.backend.commit}, got {actual}")
    os.chdir(backend)
    argv = build_backend_argv(config)
    environment = os.environ.copy()
    if config.model.local_path:
        environment["OELLM_MODEL_ID"] = config.model.name_or_path
        environment["OELLM_MODEL_PATH"] = config.model.local_path
    os.execvpe(argv[0], argv, environment)
    return 127


def command_make_math_smoke(args: argparse.Namespace) -> int:
    make_math_smoke(args.output, args.count)
    print(args.output)
    return 0


def command_pack_code(args: argparse.Namespace) -> int:
    dataset_path, task_root = pack_code_dataset(args.manifest, args.output_dir)
    _json({"dataset": str(dataset_path), "task_data_dir": str(task_root)})
    return 0


def command_sample_math(args: argparse.Namespace) -> int:
    sample_math_dataset(
        args.source,
        args.output,
        args.count,
        language=args.language,
        min_difficulty=args.min_difficulty,
        diverse_by=args.diverse_by,
    )
    print(args.output)
    return 0


def command_sample_code(args: argparse.Namespace) -> int:
    dataset_path, task_root = sample_code_dataset(
        args.source, args.output_dir, args.image, args.count, args.copies, args.max_steps
    )
    _json({"dataset": str(dataset_path), "task_data_dir": str(task_root)})
    return 0


def command_verify(args: argparse.Namespace) -> int:
    task = _task(args.task)
    completion = Path(args.completion).read_text()
    if task.kind == "math":
        verifier = MathVerifier()
    elif args.runner == "local":
        verifier = CodeVerifier(LocalRunner(allow_unsafe=args.allow_unsafe_local), timeout=args.timeout)
    else:
        if not args.image:
            raise ValueError("--image is required for the apptainer runner")
        verifier = CodeVerifier(ApptainerRunner(args.image, binary=args.binary), timeout=args.timeout)
    result = verifier.verify(task, completion)
    _json(result.model_dump(mode="json"))
    return 0 if result.passed else 1


def command_gate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = evaluate_gates(JsonlTrajectoryStore(args.trajectories), config.gates)
    _json(report.as_dict())
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oellm-rlvr")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("validate", command_validate), ("topology", command_topology), ("backend-command", command_backend_command), ("doctor", command_doctor)):
        child = sub.add_parser(name)
        child.add_argument("--config", required=True)
        child.set_defaults(handler=handler)

    render = sub.add_parser("render-slurm")
    render.add_argument("--config", required=True)
    render.add_argument("--output")
    render.set_defaults(handler=command_render_slurm)

    run = sub.add_parser("run-backend", help=argparse.SUPPRESS)
    run.add_argument("--config", required=True)
    run.set_defaults(handler=command_run_backend)

    math_smoke = sub.add_parser("make-math-smoke")
    math_smoke.add_argument("--output", required=True)
    math_smoke.add_argument("--count", type=int, default=64)
    math_smoke.set_defaults(handler=command_make_math_smoke)

    pack = sub.add_parser("pack-code")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.set_defaults(handler=command_pack_code)

    math_sample = sub.add_parser("sample-math")
    math_sample.add_argument("--source", required=True)
    math_sample.add_argument("--output", required=True)
    math_sample.add_argument("--count", type=int, default=8)
    math_sample.add_argument("--language", help="optional exact language-code filter, for example en")
    math_sample.add_argument("--min-difficulty", type=int, help="optional inclusive difficulty floor")
    math_sample.add_argument("--diverse-by", help="require a different non-empty value in this column per row")
    math_sample.set_defaults(handler=command_sample_math)

    code_sample = sub.add_parser("sample-code")
    code_sample.add_argument("--source", required=True)
    code_sample.add_argument("--output-dir", required=True)
    code_sample.add_argument("--image", required=True)
    code_sample.add_argument("--count", type=int, default=8)
    code_sample.add_argument("--copies", type=int, default=1)
    code_sample.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="maximum sandbox turns per trajectory (default: 6 for the smoke profile)",
    )
    code_sample.set_defaults(handler=command_sample_code)

    verify = sub.add_parser("verify")
    verify.add_argument("--task", required=True)
    verify.add_argument("--completion", required=True)
    verify.add_argument("--runner", choices=("apptainer", "local"), default="apptainer")
    verify.add_argument("--image")
    verify.add_argument("--binary", default="singularity")
    verify.add_argument("--timeout", type=int, default=120)
    verify.add_argument("--allow-unsafe-local", action="store_true")
    verify.set_defaults(handler=command_verify)

    gate = sub.add_parser("gate")
    gate.add_argument("--config", required=True)
    gate.add_argument("--trajectories", required=True)
    gate.set_defaults(handler=command_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to a concise exit status
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
