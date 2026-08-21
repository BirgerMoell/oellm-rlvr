from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .schemas import TaskSpec, VerifierResult

_MAX_CAPTURE = 32_000
_FINAL_TAG = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
_FINAL_LINE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    index = start + len(marker)
    begin = index
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index]
        index += 1
    return None


def extract_math_answer(completion: str) -> str:
    boxed = _last_boxed(completion)
    if boxed is not None:
        return boxed.strip()
    tags = _FINAL_TAG.findall(completion)
    if tags:
        return tags[-1].strip()
    lines = _FINAL_LINE.findall(completion)
    if lines:
        return lines[-1].strip()
    nonempty = [line.strip() for line in completion.splitlines() if line.strip()]
    return nonempty[-1] if nonempty else ""


def _latex_fractions(value: str) -> str:
    pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"((\1)/(\2))", value)
    return value


def normalize_math_answer(value: str) -> str:
    value = value.strip().strip("$")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = _latex_fractions(value)
    value = value.replace("−", "-").replace("^", "**")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[.,;:]$", "", value)
    return value


def _numeric(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _numeric(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _numeric(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _numeric(node.left), _numeric(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right.numerator) <= 20:
            return left ** right.numerator
    raise ValueError("not a bounded numeric expression")


def equivalent_math_answers(candidate: str, expected: str) -> bool:
    candidate_norm = normalize_math_answer(candidate)
    expected_norm = normalize_math_answer(expected)
    if not candidate_norm or len(candidate_norm) > 512:
        return False
    if candidate_norm.casefold() == expected_norm.casefold():
        return True
    allowed = re.compile(r"[0-9eE+\-*/().]+")
    if not allowed.fullmatch(candidate_norm) or not allowed.fullmatch(expected_norm):
        return False
    try:
        return _numeric(ast.parse(candidate_norm, mode="eval")) == _numeric(ast.parse(expected_norm, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False


class Verifier(ABC):
    @abstractmethod
    def verify(self, task: TaskSpec, completion: str) -> VerifierResult:
        raise NotImplementedError


class MathVerifier(Verifier):
    def verify(self, task: TaskSpec, completion: str) -> VerifierResult:
        if task.kind != "math":
            raise ValueError("MathVerifier requires a math task")
        if task.ground_truth is None:
            raise ValueError("math task has no ground_truth")
        started = time.monotonic()
        extracted = extract_math_answer(completion)
        answers = task.ground_truth if isinstance(task.ground_truth, list) else [task.ground_truth]
        passed = any(equivalent_math_answers(extracted, answer) for answer in answers)
        return VerifierResult(
            reward=1.0 if passed else 0.0,
            passed=passed,
            verifier="math_exact",
            reason="answer matched" if passed else "answer did not match",
            extracted_answer=extracted,
            duration_seconds=time.monotonic() - started,
        )


@dataclass(frozen=True)
class Execution:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class SandboxRunner(ABC):
    @abstractmethod
    def run(self, workspace: Path, command: Sequence[str], timeout: int) -> Execution:
        raise NotImplementedError


class LocalRunner(SandboxRunner):
    """Unsafe runner for trusted smoke tests only; construction requires explicit opt-in."""

    def __init__(self, allow_unsafe: bool = False):
        if not allow_unsafe:
            raise ValueError("LocalRunner executes untrusted code; pass allow_unsafe=True only for trusted smoke tests")

    def run(self, workspace: Path, command: Sequence[str], timeout: int) -> Execution:
        started = time.monotonic()
        result = subprocess.run(
            list(command), cwd=workspace, capture_output=True, text=True, timeout=timeout, check=False
        )
        return Execution(
            result.returncode,
            result.stdout[-_MAX_CAPTURE:],
            result.stderr[-_MAX_CAPTURE:],
            time.monotonic() - started,
        )


class ApptainerRunner(SandboxRunner):
    def __init__(self, image: str, binary: str = "singularity", contain_all: bool = True):
        self.image = image
        self.binary = binary
        self.contain_all = contain_all

    def run(self, workspace: Path, command: Sequence[str], timeout: int) -> Execution:
        if shutil.which(self.binary) is None:
            raise FileNotFoundError(f"sandbox runtime not found: {self.binary}")
        argv = [self.binary, "exec", "--cleanenv"]
        if self.contain_all:
            argv.append("--containall")
        argv.extend(["--bind", f"{workspace}:/workspace", "--pwd", "/workspace", self.image, *command])
        started = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return Execution(
            result.returncode,
            result.stdout[-_MAX_CAPTURE:],
            result.stderr[-_MAX_CAPTURE:],
            time.monotonic() - started,
        )


def extract_code(completion: str) -> str:
    fenced = _FENCE.findall(completion)
    return fenced[-1].strip() + "\n" if fenced else completion.strip() + "\n"


def _write_workspace(workspace: Path, files: dict[str, str]) -> None:
    root = workspace.resolve()
    for relative, content in files.items():
        target = (workspace / relative).resolve()
        if root not in target.parents:
            raise ValueError(f"task file escapes workspace: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


class CodeVerifier(Verifier):
    def __init__(self, runner: SandboxRunner, timeout: int = 120):
        self.runner = runner
        self.timeout = timeout

    def verify(self, task: TaskSpec, completion: str) -> VerifierResult:
        if task.kind != "code":
            raise ValueError("CodeVerifier requires a code task")
        if not task.test_command:
            raise ValueError("code task requires test_command")
        with tempfile.TemporaryDirectory(prefix="oellm-code-") as directory:
            workspace = Path(directory)
            _write_workspace(workspace, task.files)
            target = workspace / task.candidate_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(extract_code(completion))
            try:
                execution = self.runner.run(workspace, task.test_command, self.timeout)
            except subprocess.TimeoutExpired as error:
                return VerifierResult(
                    reward=0,
                    passed=False,
                    verifier="code_tests",
                    reason=f"test timeout after {self.timeout}s",
                    error_type="timeout",
                    duration_seconds=self.timeout,
                    stdout=(error.stdout or "")[-_MAX_CAPTURE:] if isinstance(error.stdout, str) else "",
                    stderr=(error.stderr or "")[-_MAX_CAPTURE:] if isinstance(error.stderr, str) else "",
                )
            except Exception as error:  # noqa: BLE001 - sandbox failures are verifier results, not trainer failures
                return VerifierResult(
                    reward=0,
                    passed=False,
                    verifier="code_tests",
                    reason=str(error),
                    error_type="sandbox_error",
                )
        passed = execution.returncode == 0
        return VerifierResult(
            reward=1.0 if passed else 0.0,
            passed=passed,
            verifier="code_tests",
            reason="all tests passed" if passed else f"tests exited {execution.returncode}",
            stdout=execution.stdout,
            stderr=execution.stderr,
            duration_seconds=execution.duration_seconds,
        )
