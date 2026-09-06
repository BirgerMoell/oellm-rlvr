from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .harbor_results import _trial_results
from .schemas import AgentTrajectoryIndexRecord

_ATIF_VERSION = re.compile(r"^ATIF-v(?P<major>\d+)\.(?P<minor>\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _integer(value: object, field: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{field} must be a non-negative integer")
        return 0
    return value


def _inspect_atif(payload: object, *, require_token_ids: bool, require_logprobs: bool) -> dict[str, Any]:
    errors: list[str] = []
    counters: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "errors": errors,
        "schema_version": None,
        "agent_name": None,
        "agent_version": None,
        "model_name": None,
        "session_id": None,
        "trajectory_id": None,
        "step_count": 0,
        "fresh_step_count": 0,
        "llm_call_count": 0,
        "tool_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "trainable_agent_steps": 0,
        "agent_steps_with_token_ids": 0,
        "agent_steps_with_logprobs": 0,
        "copied_context_steps": 0,
    }
    if not isinstance(payload, dict):
        errors.append("ATIF root must be an object")
        return summary

    version = payload.get("schema_version")
    summary["schema_version"] = version if isinstance(version, str) else None
    match = _ATIF_VERSION.fullmatch(version) if isinstance(version, str) else None
    if not match or int(match.group("major")) != 1:
        errors.append(f"unsupported ATIF schema_version: {version!r}")

    agent = payload.get("agent")
    if not isinstance(agent, dict):
        errors.append("agent must be an object")
        agent = {}
    for field in ("name", "version"):
        if not isinstance(agent.get(field), str) or not agent[field]:
            errors.append(f"agent.{field} must be a non-empty string")
    summary["agent_name"] = agent.get("name") if isinstance(agent.get("name"), str) else None
    summary["agent_version"] = agent.get("version") if isinstance(agent.get("version"), str) else None
    summary["model_name"] = agent.get("model_name") if isinstance(agent.get("model_name"), str) else None
    summary["session_id"] = payload.get("session_id")
    summary["trajectory_id"] = payload.get("trajectory_id")

    steps = payload.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be an array")
        return summary
    summary["step_count"] = len(steps)
    known_call_ids: set[str] = set()

    for position, step in enumerate(steps, start=1):
        prefix = f"steps[{position - 1}]"
        if not isinstance(step, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if step.get("step_id") != position:
            errors.append(f"{prefix}.step_id must be {position}")
        source = step.get("source")
        if source not in {"system", "user", "agent"}:
            errors.append(f"{prefix}.source is invalid")
        if "message" not in step:
            errors.append(f"{prefix}.message is required")
        copied = step.get("is_copied_context") is True
        if copied:
            counters["copied_context_steps"] += 1
        else:
            counters["fresh_step_count"] += 1

        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            errors.append(f"{prefix}.tool_calls must be an array")
            tool_calls = []
        for tool_index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                errors.append(f"{prefix}.tool_calls[{tool_index}] must be an object")
                continue
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                errors.append(f"{prefix}.tool_calls[{tool_index}].tool_call_id is required")
            elif call_id in known_call_ids:
                errors.append(f"duplicate tool_call_id: {call_id}")
            else:
                known_call_ids.add(call_id)
            if not isinstance(call.get("function_name"), str) or not isinstance(call.get("arguments"), dict):
                errors.append(f"{prefix}.tool_calls[{tool_index}] has an invalid function or arguments")
        if not copied:
            counters["tool_call_count"] += len(tool_calls)

        observation = step.get("observation")
        if observation is not None:
            results = observation.get("results") if isinstance(observation, dict) else None
            if not isinstance(results, list):
                errors.append(f"{prefix}.observation.results must be an array")
            else:
                for result_index, result in enumerate(results):
                    if not isinstance(result, dict):
                        errors.append(f"{prefix}.observation.results[{result_index}] must be an object")
                        continue
                    source_call_id = result.get("source_call_id")
                    if source_call_id is not None and source_call_id not in known_call_ids:
                        errors.append(f"{prefix}.observation references unknown tool call {source_call_id!r}")

        if source != "agent":
            continue
        llm_calls_raw = step.get("llm_call_count")
        llm_calls = 1 if llm_calls_raw is None else _integer(llm_calls_raw, f"{prefix}.llm_call_count", errors)
        if llm_calls == 0 and (step.get("metrics") is not None or step.get("reasoning_content") is not None):
            errors.append(f"{prefix} is a deterministic dispatch but contains LLM-only fields")
        if copied or llm_calls == 0:
            continue
        counters["llm_call_count"] += llm_calls
        counters["trainable_agent_steps"] += 1

        metrics = step.get("metrics")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics must be an object")
            metrics = {}
        prompt_tokens = _integer(metrics.get("prompt_tokens", 0), f"{prefix}.metrics.prompt_tokens", errors)
        completion_tokens = _integer(
            metrics.get("completion_tokens", 0), f"{prefix}.metrics.completion_tokens", errors
        )
        counters["prompt_tokens"] += prompt_tokens
        counters["completion_tokens"] += completion_tokens

        token_ids = metrics.get("completion_token_ids")
        logprobs = metrics.get("logprobs")
        if token_ids is not None:
            if not isinstance(token_ids, list) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
            ):
                errors.append(f"{prefix}.metrics.completion_token_ids must contain integers")
            else:
                counters["agent_steps_with_token_ids"] += 1
                if "completion_tokens" in metrics and len(token_ids) != completion_tokens:
                    errors.append(f"{prefix} completion token count does not match completion_token_ids")
        elif require_token_ids:
            errors.append(f"{prefix} is missing completion_token_ids required for RL")
        if logprobs is not None:
            if not isinstance(logprobs, list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in logprobs
            ):
                errors.append(f"{prefix}.metrics.logprobs must contain finite numbers")
            else:
                counters["agent_steps_with_logprobs"] += 1
                if isinstance(token_ids, list) and len(logprobs) != len(token_ids):
                    errors.append(f"{prefix} logprobs do not align with completion_token_ids")
        elif require_logprobs:
            errors.append(f"{prefix} is missing logprobs required for RL")

    summary.update(counters)
    if not steps:
        errors.append("trajectory has no steps")
    if counters["trainable_agent_steps"] == 0:
        errors.append("trajectory has no fresh LLM-generated agent steps")
    return summary


def index_harbor_atif(
    root: str | Path,
    output: str | Path,
    *,
    run_id: str,
    policy_version: int,
    learner_version: int,
    require_token_ids: bool = False,
    require_logprobs: bool = False,
) -> dict[str, Any]:
    """Validate Harbor trials and write one auditable index row per trial."""

    base = Path(root)
    destination = Path(output)
    if policy_version < 0 or learner_version < 0:
        raise ValueError("policy and learner versions must be non-negative")
    trials = _trial_results(base)
    records: list[AgentTrajectoryIndexRecord] = []
    rejection_counts: Counter[str] = Counter()

    for result_path, result in trials:
        reasons: list[str] = []
        atif_path = result_path.parent / "agent" / "trajectory.json"
        inspection: dict[str, Any] = {
            "errors": ["agent/trajectory.json is missing"],
            "schema_version": None,
            "agent_name": None,
            "agent_version": None,
            "model_name": None,
            "session_id": None,
            "trajectory_id": None,
            "step_count": 0,
            "fresh_step_count": 0,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "trainable_agent_steps": 0,
            "agent_steps_with_token_ids": 0,
            "agent_steps_with_logprobs": 0,
            "copied_context_steps": 0,
        }
        atif_sha256: str | None = None
        if atif_path.is_file():
            try:
                atif_payload = json.loads(atif_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                inspection["errors"] = [f"cannot read ATIF: {error}"]
            else:
                inspection = _inspect_atif(
                    atif_payload,
                    require_token_ids=require_token_ids,
                    require_logprobs=require_logprobs,
                )
                atif_sha256 = _sha256(atif_path)
        reasons.extend(inspection["errors"])

        exception = result.get("exception_info")
        if exception is not None:
            reasons.append("Harbor trial has exception_info")
        raw_rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward_components: dict[str, float] = {}
        for name, value in raw_rewards.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                reasons.append(f"verifier reward {name!r} is not a finite number")
            else:
                reward_components[str(name)] = float(value)
        reward = reward_components.get("reward")
        if reward is None:
            reasons.append("verifier reward 'reward' is missing")
        for reason in set(reasons):
            rejection_counts[reason] += 1

        result_agent = result.get("agent_info") or {}
        raw_model_info = result_agent.get("model_info")
        model_info = raw_model_info if isinstance(raw_model_info, dict) else {}
        agent_name = inspection["agent_name"] or result_agent.get("name") or "unknown"
        model_name = inspection["model_name"] or model_info.get("name")
        record = AgentTrajectoryIndexRecord(
            run_id=run_id,
            trial_id=str(result.get("id") or result.get("trial_name") or result_path.parent.name),
            task_name=str(result.get("task_name") or "unknown"),
            task_checksum=result.get("task_checksum"),
            result_path=_relative(result_path, base),
            result_sha256=_sha256(result_path),
            atif_path=_relative(atif_path, base) if atif_path.is_file() else None,
            atif_sha256=atif_sha256,
            atif_schema_version=inspection["schema_version"],
            agent_name=str(agent_name),
            agent_version=inspection["agent_version"] or result_agent.get("version"),
            model_name=model_name,
            reward=reward,
            reward_components=reward_components,
            policy_version=policy_version,
            learner_version=learner_version,
            policy_lag=max(0, learner_version - policy_version),
            accepted_for_rl=not reasons,
            rejection_reasons=reasons,
            step_count=inspection["step_count"],
            fresh_step_count=inspection["fresh_step_count"],
            llm_call_count=inspection["llm_call_count"],
            tool_call_count=inspection["tool_call_count"],
            prompt_tokens=inspection["prompt_tokens"],
            completion_tokens=inspection["completion_tokens"],
            trainable_agent_steps=inspection["trainable_agent_steps"],
            agent_steps_with_token_ids=inspection["agent_steps_with_token_ids"],
            agent_steps_with_logprobs=inspection["agent_steps_with_logprobs"],
            copied_context_steps=inspection["copied_context_steps"],
            started_at=result.get("started_at"),
            finished_at=result.get("finished_at"),
            metadata={
                "session_id": inspection["session_id"],
                "trajectory_id": inspection["trajectory_id"],
                "trial_name": result.get("trial_name"),
                "environment_type": ((result.get("config") or {}).get("environment") or {}).get("type"),
            },
        )
        records.append(record)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as sink:
        for record in records:
            sink.write(record.model_dump_json() + "\n")
    accepted = sum(record.accepted_for_rl for record in records)
    return {
        "ok": bool(records) and accepted == len(records),
        "root": str(base.resolve()),
        "output": str(destination.resolve()),
        "run_id": run_id,
        "policy_version": policy_version,
        "learner_version": learner_version,
        "requirements": {"completion_token_ids": require_token_ids, "logprobs": require_logprobs},
        "observed": {"trials": len(records), "accepted": accepted, "rejected": len(records) - accepted},
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
