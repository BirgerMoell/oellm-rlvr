"""Pass-rate filtering of RL pools against the current checkpoint (any `dataset`, the training verifiers).

GRPO/DAPO learns only from prompts whose sampled rewards differ within a group. A pool built once and reused goes
stale as the policy improves: W2 left 69 % of its prompts unsolved at k=8 (no signal), and solved prompts drift into
the same state from the other side. Each round therefore re-profiles candidates on the checkpoint it will train:

  sample   k completions per prompt with vLLM and the model's own chat template (the staged RL template, so
           thinking/direct defaults match training); resumable, shardable across GPUs.
  score    every completion with open-instruct's verifier for the row's `dataset`, built by
           `build_all_verifiers` under the same OELLM_PATCH_* flags as the training config, so the reward is the
           training reward (code rows need the code API, as in training). Truncated completions are kept with
           their verifier score and flagged; the trainer's non-stop penalty is not applied here.
  select   keep prompts whose reward varies within the group, optionally restricted to a solve-rate band, cap
           per `dataset`, and write a pool in the training schema plus a profile with per-dataset statistics.

Never derive these labels from a trainer's filtered rollouts (active sampling selects mixed groups; see
math_difficulty). Profiles are checkpoint-specific: record the model with the pool.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean, pstdev
from types import SimpleNamespace
from typing import Any

SOLVED = 1.0 - 1e-6


def read_pool(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    for index, row in enumerate(rows):
        missing = {"id", "messages", "ground_truth", "dataset"} - set(row)
        if missing:
            raise ValueError(f"pool row {index} is missing columns: {sorted(missing)}")
        if isinstance(row["messages"], str):
            row["messages"] = json.loads(row["messages"])
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in pool")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def shard_rows(rows: list[dict[str, Any]], shard: int, shards: int) -> list[dict[str, Any]]:
    if not 0 <= shard < shards:
        raise ValueError("shard must be in [0, shards)")
    return [r for r in rows if int(sha256(str(r["id"]).encode()).hexdigest()[:8], 16) % shards == shard]


def sample(pool: str | Path, model: str, output: str | Path, *, samples: int = 8, temperature: float = 1.0,
           top_p: float = 1.0, max_new_tokens: int = 8192, max_model_len: int = 16384, seed: int = 0,
           shard: int = 0, shards: int = 1, batch_size: int = 64, gpu_memory_utilization: float = 0.85,
           limit: int | None = None) -> dict[str, Any]:
    """Append k samples per prompt to `output` (JSONL); prompts already complete there are skipped."""
    rows = shard_rows(read_pool(pool), shard, shards)
    if limit:
        rows = rows[:limit]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = Counter(r["id"] for r in _read_jsonl(out))
    pending = [r for r in rows if done[str(r["id"])] < samples]
    identity = {"model": model, "pool": str(pool), "pool_sha256": sha256(Path(pool).read_bytes()).hexdigest(),
                "samples": samples, "temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens,
                "seed": seed, "shard": shard, "shards": shards}
    meta = out.with_suffix(".identity.json")
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError(f"refusing to resume {out} with a different model, pool or protocol")
    meta.write_text(json.dumps(identity, indent=1))
    if not pending:
        return {"prompts": len(rows), "sampled_now": 0}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError(f"{model} has no chat template")
    engine = LLM(model=model, trust_remote_code=True, dtype="bfloat16", max_model_len=max_model_len,
                 gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True, seed=seed)
    params = SamplingParams(n=samples, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    with out.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
                       for r in batch]
            for row, request in zip(batch, engine.generate(prompts, params, use_tqdm=True)):
                for index, generated in enumerate(request.outputs):
                    sink.write(json.dumps({"id": str(row["id"]), "dataset": row["dataset"], "sample_index": index,
                                           "text": generated.text, "token_ids": list(generated.token_ids),
                                           "finished": generated.finish_reason == "stop"}, ensure_ascii=False) + "\n")
            sink.flush()
    return {"prompts": len(rows), "sampled_now": len(pending)}


def build_verifiers(code_api_url: str | None = None, code_max_execution_time: float = 2.0,
                    binary_code_reward: bool = True) -> dict[str, Any]:
    """open-instruct's verifier table, as the trainer builds it (patches apply via sitecustomize + OELLM_PATCH_*)."""
    from open_instruct.ground_truth_utils import build_all_verifiers

    args = SimpleNamespace(
        code_api_url=code_api_url or "http://127.0.0.1:1/test_program", code_max_execution_time=code_max_execution_time,
        code_pass_rate_reward_threshold=1.0 if binary_code_reward else 0.0, code_apply_perf_penalty=False,
        llm_judge_model="none", llm_judge_max_tokens=1, llm_judge_max_context_length=1, llm_judge_temperature=0.0,
        llm_judge_timeout=1, seed=0, max_length_verifier_max_length=32768)
    return build_all_verifiers(args)


def _score_one(verifier, completion: dict[str, Any], label: Any, query: str) -> float:
    result = asyncio.run(verifier.async_call(completion["token_ids"], completion["text"], label, query))
    return float(result.score)


def score(pool: str | Path, sample_files: list[str | Path], output: str | Path, *,
          code_api_url: str | None = None, threads: int = 16) -> dict[str, Any]:
    """Score every sample with the training verifier for its row's `dataset`; resumable (appends to `output`)."""
    from concurrent.futures import ThreadPoolExecutor

    rows = {str(r["id"]): r for r in read_pool(pool)}
    verifiers = build_verifiers(code_api_url)
    out = Path(output)
    done = {(r["id"], r["sample_index"]) for r in _read_jsonl(out)}
    todo, unknown = [], Counter()
    for path in sample_files:
        for completion in _read_jsonl(Path(path)):
            key = (completion["id"], completion["sample_index"])
            if key in done:
                continue
            name = str(rows[completion["id"]]["dataset"]).lower()
            if name not in verifiers:
                unknown[name] += 1
                continue
            todo.append(completion)
    if unknown:
        raise ValueError(f"no verifier for datasets {dict(unknown)}; set the OELLM_PATCH_* flags of the training config")

    def one(completion):
        row = rows[completion["id"]]
        query = row["messages"][-1]["content"] if row["messages"] else ""
        try:
            value, error = _score_one(verifiers[str(row["dataset"]).lower()], completion, row["ground_truth"], query), None
        except Exception as exc:  # reported, never silently scored as 0
            value, error = None, f"{type(exc).__name__}: {exc}"[:300]
        return {"id": completion["id"], "dataset": row["dataset"], "sample_index": completion["sample_index"],
                "reward": value, "finished": completion["finished"], "response_tokens": len(completion["token_ids"]),
                "error": error}

    with ThreadPoolExecutor(threads) as pool_, out.open("a", encoding="utf-8") as sink:
        for record in pool_.map(one, todo):
            sink.write(json.dumps(record) + "\n")
    scored = _read_jsonl(out)
    return {"scored": len(scored), "new": len(todo), "errors": sum(r["error"] is not None for r in scored)}


def profile(scores: str | Path, samples: int) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(Path(scores)):
        by_id[record["id"]].append(record)
    rows = []
    for pid, records in by_id.items():
        if len({r["sample_index"] for r in records}) != samples or any(r["reward"] is None for r in records):
            rows.append({"id": pid, "dataset": records[0]["dataset"], "status": "incomplete"})
            continue
        rewards = [r["reward"] for r in records]
        solved = sum(r >= SOLVED for r in rewards)
        rows.append({"id": pid, "dataset": records[0]["dataset"], "status": "ok", "samples": len(rewards),
                     "mean_reward": fmean(rewards), "reward_std": pstdev(rewards), "solve_rate": solved / len(rewards),
                     "truncated": sum(not r["finished"] for r in records) / len(records),
                     "mean_response_tokens": fmean(r["response_tokens"] for r in records)})
    return rows


def band(row: dict[str, Any]) -> str:
    if row["status"] != "ok":
        return "incomplete"
    if row["reward_std"] == 0:
        return "no_signal_solved" if row["solve_rate"] == 1 else "no_signal_unsolved" if row["mean_reward"] <= 0 \
            else "no_signal_partial"
    rate = row["solve_rate"]
    return "hard" if rate < 0.375 else "medium" if rate <= 0.625 else "easy"


def select(pool: str | Path, scores: str | Path, output: str | Path, *, samples: int, min_solve: float = 0.0,
           max_solve: float = 1.0, max_truncated: float = 0.5, cap_per_dataset: dict[str, int] | None = None,
           seed: int = 20261029, model: str = "") -> dict[str, Any]:
    """Keep prompts with in-group reward variance (and solve rate in [min_solve, max_solve]); write pool + profile."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {str(r["id"]): r for r in read_pool(pool)}
    prof = profile(scores, samples)
    for p in prof:
        p["band"] = band(p)
    keep = [p for p in prof if p["status"] == "ok" and p["reward_std"] > 0 and min_solve <= p["solve_rate"] <= max_solve
            and p["truncated"] <= max_truncated]
    rng = random.Random(seed)
    rng.shuffle(keep)
    kept, per = [], Counter()
    for p in keep:
        cap = (cap_per_dataset or {}).get(str(p["dataset"]))
        if cap is not None and per[str(p["dataset"])] >= cap:
            continue
        per[str(p["dataset"])] += 1
        kept.append(p)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected = [rows[p["id"]] for p in kept]
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ("id", "messages", "ground_truth", "dataset")}
                                         for r in selected]), out)
    stats: dict[str, Any] = defaultdict(lambda: Counter())
    for p in prof:
        stats[str(p["dataset"])][p["band"]] += 1
    summary = {"model": model, "pool": str(pool), "samples": samples, "candidates": len(rows), "profiled": len(prof),
               "kept": len(kept), "kept_by_dataset": dict(per),
               "bands_by_dataset": {k: dict(v) for k, v in sorted(stats.items())},
               "criteria": {"reward_std": "> 0", "solve_rate": [min_solve, max_solve], "max_truncated": max_truncated,
                            "cap_per_dataset": cap_per_dataset or {}},
               "signal_fraction": round(sum(p.get("reward_std", 0) > 0 for p in prof) / max(1, len(prof)), 4)}
    out.with_suffix(".profile.json").write_text(json.dumps({"summary": summary, "prompts": prof}, indent=1))
    out.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=1))
    return summary
