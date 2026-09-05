from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any

from .verifiers import equivalent_math_answers, extract_math_answer, normalize_math_answer

_SIMPLE_BOX = re.compile(r"\\boxed\{([^{}]+)\}")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _equivalent_numeric(candidate: str, expected: str) -> bool:
    # GSM8K's canonical harness ignores thousands separators and currency
    # symbols. Keep that behavior local instead of weakening the general
    # verifier contract used by other datasets.
    return equivalent_math_answers(
        candidate.replace(",", "").replace("$", ""),
        expected.replace(",", "").replace("$", ""),
    )


def _normalized_numeric(value: str) -> str:
    return normalize_math_answer(value.replace(",", "").replace("$", ""))


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
        return pq.read_table(source).to_pylist()
    rows: list[dict[str, Any]] = []
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            yield value


def _sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_repeated_ngrams(text: str, n: int = 4) -> float:
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    grams = list(zip(*(tokens[offset:] for offset in range(n))))
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def analyze_reasoning_completion(
    text: str,
    expected: str,
    *,
    finish_reason: str = "stop",
    response_tokens: int | None = None,
) -> dict[str, Any]:
    """Score correctness and conservative, explicitly structural form checks.

    The structure flags are not claims that the chain of reasoning is sound.
    Semantic reasoning quality is handled by the blinded audit pack.
    """
    extracted = extract_math_answer(text)
    boxes = _SIMPLE_BOX.findall(text)
    last_box_position = text.rfind("\\boxed{")
    reasoning_prefix = text[:last_box_position] if last_box_position >= 0 else "\n".join(text.splitlines()[:-1])
    prefix_tokens = _TOKEN.findall(reasoning_prefix)
    repeated_fraction = _fraction_repeated_ngrams(text)
    open_think = len(re.findall(r"<think>", text, flags=re.IGNORECASE))
    close_think = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
    lower_text = text.casefold()
    open_think_position = lower_text.find("<think>")
    close_think_position = lower_text.find("</think>")
    reasoning_channel_format_pass = (
        open_think == 1
        and close_think == 1
        and open_think_position < close_think_position < last_box_position
        and len(boxes) == 1
    )
    expected_normalized = _normalized_numeric(expected)
    extracted_normalized = _normalized_numeric(extracted)
    return {
        "expected": expected,
        "expected_normalized": expected_normalized,
        "extracted_answer": extracted,
        "extracted_normalized": extracted_normalized,
        "correct": _equivalent_numeric(extracted, expected),
        "boxed_answer": bool(boxes),
        "boxed_count": len(boxes),
        "format_pass": len(boxes) == 1 and _equivalent_numeric(boxes[0], expected),
        "reasoning_structure_present": len(prefix_tokens) >= 8,
        "reasoning_prefix_tokens": len(prefix_tokens),
        "uses_think_tags": open_think > 0 or close_think > 0,
        "think_tags_balanced": open_think == close_think,
        "think_tag_pairs": min(open_think, close_think),
        "reasoning_channel_format_pass": reasoning_channel_format_pass,
        "contains_gsm8k_reference_marker": "####" in text,
        "repeated_4gram_fraction": round(repeated_fraction, 6),
        "high_repetition": repeated_fraction > 0.20,
        "finish_reason": finish_reason,
        "length_stopped": finish_reason == "length",
        "response_tokens": response_tokens,
    }


def summarize_reasoning_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no reasoning predictions")
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_prompt[str(record["id"])].append(record)
    sample_accuracy = fmean(bool(record["analysis"]["correct"]) for record in records)
    pass_at_k = fmean(any(bool(record["analysis"]["correct"]) for record in values) for values in by_prompt.values())
    majority_correct: list[bool] = []
    for values in by_prompt.values():
        answers = [str(value["analysis"]["extracted_normalized"]) for value in values]
        majority = Counter(answers).most_common(1)[0][0]
        expected = str(values[0]["analysis"]["expected_normalized"])
        majority_correct.append(_equivalent_numeric(majority, expected))
    response_token_values = [
        int(record["analysis"]["response_tokens"])
        for record in records
        if record["analysis"].get("response_tokens") is not None
    ]
    return {
        "samples": len(records),
        "prompts": len(by_prompt),
        "samples_per_prompt": sorted({len(values) for values in by_prompt.values()}),
        "sample_accuracy": sample_accuracy,
        "pass_at_k": pass_at_k,
        "majority_vote_accuracy": fmean(majority_correct),
        "boxed_answer_rate": fmean(bool(record["analysis"]["boxed_answer"]) for record in records),
        "correct_boxed_format_rate": fmean(bool(record["analysis"]["format_pass"]) for record in records),
        "reasoning_structure_rate": fmean(
            bool(record["analysis"]["reasoning_structure_present"]) for record in records
        ),
        "think_tag_use_rate": fmean(bool(record["analysis"].get("uses_think_tags")) for record in records),
        "reasoning_channel_format_rate": fmean(
            bool(record["analysis"].get("reasoning_channel_format_pass")) for record in records
        ),
        "unbalanced_think_tag_rate": fmean(
            not bool(record["analysis"]["think_tags_balanced"]) for record in records
        ),
        "gsm8k_reference_marker_rate": fmean(
            bool(record["analysis"]["contains_gsm8k_reference_marker"]) for record in records
        ),
        "high_repetition_rate": fmean(bool(record["analysis"]["high_repetition"]) for record in records),
        "length_stop_rate": fmean(bool(record["analysis"]["length_stopped"]) for record in records),
        "mean_response_tokens": fmean(response_token_values) if response_token_values else None,
    }


def _completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(str(row["id"]), int(row["sample_index"])) for row in _jsonl(path)}


def run_reasoning_eval(
    *,
    model: str,
    dataset: str | Path,
    output: str | Path,
    tokenizer: str | None = None,
    limit: int | None = None,
    limit_seed: int = 20260905,
    samples_per_prompt: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 1024,
    max_model_len: int = 3072,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.75,
    seed: int = 20260905,
) -> dict[str, Any]:
    if samples_per_prompt < 1 or batch_size < 1:
        raise ValueError("samples_per_prompt and batch_size must be positive")
    if temperature == 0 and samples_per_prompt != 1:
        raise ValueError("greedy evaluation requires samples_per_prompt=1")
    if max_model_len <= max_new_tokens:
        raise ValueError("max_model_len must exceed max_new_tokens")

    rows = _read_rows(dataset)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rng = random.Random(limit_seed)
        rows = rng.sample(rows, min(limit, len(rows)))
    required = {"id", "messages", "ground_truth"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"evaluation row {index} is missing columns: {sorted(missing)}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_digest = _sha256(dataset)
    selected_ids = "\n".join(str(row["id"]) for row in rows).encode()
    run_identity = {
        "schema_version": 1,
        "model": model,
        "tokenizer": tokenizer or model,
        "dataset": str(dataset),
        "dataset_sha256": dataset_digest,
        "selected_ids_sha256": sha256(selected_ids).hexdigest(),
        "limit": limit,
        "limit_seed": limit_seed,
        "samples_per_prompt": samples_per_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "seed": seed,
        "chat_template": "native tokenizer template",
    }
    run_path = output_path.with_suffix(".run.json")
    if run_path.exists():
        existing_identity = json.loads(run_path.read_text())
        if existing_identity != run_identity:
            raise ValueError(f"refusing to resume {output_path} with a different model, dataset, or protocol")
    elif output_path.exists() and output_path.stat().st_size:
        raise ValueError(f"refusing to resume {output_path} without its run identity file")
    else:
        run_path.write_text(json.dumps(run_identity, indent=2, sort_keys=True) + "\n")

    expected_samples = len(rows) * samples_per_prompt
    completed = _completed_keys(output_path)
    if len(completed) == expected_samples:
        summary_path = output_path.with_suffix(".summary.json")
        if summary_path.exists():
            return json.loads(summary_path.read_text())

    # Imported lazily so dataset preparation, analysis, and unit tests remain
    # usable on login nodes and developer machines without a GPU vLLM build.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer_source = tokenizer or model
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if not hf_tokenizer.chat_template:
        raise ValueError(f"tokenizer {tokenizer_source} has no chat template")
    engine = LLM(
        model=model,
        tokenizer=tokenizer_source,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        n=samples_per_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    with output_path.open("a") as sink:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            pending = [
                row
                for row in batch
                if any((str(row["id"]), sample_index) not in completed for sample_index in range(samples_per_prompt))
            ]
            if not pending:
                continue
            prompts = [
                hf_tokenizer.apply_chat_template(
                    row["messages"], add_generation_prompt=True, tokenize=False
                )
                for row in pending
            ]
            generations = engine.generate(prompts, sampling, use_tqdm=True)
            for row, request in zip(pending, generations, strict=True):
                for sample_index, generated in enumerate(request.outputs):
                    key = (str(row["id"]), sample_index)
                    if key in completed:
                        continue
                    analysis = analyze_reasoning_completion(
                        generated.text,
                        str(row["ground_truth"]),
                        finish_reason=str(generated.finish_reason or "unknown"),
                        response_tokens=len(generated.token_ids),
                    )
                    record = {
                        "id": key[0],
                        "sample_index": sample_index,
                        "prompt": row["messages"],
                        "text": generated.text,
                        "analysis": analysis,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    completed.add(key)

    records = list(_jsonl(output_path))
    if len(records) != expected_samples:
        raise ValueError(f"prediction file has {len(records)} rows; expected {expected_samples}")
    report = {
        "schema_version": 1,
        "diagnostic": limit is not None,
        "model": model,
        "tokenizer": tokenizer_source,
        "dataset": str(dataset),
        "dataset_sha256": dataset_digest,
        "protocol": {
            "limit": limit,
            "limit_seed": limit_seed,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "seed": seed,
            "chat_template": "native tokenizer template",
        },
        "metrics": summarize_reasoning_predictions(records),
        "predictions": str(output_path),
        "predictions_sha256": _sha256(output_path),
        "limitations": [
            "Correctness is numeric-answer equivalence under the oellm-rlvr lightweight verifier.",
            "Reasoning-structure metrics measure form, not semantic soundness; use the blinded audit pack.",
            "GSM8K is not a clean generalization estimate for checkpoints whose earlier data may contain GSM8K-derived rows.",
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _two_sided_binomial(discordant: int, smaller: int) -> float:
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def compare_reasoning_evals(
    baseline: str | Path,
    candidate: str | Path,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260905,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    before = {(str(row["id"]), int(row["sample_index"])): row for row in _jsonl(baseline)}
    after = {(str(row["id"]), int(row["sample_index"])): row for row in _jsonl(candidate)}
    if set(before) != set(after):
        raise ValueError("baseline and candidate must contain identical (id, sample_index) keys")
    keys = sorted(before)
    if not keys:
        raise ValueError("comparison inputs are empty")
    transitions = Counter(
        (bool(before[key]["analysis"]["correct"]), bool(after[key]["analysis"]["correct"])) for key in keys
    )
    paired = [
        float(bool(after[key]["analysis"]["correct"])) - float(bool(before[key]["analysis"]["correct"]))
        for key in keys
    ]
    rng = random.Random(seed)
    bootstrapped = sorted(
        fmean(paired[rng.randrange(len(paired))] for _ in paired) for _ in range(bootstrap_samples)
    )
    lower = bootstrapped[int(0.025 * (bootstrap_samples - 1))]
    upper = bootstrapped[int(0.975 * (bootstrap_samples - 1))]
    metrics_before = summarize_reasoning_predictions(list(before.values()))
    metrics_after = summarize_reasoning_predictions(list(after.values()))
    delta_keys = (
        "sample_accuracy",
        "pass_at_k",
        "majority_vote_accuracy",
        "boxed_answer_rate",
        "correct_boxed_format_rate",
        "reasoning_structure_rate",
        "think_tag_use_rate",
        "reasoning_channel_format_rate",
        "unbalanced_think_tag_rate",
        "gsm8k_reference_marker_rate",
        "high_repetition_rate",
        "length_stop_rate",
        "mean_response_tokens",
    )
    deltas = {
        key: (metrics_after[key] - metrics_before[key])
        if metrics_after[key] is not None and metrics_before[key] is not None
        else None
        for key in delta_keys
    }
    improvements = transitions[(False, True)]
    regressions = transitions[(True, False)]
    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "baseline_sha256": _sha256(baseline),
        "candidate": str(candidate),
        "candidate_sha256": _sha256(candidate),
        "paired_samples": len(keys),
        "baseline_metrics": metrics_before,
        "candidate_metrics": metrics_after,
        "deltas": deltas,
        "accuracy_delta_95pct_paired_bootstrap": [lower, upper],
        "transitions": {
            "both_wrong": transitions[(False, False)],
            "improved_wrong_to_right": improvements,
            "regressed_right_to_wrong": regressions,
            "both_right": transitions[(True, True)],
        },
        "mcnemar_exact_two_sided_p": _two_sided_binomial(improvements + regressions, min(improvements, regressions)),
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
    }


def build_blinded_reasoning_audit(
    baseline: str | Path,
    candidate: str | Path,
    output: str | Path,
    *,
    count: int = 24,
    seed: int = 20260905,
) -> dict[str, str]:
    if count < 1:
        raise ValueError("count must be positive")
    before = {(str(row["id"]), int(row["sample_index"])): row for row in _jsonl(baseline)}
    after = {(str(row["id"]), int(row["sample_index"])): row for row in _jsonl(candidate)}
    common = sorted(set(before) & set(after))
    if not common:
        raise ValueError("baseline and candidate have no common samples")
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in common:
        state = (bool(before[key]["analysis"]["correct"]), bool(after[key]["analysis"]["correct"]))
        label = {(False, False): "both_wrong", (False, True): "improved", (True, False): "regressed", (True, True): "both_right"}[state]
        buckets[label].append(key)
    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []
    target_per_bucket = max(1, count // 4)
    for label in ("improved", "regressed", "both_wrong", "both_right"):
        rng.shuffle(buckets[label])
        selected.extend(buckets[label][:target_per_bucket])
    remaining = [key for key in common if key not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    selected = selected[:count]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    key_path = destination.with_suffix(".key.json")
    key_rows: list[dict[str, Any]] = []
    document = [
        "# Blinded reasoning audit",
        "",
        (
            "Rate A and B independently. Do not infer quality from verbosity. For each answer record: problem "
            "setup (0–2), arithmetic/local steps (0–2), logical consistency (0–2), clarity (0–2), final-answer "
            "form (0–1), and a short error note. The automatic correctness label is deliberately hidden here."
        ),
        "",
    ]
    for index, key in enumerate(selected, start=1):
        pair = [("baseline", before[key]), ("candidate", after[key])]
        rng.shuffle(pair)
        mapping = {label: source for label, (source, _) in zip(("A", "B"), pair, strict=True)}
        key_rows.append({"item": index, "id": key[0], "sample_index": key[1], "mapping": mapping})
        document.extend(
            [
                f"## Item {index}",
                "",
                f"**Prompt:** {json.dumps(pair[0][1]['prompt'], ensure_ascii=False)}",
                "",
                f"**Answer A**\n\n{pair[0][1]['text']}",
                "",
                f"**Answer B**\n\n{pair[1][1]['text']}",
                "",
                "| Answer | Setup 0–2 | Steps 0–2 | Logic 0–2 | Clarity 0–2 | Form 0–1 | Error note |",
                "|---|---:|---:|---:|---:|---:|---|",
                "| A |  |  |  |  |  |  |",
                "| B |  |  |  |  |  |  |",
                "",
            ]
        )
    destination.write_text("\n".join(document) + "\n")
    key_path.write_text(json.dumps({"seed": seed, "items": key_rows}, indent=2, sort_keys=True) + "\n")
    return {"audit": str(destination), "key": str(key_path)}
