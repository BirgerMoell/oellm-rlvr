from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import write_rows

ISO639_3_TO_1 = {
    "bos": "bs",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "est": "et",
    "eus": "eu",
    "fin": "fi",
    "fra": "fr",
    "gle": "ga",
    "glg": "gl",
    "hrv": "hr",
    "hun": "hu",
    "isl": "is",
    "ita": "it",
    "kat": "ka",
    "lav": "lv",
    "lit": "lt",
    "mkd": "mk",
    "mlt": "mt",
    "nld": "nl",
    "nor": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "sqi": "sq",
    "srp": "sr",
    "swe": "sv",
    "tur": "tr",
    "ukr": "uk",
}

FALLBACK_HEADERS = {
    "en": (
        "Solve the following reasoning problem. Give a concise derivation and put only the final answer "
        "in \\boxed{}."
    ),
    "ka": (
        "ამოხსენით შემდეგი მსჯელობის ამოცანა. წარმოადგინეთ მოკლე დასაბუთება და მხოლოდ საბოლოო პასუხი "
        "ჩასვით \\boxed{}-ში."
    ),
}

FORMAT_CONTRACT = (
    "Keep the reasoning concise and write it in the same language as the instruction above. "
    "Return exactly:\n<think>\n…\n</think>\n\\boxed{…}"
)
GENERATOR_VERSION = "oellm-symbolic-reasoning-canary-v2"
BASIC_FAMILIES = (
    "linear_equation",
    "recurrence",
    "bit_count",
    "base_conversion",
    "boolean_logic",
    "modular_chain",
    "gcd",
    "mixed_arithmetic",
)
CHALLENGE_FAMILIES = (
    "affine_mod_chain",
    "weighted_checksum",
    "subset_sum_count",
    "logic_assignment_count",
    "second_order_recurrence",
    "modular_power",
    "base_digit_checksum",
    "linear_system_checksum",
)


@dataclass(frozen=True)
class TargetLanguage:
    macro: str
    name: str
    variants: tuple[str, ...]
    alpha2: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_language_contract(path: str | Path) -> list[TargetLanguage]:
    """Parse the canonical OpenEuroLLM ``languages`` file."""
    source = Path(path)
    targets: list[TargetLanguage] = []
    for line_number, raw in enumerate(source.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(":", 2)]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"invalid language contract line {line_number}: {raw!r}")
        macro, name, variants_raw = parts
        if macro not in ISO639_3_TO_1:
            raise ValueError(f"no ISO-639-1 runtime mapping for canonical language {macro!r}")
        variants = tuple(variants_raw.split())
        if not variants:
            raise ValueError(f"canonical language {macro!r} has no internal variants")
        targets.append(TargetLanguage(macro, name, variants, ISO639_3_TO_1[macro]))
    if not targets:
        raise ValueError("language contract is empty")
    if len({target.macro for target in targets}) != len(targets):
        raise ValueError("language contract contains duplicate macro-language codes")
    return targets


def _read_reference_headers(path: Path) -> dict[str, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("multilingual reasoning preparation requires pyarrow") from error

    candidates: dict[str, list[str]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["language", "messages"], batch_size=2048):
        for row in batch.to_pylist():
            language = str(row.get("language", ""))
            messages = row.get("messages") or []
            if not language or not messages or messages[0].get("role") != "user":
                continue
            content = str(messages[0].get("content", "")).strip()
            header = content.split("\n\n", 1)[0].strip()
            if header:
                candidates.setdefault(language, []).append(header)
    return {
        language: min(set(headers), key=lambda value: (len(value), value))
        for language, headers in candidates.items()
    }


def _base_digits(value: int, base: int) -> str:
    alphabet = "0123456789ABCDEF"
    digits = ""
    current = value
    while current:
        current, remainder = divmod(current, base)
        digits = alphabet[remainder] + digits
    return digits or "0"


def _task(family: str, rng: random.Random) -> tuple[str, str, int, dict[str, Any]]:
    if family == "linear_equation":
        x = rng.randint(2, 30)
        coefficient = rng.randint(2, 9)
        offset = rng.randint(1, 20)
        total = coefficient * x + offset
        return f"{coefficient}·x + {offset} = {total};  x = ?", str(x), 1, {
            "coefficient": coefficient,
            "offset": offset,
            "total": total,
        }
    if family == "recurrence":
        start = rng.randint(1, 20)
        delta = rng.randint(2, 12)
        index = rng.randint(5, 12)
        return (
            f"a₀ = {start};  aₙ₊₁ = aₙ + {delta};  a{index} = ?",
            str(start + index * delta),
            2,
            {"start": start, "delta": delta, "index": index},
        )
    if family == "bit_count":
        value = rng.randint(64, 4095)
        bits = format(value, "b")
        return f"popcount₂({bits}) = ?", str(bits.count("1")), 2, {"value": value}
    if family == "base_conversion":
        value = rng.randint(32, 2047)
        base = rng.choice((2, 3, 4, 5, 8))
        digits = _base_digits(value, base)
        return f"({digits}){base} = (?)10", str(value), 2, {"value": value, "base": base}
    if family == "boolean_logic":
        left, middle, right = (rng.randint(0, 1) for _ in range(3))
        answer = int((bool(left) and not bool(middle)) or bool(right))
        return (
            f"({left} ∧ ¬{middle}) ∨ {right} = ?   [0/1]",
            str(answer),
            2,
            {"left": left, "middle": middle, "right": right},
        )
    if family == "modular_chain":
        left = rng.randint(12, 99)
        right = rng.randint(7, 31)
        offset = rng.randint(2, 40)
        modulus = rng.randint(5, 19)
        answer = (left * right + offset) % modulus
        return (
            f"(({left}·{right}) + {offset}) mod {modulus} = ?",
            str(answer),
            3,
            {"left": left, "right": right, "offset": offset, "modulus": modulus},
        )
    if family == "gcd":
        factor = rng.randint(3, 20)
        left_factor = rng.choice((5, 7, 11, 13))
        right_factor = rng.choice((17, 19, 23, 29))
        left, right = factor * left_factor, factor * right_factor
        return f"gcd({left}, {right}) = ?", str(math.gcd(left, right)), 2, {
            "left": left,
            "right": right,
        }
    if family == "mixed_arithmetic":
        first = rng.randint(10, 90)
        second = rng.randint(5, 40)
        third = rng.randint(2, 12)
        offset = rng.randint(1, 30)
        return (
            f"({first} + {second})·{third} − {offset} = ?",
            str((first + second) * third - offset),
            3,
            {"first": first, "second": second, "third": third, "offset": offset},
        )
    if family == "affine_mod_chain":
        modulus = rng.choice((97, 101, 103, 107, 109, 127))
        value = rng.randint(2, modulus - 1)
        start = value
        operations = [(rng.randint(2, 13), rng.randint(3, 47)) for _ in range(8)]
        for multiplier, offset in operations:
            value = (multiplier * value + offset) % modulus
        program = "; ".join(f"x ← ({a}·x + {b}) mod {modulus}" for a, b in operations)
        return f"x₀ = {start}; apply in order: {program}. Final x = ?", str(value), 4, {
            "start": start,
            "modulus": modulus,
            "operations": operations,
        }
    if family == "weighted_checksum":
        digits = [rng.randint(0, 9) for _ in range(12)]
        answer = sum((index + 1) * digit for index, digit in enumerate(digits)) % 97
        return (
            "d = ["
            + ", ".join(str(value) for value in digits)
            + "];  (Σᵢ₌₁¹² i·dᵢ) mod 97 = ?",
            str(answer),
            4,
            {"digits": digits, "modulus": 97},
        )
    if family == "subset_sum_count":
        values = rng.sample(range(2, 31), 10)
        chosen = rng.sample(range(len(values)), 4)
        target = sum(values[index] for index in chosen)
        count = sum(
            sum(value for value, take in zip(values, mask, strict=True) if take) == target
            for mask in itertools.product((0, 1), repeat=len(values))
        )
        return (
            (
                f"S = {{{', '.join(str(value) for value in values)}}}. "
                f"How many subsets of S have sum {target}?"
            ),
            str(count),
            5,
            {"values": values, "target": target},
        )
    if family == "logic_assignment_count":
        xor_value = rng.randint(0, 1)
        implication_value = rng.randint(0, 1)
        count = 0
        for a, b, c, d, e in itertools.product((False, True), repeat=5):
            valid = (
                ((a ^ b) == bool(xor_value))
                and (c or not d)
                and (e == (a and c))
                and (((not b) or d) == bool(implication_value))
            )
            count += int(valid)
        return (
            (
                "For A,B,C,D,E ∈ {0,1}, count the assignments satisfying "
                f"(A ⊕ B)={xor_value}, (C ∨ ¬D)=1, E=(A ∧ C), "
                f"and (B → D)={implication_value}."
            ),
            str(count),
            5,
            {"xor_value": xor_value, "implication_value": implication_value},
        )
    if family == "second_order_recurrence":
        values = [rng.randint(1, 12), rng.randint(4, 18)]
        offset = rng.randint(1, 7)
        index = rng.randint(9, 13)
        while len(values) <= index:
            values.append(2 * values[-1] - values[-2] + offset)
        return (
            f"a₀={values[0]}, a₁={values[1]}, aₙ=2aₙ₋₁−aₙ₋₂+{offset}; a{index}=?",
            str(values[index]),
            4,
            {"a0": values[0], "a1": values[1], "offset": offset, "index": index},
        )
    if family == "modular_power":
        first = rng.randint(7, 80)
        second = rng.randint(5, 60)
        exponent_a = rng.randint(25, 90)
        exponent_b = rng.randint(17, 70)
        modulus = rng.choice((97, 101, 103, 107, 109, 127))
        answer = (pow(first, exponent_a, modulus) + pow(second, exponent_b, modulus)) % modulus
        return (
            f"({first}^{exponent_a} + {second}^{exponent_b}) mod {modulus} = ?",
            str(answer),
            5,
            {
                "first": first,
                "second": second,
                "exponent_a": exponent_a,
                "exponent_b": exponent_b,
                "modulus": modulus,
            },
        )
    if family == "base_digit_checksum":
        value = rng.randint(10_000, 2_000_000)
        base = rng.randint(3, 9)
        digits = _base_digits(value, base)
        numeric_digits = [int(digit, 16) for digit in digits]
        answer = sum((index + 1) * digit for index, digit in enumerate(reversed(numeric_digits))) % 97
        return (
            (
                f"Write {value} in base {base} as digits dₖ…d₀, then compute "
                "(Σᵢ₌₀ᵏ (i+1)·dᵢ) mod 97. Result = ?"
            ),
            str(answer),
            5,
            {"value": value, "base": base, "digits": digits},
        )
    if family == "linear_system_checksum":
        x = rng.randint(-20, 30)
        y = rng.randint(-20, 30)
        while True:
            a, b, c, d = (rng.randint(2, 11) for _ in range(4))
            if a * d != b * c:
                break
        first = a * x + b * y
        second = c * x + d * y
        answer = 3 * x - 2 * y
        return (
            f"{a}x+{b}y={first}; {c}x+{d}y={second}; compute 3x−2y.",
            str(answer),
            4,
            {"a": a, "b": b, "c": c, "d": d, "x": x, "y": y},
        )
    raise ValueError(f"unknown reasoning family {family!r}")


def _build_rows(
    targets: list[TargetLanguage],
    headers: dict[str, str],
    *,
    count_per_language: int,
    seed: int,
    split: str,
    families: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language_index, target in enumerate(targets):
        for item_index in range(count_per_language):
            family = families[(language_index + item_index) % len(families)]
            row_seed = seed + language_index * 1_000_003 + item_index
            question, answer, difficulty, parameters = _task(family, random.Random(row_seed))
            content = f"{headers[target.alpha2]}\n\n{question}\n\n{FORMAT_CONTRACT}"
            identity = f"{split}:{target.macro}:{family}:{row_seed}"
            row_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
            rows.append(
                {
                    "id": row_id,
                    "messages": [{"role": "user", "content": content}],
                    "ground_truth": answer,
                    "dataset": "math",
                    "split": split,
                    "language": target.alpha2,
                    "canonical_language": target.macro,
                    "canonical_variant": target.variants[0],
                    "language_name": target.name,
                    "domain": "general_reasoning",
                    "subdomain": family,
                    "difficulty": difficulty,
                    "difficulty_label": {
                        1: "easy",
                        2: "medium",
                        3: "hard",
                        4: "challenging",
                        5: "very_hard",
                    }[difficulty],
                    "verifier_kind": "integer_exact",
                    "verifier_version": "oellm-math-verifier-contract-0.1.0",
                    "generator_family": family,
                    "generator_version": GENERATOR_VERSION,
                    "generation_seed": row_seed,
                    "semantic_group_id": f"oellm-reasoning-{row_id}",
                    "parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
                    "canonical_answer": answer,
                    "source": "deterministic_symbolic_generation",
                    "source_license": "Apache-2.0",
                    "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "contamination_group": f"generator:{family}:{GENERATOR_VERSION}",
                    "oellm_source_dataset": "oellm-symbolic-reasoning-canary",
                }
            )
    return rows


def build_multilingual_reasoning_canary(
    *,
    language_contract: str | Path,
    reference_traces: str | Path,
    output_dir: str | Path,
    contract_revision: str,
    reference_revision: str,
    train_per_language: int = 64,
    eval_per_language: int = 4,
    profile_per_language: int = 1,
    families: tuple[str, ...] = BASIC_FAMILIES,
    seed: int = 20260914,
) -> dict[str, Any]:
    """Build deterministic train/profile/eval pools over every macro-language."""
    if train_per_language < 1 or eval_per_language < 1 or profile_per_language < 1:
        raise ValueError("per-language pool sizes must be positive")
    if profile_per_language > eval_per_language:
        raise ValueError("profile_per_language cannot exceed eval_per_language")
    if not families:
        raise ValueError("at least one reasoning family is required")
    unknown_families = set(families) - set(BASIC_FAMILIES) - set(CHALLENGE_FAMILIES)
    if unknown_families:
        raise ValueError(f"unknown reasoning families: {sorted(unknown_families)}")
    if not contract_revision or not reference_revision:
        raise ValueError("source revisions must be non-empty")

    contract_path = Path(language_contract)
    reference_path = Path(reference_traces)
    targets = parse_language_contract(contract_path)
    reference_headers = _read_reference_headers(reference_path)
    headers = {**reference_headers, **FALLBACK_HEADERS}
    missing = [target.alpha2 for target in targets if target.alpha2 not in headers]
    if missing:
        raise ValueError(f"no localized prompt header for canonical languages: {missing}")

    train_rows = _build_rows(
        targets,
        headers,
        count_per_language=train_per_language,
        seed=seed,
        split="train",
        families=families,
    )
    eval_rows = _build_rows(
        targets,
        headers,
        count_per_language=eval_per_language,
        seed=seed + 10_000_019,
        split="evaluation",
        families=families,
    )
    profile_rows = [
        row for index, row in enumerate(eval_rows) if index % eval_per_language < profile_per_language
    ]
    train_ids = {row["id"] for row in train_rows}
    eval_ids = {row["id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise AssertionError("training and evaluation identities overlap")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": destination / "train.parquet",
        "profile": destination / "profile.parquet",
        "evaluation": destination / "evaluation.parquet",
    }
    write_rows(train_rows, paths["train"])
    write_rows(profile_rows, paths["profile"])
    write_rows(eval_rows, paths["evaluation"])

    variant_count = sum(len(target.variants) for target in targets)
    header_sources = {
        target.alpha2: "fallback" if target.alpha2 in FALLBACK_HEADERS else "translated_reference"
        for target in targets
    }
    manifest = {
        "schema": "oellm-multilingual-reasoning-canary-v1",
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "language_contract": {
            "path": str(contract_path),
            "revision": contract_revision,
            "sha256": _sha256(contract_path),
            "macro_languages": len(targets),
            "internal_variants": variant_count,
            "trained_variants": len(targets),
        },
        "localized_header_reference": {
            "path": str(reference_path),
            "revision": reference_revision,
            "sha256": _sha256(reference_path),
            "source_languages_used": dict(sorted(Counter(header_sources.values()).items())),
        },
        "format_contract": FORMAT_CONTRACT,
        "families": sorted({row["generator_family"] for row in train_rows}),
        "profile_per_language": profile_per_language,
        "artifacts": {
            name: {
                "path": str(path),
                "rows": {"train": len(train_rows), "profile": len(profile_rows), "evaluation": len(eval_rows)}[
                    name
                ],
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "train_evaluation_id_overlap": 0,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
