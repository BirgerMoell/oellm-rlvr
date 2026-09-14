from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

_LINGUA_NAMES = {
    "bs": "BOSNIAN",
    "bg": "BULGARIAN",
    "ca": "CATALAN",
    "cs": "CZECH",
    "da": "DANISH",
    "de": "GERMAN",
    "el": "GREEK",
    "en": "ENGLISH",
    "et": "ESTONIAN",
    "eu": "BASQUE",
    "fi": "FINNISH",
    "fr": "FRENCH",
    "ga": "IRISH",
    "hr": "CROATIAN",
    "hu": "HUNGARIAN",
    "is": "ICELANDIC",
    "it": "ITALIAN",
    "ka": "GEORGIAN",
    "lv": "LATVIAN",
    "lt": "LITHUANIAN",
    "mk": "MACEDONIAN",
    "nl": "DUTCH",
    "no": "BOKMAL",
    "pl": "POLISH",
    "pt": "PORTUGUESE",
    "ro": "ROMANIAN",
    "sk": "SLOVAK",
    "sl": "SLOVENE",
    "es": "SPANISH",
    "sq": "ALBANIAN",
    "sr": "SERBIAN",
    "sv": "SWEDISH",
    "tr": "TURKISH",
    "uk": "UKRAINIAN",
}
_NAME_TO_CODE = {name: code for code, name in _LINGUA_NAMES.items()}
_NAME_TO_CODE["NYNORSK"] = "no"
_ACCEPTABLE_EQUIVALENTS = {
    "bs": {"bs", "hr", "sr"},
    "hr": {"bs", "hr", "sr"},
    "sr": {"bs", "hr", "sr"},
}
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def reasoning_prose(text: str) -> str:
    """Extract the natural-language reasoning portion for language identification."""
    value = text.split("</think>", 1)[0]
    value = value.split("\\boxed{", 1)[0]
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _build_detector() -> Any:
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as error:
        raise RuntimeError(
            "language audit requires lingua-language-detector (install oellm-rlvr[eval])"
        ) from error
    languages = [getattr(Language, name) for name in sorted(set(_LINGUA_NAMES.values()))]
    languages.append(Language.NYNORSK)
    return LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()


def _detect(detector: Any, text: str) -> tuple[str | None, float | None]:
    values = detector.compute_language_confidence_values(text)
    if not values:
        return None, None
    best = values[0]
    code = _NAME_TO_CODE.get(best.language.name)
    return code, float(best.value)


def audit_reasoning_languages(
    predictions: str | Path,
    output: str | Path,
    *,
    minimum_confidence: float = 0.55,
) -> dict[str, Any]:
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in [0, 1]")
    detector = _build_detector()
    records = []
    with Path(predictions).open() as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("prediction file is empty")

    results: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        target = str(metadata.get("language") or "")
        prose = reasoning_prose(str(record.get("text") or ""))
        supported = target in _LINGUA_NAMES
        enough_text = sum(1 for value in prose if _LETTER.match(value)) >= 20
        detected, confidence = _detect(detector, prose) if supported and enough_text else (None, None)
        match = None
        if detected is not None and confidence is not None:
            match = detected in _ACCEPTABLE_EQUIVALENTS.get(target, {target}) and confidence >= minimum_confidence
        results.append(
            {
                "id": str(record.get("id")),
                "sample_index": int(record.get("sample_index", 0)),
                "target_language": target,
                "supported": supported,
                "detected_language": detected,
                "confidence": confidence,
                "target_language_match": match,
                "reasoning_excerpt": prose[:240],
            }
        )

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_language[result["target_language"]].append(result)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [value for value in values if value["target_language_match"] is not None]
        return {
            "samples": len(values),
            "scored_samples": len(scored),
            "match_rate": (
                fmean(bool(value["target_language_match"]) for value in scored) if scored else None
            ),
            "mean_confidence": (
                fmean(float(value["confidence"]) for value in scored) if scored else None
            ),
        }

    scored_all = [value for value in results if value["target_language_match"] is not None]
    report = {
        "schema_version": 1,
        "predictions": str(predictions),
        "minimum_confidence": minimum_confidence,
        "detector": "lingua-language-detector",
        "samples": len(results),
        "scored_samples": len(scored_all),
        "unsupported_target_languages": sorted(
            {value["target_language"] for value in results if not value["supported"]}
        ),
        "target_language_match_rate": (
            fmean(bool(value["target_language_match"]) for value in scored_all) if scored_all else None
        ),
        "by_language": {
            language: summarize(values) for language, values in sorted(by_language.items())
        },
        "first_mismatch_by_language": {
            language: next(
                (
                    value
                    for value in values
                    if value["target_language_match"] is False
                ),
                None,
            )
            for language, values in sorted(by_language.items())
        },
        "limitations": [
            "This is an automatic diagnostic, not a reward and not a substitute for native review.",
            "Lingua does not support Galician or Maltese; those targets require a different detector or manual audit.",
            "Bosnian, Croatian, and Serbian are accepted as one BCMS equivalence group for this coarse gate.",
        ],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report
