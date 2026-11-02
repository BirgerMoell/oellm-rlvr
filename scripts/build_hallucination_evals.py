#!/usr/bin/env python3
"""Build the hallucination eval set hallu-v1 from the eu-tasks eval (see oellm_rlvr.hallucination).

Rows carry a ``condition``, which eval_eu_tasks.py (qwen35-posttrain) reports separately:
- ``standard``:  the eu_qa eval questions as they are (UNKNOWN allowed, wrong counts against);
- ``t50`` / ``t75`` / ``t90``: the same questions with a stated confidence target and penalty t/(1-t) (1, 3, 9);
- ``fabricated``: questions about nonexistent entities (correct behaviour: ANSWER: UNKNOWN).
With --check-wikidata (login node; needs network) every fabricated subject is searched on Wikidata in the row's
language and English, and rows whose subject matches a Wikidata label or alias exactly are dropped.
  python scripts/build_hallucination_evals.py --eu-eval data/eu-tasks/eval.jsonl --out data/hallu-v1/eval.jsonl \
      --fabricated-per-language 50 --check-wikidata
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oellm_rlvr import eu_tasks as E  # noqa: E402
from oellm_rlvr.hallucination import fabricated_qa, threshold_prompt  # noqa: E402

API = "https://www.wikidata.org/w/api.php"
UA = "oellm-hallucination-eval-builder/0.1 (https://github.com/OpenEuroLLM; research eval construction)"
THRESHOLDS = {"t50": 0.5, "t75": 0.75, "t90": 0.9}


def wikidata_knows(subject: str, lang: str) -> bool:
    for search_lang in dict.fromkeys([lang, "en"]):
        query = urllib.parse.urlencode({"action": "wbsearchentities", "search": subject, "language": search_lang,
                                        "limit": 5, "format": "json"})
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    hits = json.loads(r.read()).get("search", [])
                break
            except Exception as e:  # timeouts, 429
                print(f"  retry {attempt + 1}: {str(e)[:100]}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Wikidata search failed for {subject!r}")
        target = E.norm_text(subject)
        for hit in hits:
            names = [hit.get("label", ""), hit.get("match", {}).get("text", "")] + hit.get("aliases", [])
            if any(E.norm_text(n) == target for n in names if n):
                return True
        time.sleep(0.2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu-eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fabricated-per-language", type=int, default=50)
    ap.add_argument("--check-wikidata", action="store_true")
    a = ap.parse_args()

    qa = [r for r in map(json.loads, a.eu_eval.read_text().splitlines()) if r["suite"] == "qa"]
    rows = [{**r, "condition": "standard"} for r in qa]
    for cond, t in THRESHOLDS.items():
        rows += [{**r, "id": f"{r['id']}-{cond}", "condition": cond, "prompt": threshold_prompt(r["prompt"], r["language"], t)}
                 for r in qa]
    # Over-generate, then keep the first N per language that Wikidata does not know.
    fabricated = fabricated_qa(2 * a.fabricated_per_language)
    kept, dropped, per_lang = [], [], Counter()
    for r in fabricated:
        if per_lang[r["language"]] >= a.fabricated_per_language:
            continue
        if a.check_wikidata and wikidata_knows(r["subject"], r["language"]):
            dropped.append(r["subject"])
            continue
        per_lang[r["language"]] += 1
        kept.append({**r, "condition": "fabricated", "wikidata_checked": a.check_wikidata})
    rows += kept
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"kind": "oellm_hallucination_eval_v1", "source": str(a.eu_eval), "rows": len(rows),
                "by_condition": dict(Counter(r["condition"] for r in rows)),
                "fabricated_by_language": dict(per_lang), "fabricated_dropped_wikidata": dropped,
                "wikidata_checked": a.check_wikidata}
    a.out.with_name("manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "fabricated_dropped_wikidata"}, indent=1),
          f"dropped {len(dropped)}: {dropped[:10]}")


if __name__ == "__main__":
    main()
