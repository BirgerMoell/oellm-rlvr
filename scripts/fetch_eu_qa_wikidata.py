#!/usr/bin/env python3
"""Fetch the raw facts for the eu_qa suite from Wikidata (CC0): see oellm_rlvr.eu_tasks.

For 7 question types x 17 European countries x 3 popularity bands (subject sitelinks: head >= 30, torso 10-29,
tail 3-9) one SPARQL query each returns up to LIMIT candidate facts; facts with more than one distinct answer
(several birth years, two directors, ...) are dropped at build time. Then labels and aliases of every subject, answer,
occupation, class and country are fetched in the ten suite languages plus ``mul`` with ``wbgetentities``.

Writes <out>/facts.jsonl (one row per query result) and <out>/entities.json. Re-running skips finished queries.
  python scripts/fetch_eu_qa_wikidata.py --out data/eu-qa-raw
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "oellm-eu-qa-builder/0.1 (https://github.com/OpenEuroLLM; research eval construction)"
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
COUNTRIES = {"Q34": "SE", "Q33": "FI", "Q35": "DK", "Q183": "DE", "Q142": "FR", "Q29": "ES", "Q38": "IT", "Q55": "NL",
             "Q36": "PL", "Q45": "PT", "Q40": "AT", "Q31": "BE", "Q213": "CZ", "Q28": "HU", "Q218": "RO", "Q41": "GR",
             "Q27": "IE"}
BANDS = {"head": (30, 100000), "torso": (10, 30), "tail": (3, 10)}
LABEL_LANGS = "en|de|fr|es|it|nl|sv|pl|pt|fi|mul|da|cs|hu|ro|el|ga"
# Citizenship (P27) of Danes and Dutch people is often the kingdom, not the constituent country.
CITIZENSHIP = {"Q35": "wd:Q35 wd:Q756617", "Q55": "wd:Q55 wd:Q29999"}

YEAR = "?v wikibase:timeValue ?t; wikibase:timePrecision ?prec. FILTER(?prec >= 9)"
QUERIES = {
    "birth_year": "SELECT ?s ?sl ?t ?x WHERE {{ VALUES ?cc {{ {cc} }} ?s wdt:P27 ?cc; wdt:P31 wd:Q5; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                  "?s p:P569/psv:P569 ?v. " + YEAR + " ?s wdt:P106 ?x. }} LIMIT {limit}",
    "birth_place": "SELECT ?s ?sl ?o ?x WHERE {{ VALUES ?cc {{ {cc} }} ?s wdt:P27 ?cc; wdt:P31 wd:Q5; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                   "?s wdt:P19 ?o; wdt:P106 ?x. }} LIMIT {limit}",
    "admin_capital": "SELECT ?s ?sl ?o WHERE {{ ?s wdt:P17 wd:{c}; wdt:P36 ?o; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                     "FILTER(?s != wd:{c}) FILTER NOT EXISTS {{ ?s wdt:P31 wd:Q6256 }} }} LIMIT {limit}",
    "film_director": "SELECT ?s ?sl ?o ?t WHERE {{ ?s wdt:P31 wd:Q11424; wdt:P495 wd:{c}; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                     "?s wdt:P57 ?o; wdt:P577 ?t. }} LIMIT {limit}",
    "book_author": "SELECT ?s ?sl ?o WHERE {{ ?s wdt:P31 wd:Q7725634; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                   "?s wdt:P50 ?o. VALUES ?cc {{ {cc} }} ?o wdt:P27 ?cc. }} LIMIT {limit}",
    "org_founded": "SELECT ?s ?sl ?t ?x WHERE {{ VALUES ?x {{ wd:Q3918 wd:Q476028 wd:Q4830453 wd:Q33506 }} ?s wdt:P31 ?x; wdt:P17 wd:{c}; "
                   "wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) ?s p:P571/psv:P571 ?v. " + YEAR + " }} LIMIT {limit}",
    "river_mouth": "SELECT ?s ?sl ?o WHERE {{ ?s wdt:P31 wd:Q4022; wdt:P17 wd:{c}; wdt:P403 ?o; wikibase:sitelinks ?sl. "
                   "FILTER(?sl >= {lo} && ?sl < {hi}) }} LIMIT {limit}",
}


def get(url: str, data: dict | None = None, timeout: int = 90) -> dict:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # timeouts, 429, 5xx
            print(f"  retry {attempt + 1}: {str(e)[:120]}", flush=True)
            time.sleep(10 * (attempt + 1))
    return {}


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    facts_path = a.out / "facts.jsonl"
    done = set()
    if facts_path.exists():
        for line in facts_path.open():
            r = json.loads(line)
            done.add((r["type"], r["country"], r["band"]))
    with facts_path.open("a") as out:
        for qtype, template in QUERIES.items():
            for c in COUNTRIES:
                for band, (lo, hi) in BANDS.items():
                    if (qtype, c, band) in done:
                        continue
                    q = template.format(c=c, cc=CITIZENSHIP.get(c, f"wd:{c}"), lo=lo, hi=hi, limit=a.limit)
                    t0 = time.time()
                    res = get(SPARQL, {"query": q})
                    rows = res.get("results", {}).get("bindings", [])
                    for b in rows:
                        out.write(json.dumps({"type": qtype, "country": c, "band": band, "s": qid(b["s"]["value"]),
                                              "sl": int(b["sl"]["value"]), "o": qid(b["o"]["value"]) if "o" in b else None,
                                              "t": b["t"]["value"] if "t" in b else None,
                                              "x": qid(b["x"]["value"]) if "x" in b else None}) + "\n")
                    if not rows:  # mark the query as done even when empty
                        out.write(json.dumps({"type": qtype, "country": c, "band": band, "s": None}) + "\n")
                    out.flush()
                    print(f"{qtype:14} {COUNTRIES[c]} {band:5} {len(rows):4} rows {time.time() - t0:5.1f}s", flush=True)
                    time.sleep(1.0)
    ids = set(COUNTRIES)
    for line in facts_path.open():
        r = json.loads(line)
        ids.update(v for k, v in r.items() if k in ("s", "o", "x") and v)
    ent_path = a.out / "entities.json"
    entities = json.loads(ent_path.read_text()) if ent_path.exists() else {}
    todo = sorted(i for i in ids if i not in entities)
    print(f"fetching labels for {len(todo)} entities", flush=True)
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        res = get(API, {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases", "languages": LABEL_LANGS,
                        "format": "json"})
        for e, v in res.get("entities", {}).items():
            entities[e] = {"labels": {l: x["value"] for l, x in v.get("labels", {}).items()},
                           "aliases": {l: [x["value"] for x in xs] for l, xs in v.get("aliases", {}).items()}}
        if k % 2000 == 0:
            ent_path.write_text(json.dumps(entities, ensure_ascii=False))
            print(f"  {k + len(batch)}/{len(todo)}", flush=True)
        time.sleep(0.3)
    ent_path.write_text(json.dumps(entities, ensure_ascii=False))
    print(f"done: {len(entities)} entities", flush=True)


if __name__ == "__main__":
    main()
