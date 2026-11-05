#!/usr/bin/env python3
"""Fetch two-hop facts for the deep_search suite from Wikidata (CC0): subject -> bridge -> answer, e.g. a film ->
its director -> the director's birth year. Answering needs two lookups (find the bridge in the subject's article, then
the answer in the bridge's article), which is what the wiki_search / open_page / find_in_page tools are for.

Per chain type x 17 countries x 3 popularity bands of the subject (as in fetch_eu_qa_wikidata.py) one SPARQL query;
then labels, aliases and Wikipedia sitelinks (the ten search languages) of every subject, bridge and answer.
Writes <out>/facts.jsonl and <out>/entities.json; re-running skips finished queries.
  python scripts/fetch_deep_search_wikidata.py --out data/deep-search-raw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eu_qa_wikidata import API, BANDS, CITIZENSHIP, COUNTRIES, LABEL_LANGS, SPARQL, get, qid  # noqa: E402

SITES = "|".join(f"{lang}wiki" for lang in ["en", "de", "fr", "es", "it", "nl", "sv", "pl", "pt", "fi"])
BIRTH_YEAR = "?o p:P569/psv:P569 ?av. ?av wikibase:timeValue ?a; wikibase:timePrecision ?prec. FILTER(?prec >= 9)"
# ?s subject, ?sl its sitelinks, ?o bridge, ?a answer (time value or entity), ?t subject year (film release)
QUERIES = {
    "film_director_birth_year": "SELECT ?s ?sl ?o ?a ?t WHERE {{ ?s wdt:P31 wd:Q11424; wdt:P495 wd:{c}; wikibase:sitelinks ?sl. "
                                "FILTER(?sl >= {lo} && ?sl < {hi}) ?s wdt:P57 ?o; wdt:P577 ?t. " + BIRTH_YEAR + " }} LIMIT {limit}",
    "film_director_birth_place": "SELECT ?s ?sl ?o ?a ?t WHERE {{ ?s wdt:P31 wd:Q11424; wdt:P495 wd:{c}; wikibase:sitelinks ?sl. "
                                 "FILTER(?sl >= {lo} && ?sl < {hi}) ?s wdt:P57 ?o; wdt:P577 ?t. ?o wdt:P19 ?a. }} LIMIT {limit}",
    "book_author_birth_year": "SELECT ?s ?sl ?o ?a WHERE {{ ?s wdt:P31 wd:Q7725634; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                              "?s wdt:P50 ?o. VALUES ?cc {{ {cc} }} ?o wdt:P27 ?cc. " + BIRTH_YEAR + " }} LIMIT {limit}",
    "book_author_birth_place": "SELECT ?s ?sl ?o ?a WHERE {{ ?s wdt:P31 wd:Q7725634; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                               "?s wdt:P50 ?o. VALUES ?cc {{ {cc} }} ?o wdt:P27 ?cc. ?o wdt:P19 ?a. }} LIMIT {limit}",
    "org_founder_birth_year": "SELECT ?s ?sl ?o ?a WHERE {{ ?s wdt:P17 wd:{c}; wdt:P112 ?o; wikibase:sitelinks ?sl. FILTER(?sl >= {lo} && ?sl < {hi}) "
                              "?o wdt:P31 wd:Q5. " + BIRTH_YEAR + " }} LIMIT {limit}",
    "building_architect_birth_year": "SELECT ?s ?sl ?o ?a WHERE {{ ?s wdt:P17 wd:{c}; wdt:P84 ?o; wikibase:sitelinks ?sl. "
                                     "FILTER(?sl >= {lo} && ?sl < {hi}) ?o wdt:P31 wd:Q5. " + BIRTH_YEAR + " }} LIMIT {limit}",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
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
                    rows = get(SPARQL, {"query": q}).get("results", {}).get("bindings", [])
                    for b in rows:
                        av = b["a"]["value"]
                        out.write(json.dumps({"type": qtype, "country": c, "band": band, "s": qid(b["s"]["value"]),
                                              "sl": int(b["sl"]["value"]), "o": qid(b["o"]["value"]),
                                              "a": av if b["a"].get("datatype", "").endswith("dateTime") else qid(av),
                                              "t": b["t"]["value"] if "t" in b else None}) + "\n")
                    if not rows:
                        out.write(json.dumps({"type": qtype, "country": c, "band": band, "s": None}) + "\n")
                    out.flush()
                    print(f"{qtype:30} {COUNTRIES[c]} {band:5} {len(rows):4} rows {time.time() - t0:5.1f}s", flush=True)
                    time.sleep(1.0)
    ids = set(COUNTRIES)
    for line in facts_path.open():
        r = json.loads(line)
        ids.update(v for k, v in r.items() if k in ("s", "o", "a") and v and str(v).startswith("Q"))
    ent_path = a.out / "entities.json"
    entities = json.loads(ent_path.read_text()) if ent_path.exists() else {}
    todo = sorted(i for i in ids if i not in entities)
    print(f"fetching labels and sitelinks for {len(todo)} entities", flush=True)
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        res = get(API, {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases|sitelinks",
                        "languages": LABEL_LANGS, "sitefilter": SITES, "format": "json"})
        for e, v in res.get("entities", {}).items():
            entities[e] = {"labels": {l: x["value"] for l, x in v.get("labels", {}).items()},
                           "aliases": {l: [x["value"] for x in xs] for l, xs in v.get("aliases", {}).items()},
                           "sitelinks": {s[:-4]: x["title"] for s, x in v.get("sitelinks", {}).items()}}
        if k % 2000 == 0:
            ent_path.write_text(json.dumps(entities, ensure_ascii=False))
            print(f"  {k + len(batch)}/{len(todo)}", flush=True)
        time.sleep(0.3)
    ent_path.write_text(json.dumps(entities, ensure_ascii=False))
    print(f"done: {len(entities)} entities", flush=True)


if __name__ == "__main__":
    main()
