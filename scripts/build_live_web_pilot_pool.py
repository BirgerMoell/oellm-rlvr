#!/usr/bin/env python3
"""Live-web pilot pool: N two-hop deep_search questions (default mode, the three browsing tools) that neither the
warm-start trajectories nor the E10-web pool used, an equal share per language. The answers come from Wikidata
(build_deep_search.py), the tools answer from live Wikipedia (scripts/live_wiki_relay.py).

  build_live_web_pilot_pool.py --data <oellm-rl data> --n 8 --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

TOOLS = ["wiki_search", "open_page", "find_in_page"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--languages", nargs="+", default=["en", "sv", "de", "fr"])
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    d = a.data
    used = {json.loads(line)["id"].split("#")[0] for line in open(d / "deep-search/trajectories.jsonl", encoding="utf-8")}
    used |= {i.split("#")[0] for i in pq.read_table(d / "e10-web/train-mixed.parquet", columns=["id"]).column("id").to_pylist()}
    by_lang = collections.defaultdict(list)
    for r in sorted(pq.read_table(d / "deep-search/rl_deep_search.parquet").to_pylist(), key=lambda r: r["id"]):
        if r["id"] not in used and r["language"] in a.languages:
            by_lang[r["language"]].append(r)
    rng = random.Random(a.seed)
    picked = []
    for i in range(a.n):
        lang = a.languages[i % len(a.languages)]
        r = by_lang[lang].pop(rng.randrange(len(by_lang[lang])))
        picked.append({"id": f"{r['id']}#live_deep_search", "dataset": "eu_qa", "tools": TOOLS,
                       "ground_truth": r["ground_truth"],
                       "messages": [{"role": m["role"], "content": m["content"], "rl_mode": "default"}
                                    for m in r["messages"]]})
    schema = pa.schema([("id", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()),
                                                         ("rl_mode", pa.string())]))),
                        ("ground_truth", pa.string()), ("dataset", pa.string()), ("tools", pa.list_(pa.string()))])
    a.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(picked, schema=schema), a.out / "train-mixed.parquet")
    manifest = {"kind": "live_web_pilot_pool", "rows": len(picked), "seed": a.seed, "excluded_ids": len(used),
                "languages": dict(collections.Counter(r["id"].rsplit("-", 1)[-1].split("#")[0] for r in picked)),
                "ids": [r["id"] for r in picked]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    for r in picked:
        print(r["messages"][-1]["content"][:200].replace("\n", " "), "->", r["ground_truth"][:120])


if __name__ == "__main__":
    main()
