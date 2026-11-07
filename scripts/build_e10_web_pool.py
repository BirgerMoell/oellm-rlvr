#!/usr/bin/env python3
"""E10-web pool (configs/leonardo-grpo-math-oellm9b-e10-web.yaml): browse for two-hop facts, abstain when the browsing
finds nothing. 2,560 prompts = 160 updates x 16, exact counts:

  deep_search   896  two-hop questions (build_deep_search.py, rl_deep_search.parquet), default mode, the three tools
                     (wiki_search, open_page, find_in_page); an equal share per language, none whose id is in the
                     warm-start trajectories (the parent has seen those answers);
  fab_deep      128  the same questions with the subject replaced by a made-up name (oellm_rlvr.hallucination
                     syllables, not a title in any snapshot language), default mode, the three tools; the right
                     answer is UNKNOWN;
  e8           1536  the E8 pool (build_e8_pool.py) subsampled per slice in proportion: single-hop search, closed-book
                     and made-up-entity QA in both modes (the calibration E8 learned), and its replay (doc, math, IF,
                     code, tools).

Reward for every QA row: +1 correct, 0 ANSWER: UNKNOWN, -1 wrong or no answer (oellm_rlvr.eu_tasks). The v1 warm start
showed what is lost without the calibration rows: trained on answered chains only, it stopped abstaining everywhere.

  build_e10_web_pool.py --data <oellm-rl data> --search <search root> --model <dir> --out <dir>   (search venv)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20261107
COUNTS = {"deep_search": 896, "fab_deep": 128, "e8": 1536}
TOOLS = ["wiki_search", "open_page", "find_in_page"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="oellm-rl data dir (deep-search, deep-search-raw, e8)")
    ap.add_argument("--search", type=Path, required=True, help="search root (leads/<lang>.parquet titles)")
    ap.add_argument("--model", type=Path, required=True, help="tokenizer + RL template, for the prompt-length check")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompt", type=int, default=3072)
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from build_eu_task_suites import label_in
    from oellm_rlvr import hallucination as H
    from oellm_rlvr import wiki_search as W

    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(a.model)
    d = a.data

    def fits(r):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"]) <= a.max_prompt

    deep = pq.read_table(d / "deep-search/rl_deep_search.parquet").to_pylist()
    seen = {json.loads(line)["id"].split("#")[0] for line in open(d / "deep-search/trajectories.jsonl", encoding="utf-8")}
    by_lang = collections.defaultdict(list)
    for r in sorted(deep, key=lambda r: r["id"]):
        if r["id"] not in seen:
            by_lang[r["language"]].append(r)
    for rows in by_lang.values():
        rng.shuffle(rows)
    langs = sorted(by_lang)

    def round_robin(n, what):
        got, cursor = [], {lang: 0 for lang in langs}
        while len(got) < n:
            progressed = False
            for lang in langs:
                rows = by_lang[lang]
                while cursor[lang] < len(rows):
                    r = rows[cursor[lang]]
                    cursor[lang] += 1
                    if r.get("_used") or not fits(r):
                        continue
                    r["_used"] = True
                    got.append(r)
                    progressed = True
                    break
                if len(got) == n:
                    break
            if not progressed:
                raise SystemExit(f"{what}: want {n}, have {len(got)}")
        return got

    def clean(r, tag):
        return {"id": f"{r['id']}#{tag}", "dataset": "eu_qa", "tools": TOOLS, "ground_truth": r["ground_truth"],
                "messages": [{"role": m["role"], "content": m["content"], "rl_mode": "default"} for m in r["messages"]]}

    picked = [clean(r, "deep_search") for r in round_robin(COUNTS["deep_search"], "deep_search")]

    # made-up subjects: same question, subject swapped for a name that is no title in any snapshot language
    entities = json.loads((d / "deep-search-raw/entities.json").read_text())
    titles = set()
    for lang in W.LANGUAGES:
        names = pq.read_table(a.search / "leads" / f"{lang}.parquet", columns=["title"]).column("title").to_pylist()
        titles |= {W.norm_title(t) for t in names}
    fab, skipped = [], collections.Counter()
    for r in round_robin(4 * COUNTS["fab_deep"], "fab_deep candidates"):
        if len(fab) == COUNTS["fab_deep"]:
            break
        subj = label_in(entities.get(r["id"].split("-")[-2]), r["language"])
        prompt = r["messages"][0]["content"]
        if not subj or prompt.count(subj) != 1:
            skipped["subject_not_in_prompt"] += 1
            continue
        fake = " ".join(H.made_up_word(rng, n) for n in ((3, 2) if " " in subj else (3,)))
        if W.norm_title(fake) in titles:
            skipped["real_title"] += 1
            continue
        label = {"suite": "qa", "kind": "unanswerable", "answers": [], "mode": "default"}
        fab.append({"id": f"{r['id']}#fab_deep", "dataset": "eu_qa", "tools": TOOLS,
                    "ground_truth": json.dumps(label),
                    "messages": [{"role": "user", "content": prompt.replace(subj, fake), "rl_mode": "default"}]})
    if len(fab) < COUNTS["fab_deep"]:
        raise SystemExit(f"fab_deep: want {COUNTS['fab_deep']}, have {len(fab)} ({dict(skipped)})")
    picked += fab

    # E8 pool, per slice in proportion (largest remainder), so its calibration slices keep their shares
    e8 = pq.read_table(d / "e8/train-mixed.parquet").to_pylist()
    slices = collections.defaultdict(list)
    for r in sorted(e8, key=lambda r: r["id"]):
        slices[r["id"].split("#")[1] if "#" in r["id"] else r["dataset"]].append(r)
    share = {k: COUNTS["e8"] * len(v) / len(e8) for k, v in slices.items()}
    quota = {k: int(s) for k, s in share.items()}
    for k in sorted(share, key=lambda k: share[k] - quota[k], reverse=True)[:COUNTS["e8"] - sum(quota.values())]:
        quota[k] += 1
    for k, rows in sorted(slices.items()):
        rng.shuffle(rows)
        picked += [{"id": r["id"], "dataset": r["dataset"], "tools": list(r["tools"] or []),
                    "ground_truth": r["ground_truth"],
                    "messages": [{k2: m.get(k2) for k2 in ("role", "content", "rl_mode")} for m in r["messages"]]}
                   for r in rows[:quota[k]]]

    ids = [r["id"] for r in picked]
    assert len(ids) == len(set(ids)) == sum(COUNTS.values()), (len(ids), len(set(ids)))
    rng.shuffle(picked)
    schema = pa.schema([("id", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()),
                                                         ("rl_mode", pa.string())]))),
                        ("ground_truth", pa.string()), ("dataset", pa.string()), ("tools", pa.list_(pa.string()))])
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "train-mixed.parquet"
    pq.write_table(pa.Table.from_pylist(picked, schema=schema), out)

    def slice_of(r):
        return r["id"].split("#")[1] if "#" in r["id"] else r["dataset"]

    manifest = {"kind": "e10_web_pool", "rows": len(picked), "seed": SEED,
                "by_slice": dict(collections.Counter(slice_of(r) for r in picked)),
                "e8_quota": quota, "fab_deep_skipped": dict(skipped),
                "deep_search_languages": dict(collections.Counter(r["id"].rsplit("-", 1)[-1].split("#")[0]
                                                                  for r in picked if slice_of(r) == "deep_search")),
                "with_tools": sum(bool(r["tools"]) for r in picked), "excluded_warmstart_ids": len(seen),
                "max_prompt_tokens": a.max_prompt, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
