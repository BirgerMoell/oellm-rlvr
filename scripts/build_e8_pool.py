#!/usr/bin/env python3
"""E8 pool (qwen35-posttrain docs/OELLM_HALLUCINATION_PLAN.md): answer what you know, search what you don't, abstain
on what doesn't exist. 2,560 prompts = 160 updates x 16, exact counts:

  qa_search      640  default mode, wiki_search available (row `tools` = ["wiki_search"]); W2 knowledge strata
                      160 known / 160 mid / 320 unknown, so most rows need the search;
  fab_search      96  made-up entities, default mode, search available (the right answer is UNKNOWN);
  qa_closed      448  no tools, half default / half thinking: 288 known (abstaining scores 0 < answering's +1: the
                      guard against W3's collapse), 64 mid, 96 unknown (abstaining beats a wrong guess);
  fab_closed      32  made-up entities, no tools, both modes;
  replay        1344  eu_doc 320, math 384, IF 192 (144 direct + 48 think), code 192, tools 256 (A4b; their tool
                      schemas are in the prompt, no executed tool).

Reward for every QA row: +1 correct, 0 ANSWER: UNKNOWN, -1 wrong or no answer (oellm_rlvr.eu_tasks). "Known" uses
W2's 8 default-mode samples per question (hallu-pilot-w2-v2): known = >= 5/8 correct, unknown = 0/8 correct, mid =
the rest. Thinking-mode rows never get the tool: the backend's observation format ends in the default-mode assistant
header (compat.patch_oellm_hermes_parser).

  build_e8_pool.py fabricated --root <search root> --out <dir>      (search venv: drops real Wikipedia titles)
  build_e8_pool.py pool --data <oellm-rl data> --w2 <hallu-pilot-w2-v2> --model <dir> --out <dir>   (container)
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

SEED = 20261105
EVAL_SEED, EVAL_PER_LANGUAGE = 20261102, 100   # hallu-v1 fabricated rows (and more) to exclude
COUNTS = {"qa_search": {"known": 160, "mid": 160, "unknown": 320}, "fab_search": 96,
          "qa_closed": {"known": 288, "mid": 64, "unknown": 96}, "fab_closed": 32}
REPLAY = {"eu_doc": 320, "math": 384, "ifeval_direct": 144, "ifeval_think": 48, "code": 192, "tools": 256}


def fabricated(a) -> None:
    import pyarrow.parquet as pq

    from oellm_rlvr import hallucination as H
    from oellm_rlvr import wiki_search as W

    eval_subjects = {r["subject"] for r in H.fabricated_qa(EVAL_PER_LANGUAGE, seed=EVAL_SEED)}
    rows = H.fabricated_qa(a.per_language, seed=SEED)
    titles: dict[str, set[str]] = {}
    for lang in W.LANGUAGES:
        names = pq.read_table(a.root / "leads" / f"{lang}.parquet", columns=["title"]).column("title").to_pylist()
        titles[lang] = {W.norm_title(t) for t in names}
    kept, dropped = [], collections.Counter()
    for r in rows:
        key = W.norm_title(r["subject"])
        if r["subject"] in eval_subjects:
            dropped["in_eval"] += 1
        elif any(key in t for t in titles.values()):
            dropped["real_title"] += 1
        else:
            kept.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "fabricated.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    print(json.dumps({"generated": len(rows), "kept": len(kept), "dropped": dict(dropped)}))


def pool(a) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from oellm_rlvr.eu_tasks import grade_qa

    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(a.model)
    d = a.data

    def rows(path):
        return pq.read_table(path).to_pylist()

    def fits(r):
        msgs = [{k: v for k, v in m.items() if v is not None and k != "rl_mode"} for m in r["messages"]]
        tools = [TOOL] if r.get("tools") else None
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"]) <= a.max_prompt

    def take(candidates, n, what, partial=False):
        candidates = sorted(candidates, key=lambda r: r["id"])
        rng.shuffle(candidates)
        got = [r for r in candidates if fits(r)][:n]
        if len(got) < n and not partial:
            raise SystemExit(f"{what}: want {n}, have {len(got)}")
        return got

    from oellm_rlvr.wiki_search import TOOL_DESCRIPTION, TOOL_PARAMETERS
    TOOL = {"type": "function", "function": {"name": "wiki_search", "description": TOOL_DESCRIPTION,
                                             "parameters": TOOL_PARAMETERS}}

    def qa_row(r, mode, tools, tag):
        label = json.loads(r["ground_truth"])
        label["mode"] = mode
        msgs = [dict(m) for m in r["messages"]]
        msgs[-1]["rl_mode"] = "default" if mode == "default" else None
        return {"id": f"{r['id']}#{tag}", "messages": msgs, "ground_truth": json.dumps(label, ensure_ascii=False),
                "dataset": "eu_qa", "tools": tools}

    def plain(r):
        return {"id": r["id"], "messages": [{k: m.get(k) for k in ("role", "content", "rl_mode")} for m in r["messages"]],
                "ground_truth": r["ground_truth"], "dataset": r["dataset"], "tools": []}

    # W2 knowledge strata from its 8 default-mode samples per question
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(a.w2 / "prompts.jsonl", encoding="utf-8")}
    stratum = {}
    for line in open(a.w2 / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        p = prompts.get(s["id"])
        if not p or p["label"].get("kind") == "unanswerable":
            continue
        correct = sum(grade_qa(x, {**p["label"], "mode": "default"}) == "correct" for x in s["replies"])
        stratum[s["id"]] = "known" if correct >= 5 else "unknown" if correct == 0 else "mid"
    w3 = {r["id"] for r in rows(d / "eu-tasks/w3-eu.parquet")}
    qa = [r for r in rows(d / "eu-tasks/rl_qa.parquet") if r["id"] not in w3 and r["id"] in stratum]
    print("qa candidates", len(qa), collections.Counter(stratum[r["id"]] for r in qa), flush=True)

    if a.exclude:  # release checklist 7.1: no eval question in training
        held = {json.loads(l)["id"] for l in open(a.exclude, encoding="utf-8")}
        qa = [r for r in qa if r["id"] not in held]
    picked, used = [], set()
    for slice_name, tools in (("qa_search", ["wiki_search"]), ("qa_closed", [])):
        for band, n in COUNTS[slice_name].items():
            cands = [r for r in qa if stratum[r["id"]] == band and r["id"] not in used]
            # search rows are default mode; closed-book rows alternate default / thinking
            got = take([qa_row(r, "default" if tools or i % 2 == 0 else "thinking", tools, slice_name)
                        for i, r in enumerate(cands)], n, f"{slice_name}/{band}")
            used |= {g["id"].split("#")[0] for g in got}
            picked += got
    fab = [json.loads(l) for l in open(a.fabricated / "fabricated.jsonl", encoding="utf-8")]
    rng.shuffle(fab)

    def fab_row(r, mode, tools, tag):
        label = {**r["label"], "mode": mode}
        return {"id": f"{r['id']}#{tag}", "dataset": "eu_qa", "tools": tools, "ground_truth": json.dumps(label),
                "messages": [{"role": "user", "content": r["prompt"], "rl_mode": "default" if mode == "default" else None}]}

    n_s, n_c = COUNTS["fab_search"], COUNTS["fab_closed"]
    if len(fab) < n_s + n_c:
        raise SystemExit(f"fabricated: want {n_s + n_c}, have {len(fab)}")
    picked += [fab_row(r, "default", ["wiki_search"], "fab_search") for r in fab[:n_s]]
    picked += [fab_row(r, "default" if i % 2 == 0 else "thinking", [], "fab_closed")
               for i, r in enumerate(fab[n_s:n_s + n_c])]

    # replay: rows unused by W1a, W2 and W3 first, topped up with rows only W1a saw (two windows back)
    seen = set()
    for w in ("w1a", "w2", "w3"):
        seen |= {r["id"] for r in rows(d / f"{w}/train-mixed.parquet")}
    recent = {r["id"] for w in ("w2", "w3") for r in rows(d / f"{w}/train-mixed.parquet")}
    replay_stats = {}

    def replay(pool_rows, n, what):
        fresh = take([r for r in pool_rows if r["id"] not in seen], n, what, partial=True)
        old = take([r for r in pool_rows if r["id"] in seen and r["id"] not in recent], n - len(fresh), f"{what} (W1a top-up)")
        replay_stats[what] = {"unused": len(fresh), "w1a_top_up": len(old)}
        return [plain(r) for r in fresh + old]

    picked += replay(rows(d / "eu-tasks/rl_doc.parquet"), REPLAY["eu_doc"], "eu_doc")
    picked += replay(rows(d / "answer-language/train-mixed.parquet"), REPLAY["math"], "math")
    iff = rows(d / "w2-if/train.parquet")
    for name in ("ifeval_direct", "ifeval_think"):
        picked += replay([r for r in iff if r["dataset"] == name], REPLAY[name], name)
    picked += replay(rows(d / "a5-code/train.parquet"), REPLAY["code"], "code")
    picked += replay(rows(d / "a4b-tool/train.parquet") + rows(d / "a4b-tool-translated/train.parquet"),
                     REPLAY["tools"], "tools")

    ids = [r["id"] for r in picked]
    assert len(ids) == len(set(ids)) == 2560, (len(ids), len(set(ids)))
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

    def mode_of(r):
        return "default" if r["messages"][-1]["rl_mode"] == "default" else "thinking"
    manifest = {"kind": "e8_pool", "rows": len(picked), "seed": SEED,
                "by_slice": dict(collections.Counter(slice_of(r) for r in picked)),
                "by_slice_mode": dict(collections.Counter(f"{slice_of(r)}/{mode_of(r)}" for r in picked)),
                "with_tools": sum(bool(r["tools"]) for r in picked), "replay": replay_stats, "w2_samples": str(a.w2),
                "max_prompt_tokens": a.max_prompt, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fabricated")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--per-language", type=int, default=30)
    p = sub.add_parser("pool")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--w2", type=Path, required=True, help="dir with W2's prompts.jsonl + samples.jsonl (k=8)")
    p.add_argument("--fabricated", type=Path, required=True, help="dir with fabricated.jsonl")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-prompt", type=int, default=3072)
    p.add_argument("--exclude", type=Path, help="eval jsonl whose ids must not appear in training (hallu-v1)")
    a = ap.parse_args()
    {"fabricated": fabricated, "pool": pool}[a.cmd](a)


if __name__ == "__main__":
    main()
