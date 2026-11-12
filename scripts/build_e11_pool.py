#!/usr/bin/env python3
"""E11 pool (configs/leonardo-grpo-math-oellm9b-e11.yaml): keep E10-web's browsing, win back closed-book calibration.
Parent: E8 + 0.7 MOPD merge (a07) + 0.7 (E10-web - a07). 2,560 prompts = 160 updates x 16, exact counts:

  deep_search   768  two-hop browsing questions (build_deep_search.py), default mode, the three tools; none used by the
                     warm-start trajectories, the E10-web pool or the live-web pilot; an equal share per language;
  fab_deep      128  the same kind with the subject replaced by a made-up name (as in E10-web); right answer UNKNOWN;
  fab_closed    320  made-up entities, no tools (hallucination.fabricated_qa, new seed, hallu-v1 eval subjects and
                     real titles dropped), 3/4 default mode, 1/4 thinking. E10-web had 19 such rows of 2,560 and its
                     default-mode closed-book fabricated-wrong rate went from 7.0 % (a07) to 13.6 %;
  fab_people    192  made-up people with realistic names (a real first name + a stem with a surname ending of that
                     country, e.g. "Elin Velkenqvist"), birth year / place, no tools, 5/6 default mode: the base model
                     invents biographies for plausible unknown names (wsp/ROME probes, 2026-11-10);
  qa_closed     320  real closed-book questions from the E8 pool (all its bands), so abstaining stays selective;
  e8            832  the rest of the E8 pool per slice in proportion (single-hop search, made-up-entity search, doc,
                     math, IF, code, tools).

Reward for every QA row: +1 correct, 0 ANSWER: UNKNOWN, -1 wrong or no answer (oellm_rlvr.eu_tasks).

  build_e11_pool.py --data <oellm-rl data> --search <search root> --model <dir> --out <dir>   (training container)
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

SEED = 20261112
COUNTS = {"deep_search": 768, "fab_deep": 128, "fab_closed": 320, "fab_people": 192, "qa_closed": 320, "e8": 832}
TOOLS = ["wiki_search", "open_page", "find_in_page"]
EVAL_SEED, EVAL_PER_LANGUAGE = 20261102, 100   # hallu-v1 fabricated eval rows: never train on their subjects
SURNAME_ENDINGS = {"sv": ["qvist", "ström", "berg", "sson", "lund"], "de": ["mann", "berger", "hardt", "ner"],
                   "fr": ["ier", "ard", "eau", "ot"], "it": ["elli", "ini", "otti", "ucci"],
                   "pl": ["ski", "wicz", "czyk", "ak"], "fi": ["nen", "la", "inen", "lä"],
                   "es": ["ez", "ero", "illa", "ado"], "nl": ["stra", "ma", "huis", "ink"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="oellm-rl data dir (deep-search, e8, e10-web, ...)")
    ap.add_argument("--search", type=Path, required=True, help="search root (leads/<lang>.parquet titles)")
    ap.add_argument("--model", type=Path, required=True, help="tokenizer + RL template, for the prompt-length check")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompt", type=int, default=3072)
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from build_eu_task_suites import label_in
    from oellm_rlvr import eu_tasks as E
    from oellm_rlvr import hallucination as H
    from oellm_rlvr import wiki_search as W

    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(a.model)
    d = a.data

    def fits(r):
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"]) <= a.max_prompt

    titles = set()
    for lang in W.LANGUAGES:
        names = pq.read_table(a.search / "leads" / f"{lang}.parquet", columns=["title"]).column("title").to_pylist()
        titles |= {W.norm_title(t) for t in names}
    eval_subjects = {r["subject"] for r in H.fabricated_qa(EVAL_PER_LANGUAGE, seed=EVAL_SEED)}
    skipped = collections.Counter()

    # ---- deep search (+ made-up subjects), as in build_e10_web_pool.py, minus every id used before
    deep = pq.read_table(d / "deep-search/rl_deep_search.parquet").to_pylist()
    seen = {json.loads(line)["id"].split("#")[0] for line in open(d / "deep-search/trajectories.jsonl", encoding="utf-8")}
    for prior in ("e10-web", "live-web-pilot"):
        p = d / prior / "train-mixed.parquet"
        if p.exists():
            seen |= {i.split("#")[0] for i in pq.read_table(p, columns=["id"]).column("id").to_pylist()}
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

    entities = json.loads((d / "deep-search-raw/entities.json").read_text())
    fab = []
    for r in round_robin(4 * COUNTS["fab_deep"], "fab_deep candidates"):
        if len(fab) == COUNTS["fab_deep"]:
            break
        subj = label_in(entities.get(r["id"].split("-")[-2]), r["language"])
        prompt = r["messages"][0]["content"]
        if not subj or prompt.count(subj) != 1:
            skipped["fab_deep/subject_not_in_prompt"] += 1
            continue
        fake = " ".join(H.made_up_word(rng, n) for n in ((3, 2) if " " in subj else (3,)))
        if W.norm_title(fake) in titles:
            skipped["fab_deep/real_title"] += 1
            continue
        label = {"suite": "qa", "kind": "unanswerable", "answers": [], "mode": "default"}
        fab.append({"id": f"{r['id']}#fab_deep", "dataset": "eu_qa", "tools": TOOLS, "ground_truth": json.dumps(label),
                    "messages": [{"role": "user", "content": prompt.replace(subj, fake), "rl_mode": "default"}]})
    if len(fab) < COUNTS["fab_deep"]:
        raise SystemExit(f"fab_deep: want {COUNTS['fab_deep']}, have {len(fab)} ({dict(skipped)})")
    picked += fab

    # ---- closed-book made-up entities and people
    def closed_row(rid, prompt, mode, tag):
        label = {"suite": "qa", "kind": "unanswerable", "answers": [], "mode": mode}
        return {"id": f"{rid}#{tag}", "dataset": "eu_qa", "tools": [], "ground_truth": json.dumps(label),
                "messages": [{"role": "user", "content": prompt, "rl_mode": "default" if mode == "default" else None}]}

    def usable(subject):
        if subject in eval_subjects:
            skipped["closed/in_eval"] += 1
            return False
        if W.norm_title(subject) in titles:
            skipped["closed/real_title"] += 1
            return False
        return True

    per_lang = -(-COUNTS["fab_closed"] * 2 // len(E.LANGS))   # generate twice what is needed, then filter
    cands = [r for r in H.fabricated_qa(per_lang, seed=SEED) if usable(r["subject"])]
    rng.shuffle(cands)
    if len(cands) < COUNTS["fab_closed"]:
        raise SystemExit(f"fab_closed: want {COUNTS['fab_closed']}, have {len(cands)}")
    picked += [closed_row(f"e11-{r['id']}", r["prompt"], "thinking" if i % 4 == 3 else "default", "fab_closed")
               for i, r in enumerate(cands[:COUNTS["fab_closed"]])]

    people, names = [], set()
    while len(people) < COUNTS["fab_people"]:
        lang = E.LANGS[len(people) % len(E.LANGS)]
        country = rng.choice(sorted(H.COUNTRIES))
        name_lang, country_labels = H.COUNTRIES[country]
        surname = H.made_up_word(rng, 2) + rng.choice(SURNAME_ENDINGS[name_lang])
        subject = f"{rng.choice(E.FIRST[name_lang])} {surname}"
        if subject in names or not usable(subject) or W.norm_title(surname) in titles:
            skipped["people/rejected"] += 1
            continue
        names.add(subject)
        qtype = rng.choice(["birth_year", "birth_place"])
        occupation = H.OCCUPATIONS[rng.choice(sorted(H.OCCUPATIONS))][lang]
        prompt = E.qa_prompt(qtype, lang, {"s": subject, "d": f"{occupation}, {country_labels[lang]}"})
        mode = "thinking" if len(people) % 6 == 5 else "default"
        people.append(closed_row(f"e11-people-{qtype}-{len(people)}-{lang}", prompt, mode, "fab_people"))
    picked += people

    # ---- E8 pool: qa_closed fixed, the rest per slice in proportion (largest remainder)
    e8 = pq.read_table(d / "e8/train-mixed.parquet").to_pylist()
    slices = collections.defaultdict(list)
    for r in sorted(e8, key=lambda r: r["id"]):
        slices[r["id"].split("#")[1] if "#" in r["id"] else r["dataset"]].append(r)
    rest = {k: v for k, v in slices.items() if k != "qa_closed"}
    total = sum(len(v) for v in rest.values())
    share = {k: COUNTS["e8"] * len(v) / total for k, v in rest.items()}
    quota = {k: int(s) for k, s in share.items()}
    for k in sorted(share, key=lambda k: share[k] - quota[k], reverse=True)[:COUNTS["e8"] - sum(quota.values())]:
        quota[k] += 1
    quota["qa_closed"] = COUNTS["qa_closed"]
    for k, rows in sorted(slices.items()):
        if quota[k] > len(rows):
            raise SystemExit(f"{k}: want {quota[k]}, have {len(rows)}")
        rng.shuffle(rows)
        picked += [{"id": r["id"], "dataset": r["dataset"], "tools": list(r["tools"] or []),
                    "ground_truth": r["ground_truth"],
                    "messages": [{k2: m.get(k2) for k2 in ("role", "content", "rl_mode")} for m in r["messages"]]}
                   for r in rows[:quota[k]]]

    ids = [r["id"] for r in picked]
    assert len(ids) == len(set(ids)) == sum(COUNTS.values()), (len(ids), len(set(ids)))
    too_long = [r["id"] for r in picked if not fits(r)]
    assert not too_long, too_long[:5]
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
        return json.loads(r["ground_truth"]).get("mode") if r["dataset"] == "eu_qa" else None

    closed_abstain = sum(slice_of(r) in ("fab_closed", "fab_people") for r in picked)
    manifest = {"kind": "e11_pool", "rows": len(picked), "seed": SEED,
                "by_slice": dict(collections.Counter(slice_of(r) for r in picked)),
                "by_slice_mode": {f"{s}/{m}": n for (s, m), n in sorted(collections.Counter(
                    (slice_of(r), mode_of(r)) for r in picked if r["dataset"] == "eu_qa").items())},
                "e8_quota": quota, "skipped": dict(skipped),
                "closed_book_made_up": closed_abstain,   # incl. the E8 replay's fab_closed rows (same tag)
                "deep_search_languages": dict(collections.Counter(r["id"].rsplit("-", 1)[-1].split("#")[0]
                                                                  for r in picked if slice_of(r) == "deep_search")),
                "with_tools": sum(bool(r["tools"]) for r in picked), "excluded_ids": len(seen),
                "max_prompt_tokens": a.max_prompt, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "examples": {s: next(r["messages"][-1]["content"][:220] for r in picked if slice_of(r) == s)
                             for s in ("fab_closed", "fab_people", "fab_deep")}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
