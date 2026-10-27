#!/usr/bin/env python3
"""Build the eu_qa and eu_doc eval sets and RL pools (oellm_rlvr.eu_tasks).

eu_qa: facts from scripts/fetch_eu_qa_wikidata.py are grouped per (question type, subject); a fact is kept only if
it has exactly one answer (one birth year, one director, ...), the subject has a label in (or usable for) the target
language, and no answer alias occurs in the subject's own label (e.g. "Uppsala County" -> "Uppsala"). Subjects are
split by hash: 15 % eval candidates, 85 % train. The eval set is balanced over type x popularity band x language.

eu_doc: generated tasks; eval seeds and train seeds are disjoint ("eval-*" / "train-*"), and each family's
HELD_OUT variant appears only in the eval set.

Outputs in --out: eval.jsonl (both suites; scored in default and thinking mode by the eval script), rl_qa.parquet and
rl_doc.parquet (TMAX rows: messages with rl_mode, ground_truth = JSON label incl. mode, dataset eu_qa / eu_doc),
manifest.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oellm_rlvr import eu_tasks as E  # noqa: E402

ENTITY_TYPES = {"birth_place", "admin_capital", "film_director", "book_author", "river_mouth"}


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:12], 16) / 16 ** 12


def label_in(ent: dict, lang: str) -> str | None:
    labels = ent.get("labels", {}) if ent else {}
    for key in (lang, "mul", "en"):
        if labels.get(key) and not re.fullmatch(r"Q\d+", labels[key]):
            return labels[key]
    return None


def all_names(ent: dict) -> list[str]:
    names = set(ent.get("labels", {}).values())
    for xs in ent.get("aliases", {}).values():
        names.update(xs)
    return sorted(n for n in names if len(n) >= 2)


def build_qa(raw: Path, eval_per_lang: int, rng: random.Random):
    entities = json.loads((raw / "entities.json").read_text())
    groups: dict[tuple[str, str], dict] = {}
    for line in (raw / "facts.jsonl").open():
        r = json.loads(line)
        if not r.get("s"):
            continue
        g = groups.setdefault((r["type"], r["s"]), {"type": r["type"], "s": r["s"], "sl": r["sl"], "band": r["band"],
                                                    "country": r["country"], "o": set(), "years": set(), "x": []})
        if r.get("o"):
            g["o"].add(r["o"])
        if r.get("t") and re.match(r"^\d{4}-", r["t"]):
            g["years"].add(int(r["t"][:4]))
        if r.get("x") and r["x"] not in g["x"]:
            g["x"].append(r["x"])
    skipped = collections.Counter()
    items = []
    for (qtype, s), g in groups.items():
        if qtype in ("birth_year", "org_founded"):
            if len(g["years"]) != 1 or not 1000 <= min(g["years"]) <= 2015:
                skipped["year_not_unique"] += 1
                continue
            answers, kind = [str(min(g["years"]))], "year"
        else:
            if len(g["o"]) != 1:
                skipped["answer_not_unique"] += 1
                continue
            o = next(iter(g["o"]))
            answers, kind = all_names(entities.get(o, {})), "entity"
            if not answers:
                skipped["answer_unlabelled"] += 1
                continue
        subject_names = [E.norm_text(n) for n in all_names(entities.get(s, {}))]
        if kind == "entity" and any(re.search(rf"(?<!\w){re.escape(E.norm_text(a))}(?!\w)", n) for a in answers for n in subject_names if E.norm_text(a)):
            skipped["answer_in_subject_name"] += 1
            continue
        if qtype == "film_director":
            g["film_year"] = min(g["years"]) if g["years"] else None
            if g["film_year"] is None:
                skipped["film_without_year"] += 1
                continue
        items.append({**g, "answers": answers, "kind": kind})
    print(f"qa: {len(items)} usable facts; skipped {dict(skipped)}", flush=True)

    def render(it: dict, lang: str) -> dict | None:
        subj = label_in(entities.get(it["s"]), lang)
        country = label_in(entities.get(it["country"]), lang)
        if not subj or not country:
            return None
        slots = {"s": subj, "c": country}
        if it["type"] in ("birth_year", "birth_place", "org_founded"):
            x = next((label_in(entities.get(x), lang) for x in it["x"] if label_in(entities.get(x), lang)), None)
            if not x:
                return None
            slots["d"] = f"{x}, {country}"
        if it["type"] == "film_director":
            slots["y"] = str(it["film_year"])
        return {"id": f"eu_qa-{it['type']}-{it['s']}-{lang}", "suite": "qa", "subset": it["type"], "language": lang,
                "band": it["band"], "sitelinks": it["sl"], "prompt": E.qa_prompt(it["type"], lang, slots),
                "label": {"suite": "qa", "kind": it["kind"], "answers": it["answers"]}, "source": f"wikidata:{it['s']}"}

    eval_pool = [it for it in items if h(it["s"] + "#eval") < 0.15]
    train_pool = [it for it in items if h(it["s"] + "#eval") >= 0.15]
    cells = collections.defaultdict(list)
    for it in eval_pool:
        cells[(it["type"], it["band"])].append(it)
    for v in cells.values():
        rng.shuffle(v)
    # Eval: round-robin over (type, band) cells and languages until each language has eval_per_lang questions.
    eval_rows, used = [], set()
    per_lang = collections.Counter()
    keys = sorted(cells)
    progress = True
    while progress and min(per_lang[l] for l in E.LANGS) < eval_per_lang:
        progress = False
        for lang in E.LANGS:
            for key in keys:
                if per_lang[lang] >= eval_per_lang:
                    break
                while cells[key]:
                    it = cells[key].pop()
                    row = render(it, lang)
                    if row:
                        eval_rows.append(row); used.add(it["s"]); per_lang[lang] += 1; progress = True
                        break
    train_rows = []
    for it in train_pool:
        lang = E.LANGS[int(h(it["s"] + "#lang") * len(E.LANGS))]
        row = render(it, lang) or render(it, "en")
        if row:
            train_rows.append(row)
    rng.shuffle(train_rows)
    return eval_rows, train_rows


def build_doc(eval_per_family: int, train_n: int, rng: random.Random):
    eval_rows, train_rows = [], []
    for family in E.FAMILIES:
        vs = E.variants(family)
        for k in range(eval_per_family):
            lang, variant, level = E.LANGS[k % len(E.LANGS)], vs[(k // len(E.LANGS)) % len(vs)], 1 + k % 3
            t = E.make_doc_task(family, lang, level, variant, f"eval-{k}")
            eval_rows.append({"id": f"eu_doc-{family}-{variant}-{lang}-L{level}-e{k}", "suite": "doc", "subset": family,
                              "variant": variant, "held_out_variant": t["held_out_variant"], "language": lang, "level": level,
                              "prompt": t["prompt"], "label": {"suite": "doc", **t["label"]}})
    train_specs = [(f, v) for f in E.FAMILIES for v in E.variants(f) if E.HELD_OUT.get(f) != v]
    for k in range(train_n):
        family, variant = train_specs[k % len(train_specs)]
        lang, level = rng.choice(E.LANGS), rng.choice([1, 2, 2, 3])
        t = E.make_doc_task(family, lang, level, variant, f"train-{k}")
        train_rows.append({"id": f"eu_doc-{family}-{variant}-{lang}-L{level}-t{k}", "suite": "doc", "subset": family,
                           "variant": variant, "language": lang, "level": level, "prompt": t["prompt"],
                           "label": {"suite": "doc", **t["label"]}})
    rng.shuffle(train_rows)
    return eval_rows, train_rows


def write_pool(rows: list[dict], path: Path, dataset: str, think_share: float, rng: random.Random) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    records = []
    for r in rows:
        mode = "thinking" if rng.random() < think_share else "default"
        records.append({"id": r["id"], "messages": [{"role": "user", "content": r["prompt"], "rl_mode": "default" if mode == "default" else None}],
                        "ground_truth": json.dumps({**r["label"], "mode": mode}, ensure_ascii=False), "dataset": dataset,
                        "language": r["language"], "subset": r["subset"]})
    schema = pa.schema([("id", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                        ("ground_truth", pa.string()), ("dataset", pa.string()), ("language", pa.string()), ("subset", pa.string())])
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
    return {"rows": len(records), "thinking": sum(json.loads(r["ground_truth"])["mode"] == "thinking" for r in records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="output dir of fetch_eu_qa_wikidata.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qa-eval-per-lang", type=int, default=60)
    ap.add_argument("--doc-eval-per-family", type=int, default=100)
    ap.add_argument("--doc-train", type=int, default=6000)
    ap.add_argument("--think-share", type=float, default=0.4)
    a = ap.parse_args()
    rng = random.Random(20261027)
    a.out.mkdir(parents=True, exist_ok=True)
    qa_eval, qa_train = build_qa(a.raw, a.qa_eval_per_lang, rng)
    doc_eval, doc_train = build_doc(a.doc_eval_per_family, a.doc_train, rng)
    with (a.out / "eval.jsonl").open("w") as f:
        for r in qa_eval + doc_eval:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in (("train_qa", qa_train), ("train_doc", doc_train)):
        with (a.out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = lambda rows, key: dict(sorted(collections.Counter(r[key] for r in rows).items()))
    manifest = {
        "kind": "oellm_eu_task_suites_v1",
        "eval": {"qa": len(qa_eval), "doc": len(doc_eval), "qa_by_type": count(qa_eval, "subset"), "qa_by_band": count(qa_eval, "band"),
                 "qa_by_language": count(qa_eval, "language"), "doc_by_family": count(doc_eval, "subset"),
                 "doc_held_out_variant": sum(r["held_out_variant"] for r in doc_eval)},
        "train": {"qa": len(qa_train), "doc": len(doc_train), "qa_by_type": count(qa_train, "subset"), "qa_by_band": count(qa_train, "band")},
        "held_out_variants": E.HELD_OUT,
        "source": "Wikidata (CC0) via query.wikidata.org and wbgetentities, fetched 2026-10-27; documents generated by oellm_rlvr.eu_tasks",
    }
    try:
        manifest["rl_qa"] = write_pool(qa_train, a.out / "rl_qa.parquet", "eu_qa", a.think_share, rng)
        manifest["rl_doc"] = write_pool(doc_train, a.out / "rl_doc.parquet", "eu_doc", a.think_share, rng)
    except ImportError:
        print("pyarrow missing: RL parquet pools not written (JSONL train files are)", flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
