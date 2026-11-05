#!/usr/bin/env python3
"""Build the deep_search suite: two-hop questions (subject -> bridge -> answer) that the policy answers with the offline
Wikipedia tools (wiki_search, open_page, find_in_page). Facts come from scripts/fetch_deep_search_wikidata.py; rows are
graded by the eu_qa verifier (oellm_rlvr.eu_tasks.grade_qa: correct +1, UNKNOWN 0, wrong or no answer -1).

A (chain, subject) is kept only if it has exactly one bridge and one answer, and, per language, only if the snapshot
can answer it the way the question intends:
  - the subject and the bridge both have an article in that Wikipedia edition (Wikidata sitelinks);
  - the subject's article names the bridge (any label or alias), so hop 1 is findable by reading;
  - the bridge's article contains the answer (the year, or a label/alias of the place);
  - the subject's article does NOT contain the answer (otherwise one lookup suffices: dropped as "one_hop").
Each row records whether the answer is already in the bridge's search snippet ("snippet") or needs open_page /
find_in_page ("body"). Subjects are split by hash, 15 % eval, 85 % train, as in build_eu_task_suites.py.

Runs where the page store is (scripts/leonardo_wiki_pages.sbatch), in the search venv:
  python scripts/build_deep_search.py --raw data/deep-search-raw --search $S --out data/deep-search
Outputs: eval.jsonl, train_rows.jsonl (for deep_search_trajectories.py), rl_deep_search.parquet (TMAX rows, default mode, tools column), manifest.json.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eu_task_suites import all_names, h, label_in  # noqa: E402
from oellm_rlvr import eu_tasks as E  # noqa: E402
from oellm_rlvr import wiki_search as W  # noqa: E402

TOOLS = ["wiki_search", "open_page", "find_in_page"]
YEAR_CHAINS = {"film_director_birth_year", "book_author_birth_year", "org_founder_birth_year",
               "building_architect_birth_year"}
TEMPLATES = {
    "film_director_birth_year": {
        "en": "In which year was the director of the film “{s}” ({y}, {c}) born?",
        "de": "In welchem Jahr wurde die Person geboren, die beim Film „{s}“ ({y}, {c}) Regie führte?",
        "fr": "En quelle année est né(e) le réalisateur ou la réalisatrice du film « {s} » ({y}, {c}) ?",
        "es": "¿En qué año nació la persona que dirigió la película «{s}» ({y}, {c})?",
        "it": "In che anno è nato/a il/la regista del film «{s}» ({y}, {c})?",
        "nl": "In welk jaar is de regisseur van de film ‘{s}’ ({y}, {c}) geboren?",
        "sv": "Vilket år föddes regissören av filmen ”{s}” ({y}, {c})?",
        "pl": "W którym roku urodził(a) się reżyser(ka) filmu „{s}” ({y}, {c})?",
        "pt": "Em que ano nasceu o realizador ou a realizadora do filme «{s}» ({y}, {c})?",
        "fi": "Minä vuonna elokuvan ”{s}” ({y}, {c}) ohjaaja syntyi?"},
    "film_director_birth_place": {
        "en": "In which city or town was the director of the film “{s}” ({y}, {c}) born?",
        "de": "In welcher Stadt oder welchem Ort wurde die Person geboren, die beim Film „{s}“ ({y}, {c}) Regie führte?",
        "fr": "Dans quelle ville ou localité est né(e) le réalisateur ou la réalisatrice du film « {s} » ({y}, {c}) ?",
        "es": "¿En qué ciudad o localidad nació la persona que dirigió la película «{s}» ({y}, {c})?",
        "it": "In quale città o località è nato/a il/la regista del film «{s}» ({y}, {c})?",
        "nl": "In welke stad of plaats is de regisseur van de film ‘{s}’ ({y}, {c}) geboren?",
        "sv": "I vilken stad eller ort föddes regissören av filmen ”{s}” ({y}, {c})?",
        "pl": "W jakim mieście lub miejscowości urodził(a) się reżyser(ka) filmu „{s}” ({y}, {c})?",
        "pt": "Em que cidade ou localidade nasceu o realizador ou a realizadora do filme «{s}» ({y}, {c})?",
        "fi": "Missä kaupungissa tai paikkakunnalla elokuvan ”{s}” ({y}, {c}) ohjaaja syntyi?"},
    "book_author_birth_year": {
        "en": "In which year was the author of the literary work “{s}” (author's country: {c}) born?",
        "de": "In welchem Jahr wurde die Person geboren, die das literarische Werk „{s}“ (Land der Autorin/des Autors: {c}) geschrieben hat?",
        "fr": "En quelle année est né(e) l'auteur ou l'autrice de l'œuvre littéraire « {s} » (pays de l'auteur : {c}) ?",
        "es": "¿En qué año nació la persona que escribió la obra literaria «{s}» (país del autor: {c})?",
        "it": "In che anno è nato/a l'autore o l'autrice dell'opera letteraria «{s}» (paese dell'autore: {c})?",
        "nl": "In welk jaar is de auteur van het literaire werk ‘{s}’ (land van de auteur: {c}) geboren?",
        "sv": "Vilket år föddes författaren till det litterära verket ”{s}” (författarens land: {c})?",
        "pl": "W którym roku urodził(a) się autor(ka) utworu literackiego „{s}” (kraj autora: {c})?",
        "pt": "Em que ano nasceu o autor ou a autora da obra literária «{s}» (país do autor: {c})?",
        "fi": "Minä vuonna kirjallisen teoksen ”{s}” (kirjailijan maa: {c}) kirjoittaja syntyi?"},
    "book_author_birth_place": {
        "en": "In which city or town was the author of the literary work “{s}” (author's country: {c}) born?",
        "de": "In welcher Stadt oder welchem Ort wurde die Person geboren, die das literarische Werk „{s}“ (Land der Autorin/des Autors: {c}) geschrieben hat?",
        "fr": "Dans quelle ville ou localité est né(e) l'auteur ou l'autrice de l'œuvre littéraire « {s} » (pays de l'auteur : {c}) ?",
        "es": "¿En qué ciudad o localidad nació la persona que escribió la obra literaria «{s}» (país del autor: {c})?",
        "it": "In quale città o località è nato/a l'autore o l'autrice dell'opera letteraria «{s}» (paese dell'autore: {c})?",
        "nl": "In welke stad of plaats is de auteur van het literaire werk ‘{s}’ (land van de auteur: {c}) geboren?",
        "sv": "I vilken stad eller ort föddes författaren till det litterära verket ”{s}” (författarens land: {c})?",
        "pl": "W jakim mieście lub miejscowości urodził(a) się autor(ka) utworu literackiego „{s}” (kraj autora: {c})?",
        "pt": "Em que cidade ou localidade nasceu o autor ou a autora da obra literária «{s}» (país do autor: {c})?",
        "fi": "Missä kaupungissa tai paikkakunnalla kirjallisen teoksen ”{s}” (kirjailijan maa: {c}) kirjoittaja syntyi?"},
    "org_founder_birth_year": {
        "en": "In which year was the founder of {s} ({c}) born?",
        "de": "In welchem Jahr wurde die Gründerin bzw. der Gründer von {s} ({c}) geboren?",
        "fr": "En quelle année est né(e) le fondateur ou la fondatrice de {s} ({c}) ?",
        "es": "¿En qué año nació la persona que fundó {s} ({c})?",
        "it": "In che anno è nato/a il fondatore o la fondatrice di {s} ({c})?",
        "nl": "In welk jaar is de oprichter van {s} ({c}) geboren?",
        "sv": "Vilket år föddes grundaren av {s} ({c})?",
        "pl": "W którym roku urodziła się osoba, która założyła {s} ({c})?",
        "pt": "Em que ano nasceu o fundador ou a fundadora de {s} ({c})?",
        "fi": "Minä vuonna kohteen {s} ({c}) perustaja syntyi?"},
    "building_architect_birth_year": {
        "en": "In which year was the architect of {s} ({c}) born?",
        "de": "In welchem Jahr wurde die Architektin bzw. der Architekt von {s} ({c}) geboren?",
        "fr": "En quelle année est né(e) l'architecte de {s} ({c}) ?",
        "es": "¿En qué año nació el arquitecto o la arquitecta de {s} ({c})?",
        "it": "In che anno è nato/a l'architetto di {s} ({c})?",
        "nl": "In welk jaar is de architect van {s} ({c}) geboren?",
        "sv": "Vilket år föddes arkitekten bakom {s} ({c})?",
        "pl": "W którym roku urodziła się osoba, która zaprojektowała {s} ({c})?",
        "pt": "Em que ano nasceu o arquiteto ou a arquiteta de {s} ({c})?",
        "fi": "Minä vuonna kohteen {s} ({c}) arkkitehti syntyi?"},
}


def contains(text_norm: str, names: list[str]) -> bool:
    return any(n and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text_norm) for n in names)


def chains(raw: Path) -> tuple[list[dict], dict, collections.Counter]:
    entities = json.loads((raw / "entities.json").read_text())
    groups: dict[tuple[str, str], dict] = {}
    for line in (raw / "facts.jsonl").open():
        r = json.loads(line)
        if not r.get("s"):
            continue
        g = groups.setdefault((r["type"], r["s"]), {"type": r["type"], "s": r["s"], "sl": r["sl"], "band": r["band"],
                                                    "country": r["country"], "o": set(), "a": set(), "t": set()})
        g["o"].add(r["o"])
        g["a"].add(r["a"][:4] if r["type"] in YEAR_CHAINS else r["a"])
        if r.get("t") and re.match(r"^\d{4}-", r["t"]):
            g["t"].add(int(r["t"][:4]))
    skipped = collections.Counter()
    items = []
    for g in groups.values():
        if len(g["o"]) != 1 or len(g["a"]) != 1:
            skipped["bridge_or_answer_not_unique"] += 1
            continue
        g["o"], g["a"] = next(iter(g["o"])), next(iter(g["a"]))
        if g["type"].startswith("film_") and not g["t"]:
            skipped["film_without_year"] += 1
            continue
        g["film_year"] = min(g["t"]) if g["t"] else None
        if g["type"] in YEAR_CHAINS:
            if not 1000 <= int(g["a"]) <= 2015:
                skipped["year_out_of_range"] += 1
                continue
            g["answers"], g["kind"] = [g["a"]], "year"
        else:
            g["answers"], g["kind"] = all_names(entities.get(g["a"], {})), "entity"
            if not g["answers"]:
                skipped["answer_unlabelled"] += 1
                continue
        del g["t"]
        items.append(g)
    return items, entities, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--search", type=Path, required=True, help="search root with leads/, index/ and pages/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    items, entities, skipped = chains(a.raw)
    print(f"{len(items)} unique chains; skipped {dict(skipped)}", flush=True)
    searcher = W.Searcher(a.search)
    texts: dict[tuple[str, str], str | None] = {}

    def page(qid: str, lang: str) -> str | None:
        """Normalised article text of an entity in one edition, via its sitelink title."""
        key = (qid, lang)
        if key not in texts:
            title = entities.get(qid, {}).get("sitelinks", {}).get(lang)
            doc = None
            if title:
                _, titles, _, exact, _ = searcher._load(lang)
                keys = W.title_keys(title)
                doc = next((d for d in exact.get(keys[0], []) if titles[d] == title), None) if keys else None
            texts[key] = E.norm_text(searcher.page_text(lang, doc)) if doc is not None else None
            if len(texts) > 100000:
                texts.clear()
        return texts[key]

    rows, why = [], collections.Counter()
    names = {it["s"]: ([E.norm_text(n) for n in all_names(entities.get(it["o"], {}))],
                       [E.norm_text(x) for x in it["answers"]]) for it in items}
    # One language at a time: the serial partition allows ~30 GB, and one edition's titles and maps take several GB.
    for lang in E.LANGS:
        for key in [k for k in texts if k[1] != lang]:
            del texts[key]
        searcher._indexes.clear()
        searcher._pages.clear()
        for it in items:
            bridge_names, answers = names[it["s"]]
            subj, country = label_in(entities.get(it["s"]), lang), label_in(entities.get(it["country"]), lang)
            if not subj or not country:
                why["no_label"] += 1
                continue
            s_text, o_text = page(it["s"], lang), page(it["o"], lang)
            if s_text is None or o_text is None:
                why["no_article"] += 1
                continue
            if not contains(s_text, bridge_names):
                why["bridge_not_in_subject_page"] += 1
                continue
            if not contains(o_text, answers):
                why["answer_not_in_bridge_page"] += 1
                continue
            if contains(s_text, answers):
                why["one_hop"] += 1
                continue
            slots = {"s": subj, "c": country, "y": str(it["film_year"] or "")}
            o_head = o_text[:W.SNIPPET_CHARS]
            rows.append({"id": f"deep_search-{it['type']}-{it['s']}-{lang}", "suite": "deep_search", "subset": it["type"],
                         "language": lang, "band": it["band"], "sitelinks": it["sl"],
                         "evidence": "snippet" if contains(o_head, answers) else "body",
                         "prompt": TEMPLATES[it["type"]][lang].format(**slots) + "\n\n" + E.QA_INSTRUCTION[lang],
                         "label": {"suite": "qa", "kind": it["kind"], "answers": it["answers"]},
                         "source": f"wikidata:{it['s']}>{it['o']}>{it['a']}"})
            why["kept"] += 1
        print(f"  {lang}: {dict(why)}", flush=True)
    print(f"rows {len(rows)}: {dict(why)}", flush=True)

    eval_rows = [r for r in rows if h(r["source"].split(">")[0] + "#eval") < 0.15]
    train_rows = [r for r in rows if h(r["source"].split(">")[0] + "#eval") >= 0.15]
    rng.shuffle(train_rows)
    for name, part in (("eval.jsonl", eval_rows), ("train_rows.jsonl", train_rows)):
        with (a.out / name).open("w") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    import pyarrow as pa
    import pyarrow.parquet as pq
    records = [{"id": r["id"], "messages": [{"role": "user", "content": r["prompt"], "rl_mode": "default"}],
                "ground_truth": json.dumps({**r["label"], "mode": "default"}, ensure_ascii=False), "dataset": "eu_qa",
                "tools": TOOLS, "language": r["language"], "subset": r["subset"], "evidence": r["evidence"]}
               for r in train_rows]
    schema = pa.schema([("id", pa.string()),
                        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                        ("ground_truth", pa.string()), ("dataset", pa.string()), ("tools", pa.list_(pa.string())),
                        ("language", pa.string()), ("subset", pa.string()), ("evidence", pa.string())])
    path = a.out / "rl_deep_search.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)

    def counts(rs, key):
        return dict(collections.Counter(r[key] for r in rs))

    manifest = {"chains": len(items), "skipped_chains": dict(skipped), "row_filter": dict(why),
                "eval": {"rows": len(eval_rows), "language": counts(eval_rows, "language"),
                         "subset": counts(eval_rows, "subset"), "evidence": counts(eval_rows, "evidence")},
                "train": {"rows": len(train_rows), "language": counts(train_rows, "language"),
                          "subset": counts(train_rows, "subset"), "evidence": counts(train_rows, "evidence"),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
