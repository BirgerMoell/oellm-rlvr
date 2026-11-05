#!/usr/bin/env python3
"""Reference browsing trajectories for deep_search (warm-start SFT data if RL has no signal).

A scripted agent answers the two-hop train questions through the same three tools the policy gets (wiki_search,
open_page, find_in_page; oellm_rlvr.wiki_search.Searcher, called in-process), one call per turn:
  1. search the subject; 2. find the bridge in the subject's article (find_in_page with the chain's role words, else
  read the parts in order); 3. search the bridge; 4. if the snippet lacks the answer, find it in the bridge's article.
It uses the facts only to decide when it has found something (as a reader would recognise it), never to form a query
beyond the names in the question and in what it read. A trajectory is kept only if the final reply is graded
"correct" by the RL reward (eu_tasks.grade_qa).

Chat format: OpenAI/Hermes messages that template v2.1 renders (user, assistant with tool_calls, tool, final assistant).
  python scripts/deep_search_trajectories.py --raw data/deep-search-raw --search $S --data data/deep-search \
    --out data/deep-search/trajectories.jsonl [--limit 3000]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deep_search import contains  # noqa: E402
from build_eu_task_suites import all_names, label_in  # noqa: E402
from oellm_rlvr import eu_tasks as E  # noqa: E402
from oellm_rlvr import wiki_search as W  # noqa: E402

ROLE = {
    "director": {"en": "directed by director", "de": "Regie Regisseur", "fr": "réalisé par réalisateur", "es": "dirigida por director",
                 "it": "regia regista diretto", "nl": "geregisseerd regie regisseur", "sv": "regi regisserad regissör",
                 "pl": "reżyseria reżyser", "pt": "realizado realizador dirigido", "fi": "ohjannut ohjaaja ohjaus"},
    "author": {"en": "written by author novel", "de": "Autor geschrieben Roman", "fr": "écrit par auteur roman", "es": "escrita por autor novela",
               "it": "scritto autore romanzo", "nl": "geschreven auteur roman", "sv": "skriven författare roman",
               "pl": "napisana autor powieść", "pt": "escrito autor romance", "fi": "kirjoittanut kirjailija romaani"},
    "founder": {"en": "founded by founder", "de": "gegründet Gründer", "fr": "fondé par fondateur", "es": "fundada por fundador",
                "it": "fondata fondatore", "nl": "opgericht oprichter", "sv": "grundades grundare", "pl": "założona założyciel",
                "pt": "fundada fundador", "fi": "perusti perustaja"},
    "architect": {"en": "architect designed by", "de": "Architekt entworfen", "fr": "architecte conçu par", "es": "arquitecto diseñado",
                  "it": "architetto progettato", "nl": "architect ontworpen", "sv": "arkitekt ritad", "pl": "architekt zaprojektowany",
                  "pt": "arquiteto projetado", "fi": "arkkitehti suunnitteli"},
}
BORN = {"en": "born", "de": "geboren", "fr": "né née", "es": "nació nacido", "it": "nato nata", "nl": "geboren", "sv": "född",
        "pl": "urodził urodziła", "pt": "nasceu nascido", "fi": "syntyi syntynyt"}


def role_of(chain: str) -> str:
    return {"film": "director", "book": "author", "org": "founder", "building": "architect"}[chain.split("_")[0]]


class Run:
    def __init__(self, searcher: W.Searcher, prompt: str) -> None:
        self.s = searcher
        self.messages = [{"role": "user", "content": prompt}]

    def call(self, name: str, args: dict) -> str:
        self.messages.append({"role": "assistant", "content": "",
                              "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
        if name == "wiki_search":
            out = W.format_results(self.s.search(args["query"], args["language"], 5))
        elif name == "open_page":
            out = W.format_page(self.s.open(args["title"], args["language"], args.get("part", 1)))
        else:
            out = W.format_find(self.s.find(args["title"], args["query"], args["language"]))
        self.messages.append({"role": "tool", "content": out})
        return out


def locate(run: Run, title: str, lang: str, query: str, names: list[str]) -> bool:
    """find_in_page, then read the article part by part (at most 4 parts) until one of the names shows up."""
    if contains(E.norm_text(run.call("find_in_page", {"title": title, "query": query, "language": lang})), names):
        return True
    page = run.s.open(title, lang, 1)
    for part in range(1, min(page.get("parts", 1), 4) + 1):
        if contains(E.norm_text(run.call("open_page", {"title": title, "language": lang, "part": part})), names):
            return True
    return False


def solve(searcher: W.Searcher, row: dict, entities: dict) -> tuple[list[dict] | None, str]:
    s_id, o_id, a_id = row["source"].removeprefix("wikidata:").split(">")
    lang, chain = row["language"], row["subset"]
    subj = label_in(entities.get(s_id), lang)
    bridge = label_in(entities.get(o_id), lang)
    s_title = entities[s_id]["sitelinks"][lang]
    o_title = entities[o_id]["sitelinks"][lang]
    bridge_names = [E.norm_text(n) for n in all_names(entities.get(o_id, {}))]
    answers = [E.norm_text(x) for x in row["label"]["answers"]]
    run = Run(searcher, row["prompt"])
    found = run.call("wiki_search", {"query": subj, "language": lang})
    if s_title not in found:
        return None, "subject_not_found"
    if not contains(E.norm_text(found.split(f"] {s_title} (")[1].split("\n\n")[0]), bridge_names):
        if not locate(run, s_title, lang, ROLE[role_of(chain)][lang], bridge_names):
            return None, "bridge_not_located"
    found = run.call("wiki_search", {"query": bridge, "language": lang})
    if o_title not in found:
        return None, "bridge_not_found"
    snippet = E.norm_text(found.split(f"] {o_title} (")[1].split("\n\n")[0])
    if not contains(snippet, answers):
        query = BORN[lang] if row["label"]["kind"] == "year" or "birth" in chain else bridge
        if not locate(run, o_title, lang, query, answers):
            return None, "answer_not_located"
    answer = row["label"]["answers"][0] if row["label"]["kind"] == "year" else label_in(entities.get(a_id), lang)
    final = f"{bridge}: {answer}.\n\nANSWER: {answer}"
    if E.grade_qa(final, {**row["label"], "mode": "default"}) != "correct":
        return None, "final_not_correct"
    run.messages.append({"role": "assistant", "content": final})
    return run.messages, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="build_deep_search.py output (train_rows.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    entities = json.loads((a.raw / "entities.json").read_text())
    searcher = W.Searcher(a.search)
    rows = [json.loads(line) for line in (a.data / "train_rows.jsonl").open()]
    stats, kept = collections.Counter(), 0
    with a.out.open("w") as fh:
        for row in rows:
            if kept >= a.limit:
                break
            messages, why = solve(searcher, row, entities)
            stats[why] += 1
            if messages:
                kept += 1
                turns = sum(m["role"] == "tool" for m in messages)
                fh.write(json.dumps({"id": row["id"], "language": row["language"], "subset": row["subset"],
                                     "evidence": row["evidence"], "tool_turns": turns, "messages": messages},
                                    ensure_ascii=False) + "\n")
    print(json.dumps({"kept": kept, "outcomes": dict(stats)}, indent=1))


if __name__ == "__main__":
    main()
