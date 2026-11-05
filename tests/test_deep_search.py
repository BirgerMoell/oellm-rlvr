"""deep_search: the builder keeps only two-hop questions the snapshot answers, and the scripted browsing agent solves
them through the three tools with trajectories the RL reward grades correct."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("Stemmer")

from oellm_rlvr import wiki_search as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILLER = "Filmen fick goda recensioner och visades på flera festivaler i Norden. " * 25


def _snapshot(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pages = [
        ("1", "Fönstret", f"Fönstret är en svensk dramafilm från 2010.\n\n{FILLER}\n\nHandling\n\nFilmen regisserades av Anna Berg och spelades in i Ystad."),
        ("2", "Anna Berg", "Anna Berg, född 3 maj 1961 i Malmö, är en svensk regissör och manusförfattare."),
        ("3", "Sjön (roman)", "Sjön är en roman av Per Lund, utgiven 1988. Den handlar om en sommar vid en sjö i Småland."),
        ("4", "Per Lund", "Per Lund är en svensk författare.\n\n" + "Han har skrivit romaner och noveller om landsbygden. " * 40
         + "\n\nBiografi\n\nPer Lund föddes 1950 i Växjö och studerade i Lund."),
        ("5", "Bron (film)", "Bron är en svensk film från 1999 i regi av Karin Ek (född 1970)."),
        ("6", "Karin Ek", "Karin Ek, född 1970, är en svensk regissör."),
    ]
    src = tmp_path / "wikipedia" / "20231101.sv"
    src.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": i, "title": t, "text": x} for i, t, x in pages]), src / "train-0.parquet")
    search = tmp_path / "search"
    W.build_leads(tmp_path / "wikipedia", "sv", search / "leads" / "sv.parquet")
    W.build_index(search / "leads" / "sv.parquet", "sv", search / "index" / "sv")
    W.build_pages(tmp_path / "wikipedia", "sv", search / "leads" / "sv.parquet", search / "pages")
    raw = tmp_path / "raw"
    raw.mkdir()
    facts = [
        {"type": "film_director_birth_year", "country": "Q34", "band": "tail", "s": "Q1", "sl": 4, "o": "Q2",
         "a": "1961-05-03T00:00:00Z", "t": "2010-01-01T00:00:00Z"},
        {"type": "book_author_birth_year", "country": "Q34", "band": "tail", "s": "Q3", "sl": 4, "o": "Q4",
         "a": "1950-01-01T00:00:00Z", "t": None},
        {"type": "film_director_birth_year", "country": "Q34", "band": "tail", "s": "Q5", "sl": 4, "o": "Q6",
         "a": "1970-01-01T00:00:00Z", "t": "1999-01-01T00:00:00Z"},
    ]
    (raw / "facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))

    def ent(label, title=None):
        return {"labels": {"sv": label, "en": label}, "aliases": {}, "sitelinks": {"sv": title or label}}
    entities = {"Q1": ent("Fönstret"), "Q2": ent("Anna Berg"), "Q3": ent("Sjön", "Sjön (roman)"), "Q4": ent("Per Lund"),
                "Q5": ent("Bron", "Bron (film)"), "Q6": ent("Karin Ek"), "Q34": ent("Sverige")}
    (raw / "entities.json").write_text(json.dumps(entities, ensure_ascii=False))
    return raw, search


def test_builder_and_solver(tmp_path):
    raw, search = _snapshot(tmp_path)
    out = tmp_path / "deep"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_deep_search.py"), "--raw", str(raw), "--search", str(search),
                    "--out", str(out)], check=True, capture_output=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["row_filter"]["one_hop"] == 1  # "Bron" names the director's birth year itself
    rows = [json.loads(line) for name in ("eval.jsonl", "train_rows.jsonl") for line in (out / name).open()]
    by_subject = {r["source"].split(">")[0]: r for r in rows}
    assert set(by_subject) == {"wikidata:Q1", "wikidata:Q3"}
    assert by_subject["wikidata:Q1"]["evidence"] == "snippet" and by_subject["wikidata:Q3"]["evidence"] == "body"
    assert by_subject["wikidata:Q1"]["prompt"].startswith("Vilket år föddes regissören av filmen ”Fönstret” (2010, Sverige)?")

    (out / "train_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    traj = out / "trajectories.jsonl"
    done = subprocess.run([sys.executable, str(ROOT / "scripts/deep_search_trajectories.py"), "--raw", str(raw),
                           "--search", str(search), "--data", str(out), "--out", str(traj)],
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout[done.stdout.index("{"):])["kept"] == 2, done.stdout
    for t in map(json.loads, traj.open()):
        names = [m["tool_calls"][0]["function"]["name"] for m in t["messages"] if m.get("tool_calls")]
        assert names[0] == "wiki_search" and t["messages"][-1]["content"].endswith(("ANSWER: 1961", "ANSWER: 1950"))
        if t["subset"] == "film_director_birth_year":  # director only in the body of the film article
            assert "find_in_page" in names or "open_page" in names
