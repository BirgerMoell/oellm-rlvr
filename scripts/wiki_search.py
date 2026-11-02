#!/usr/bin/env python3
"""Offline Wikipedia search (oellm_rlvr.wiki_search): build leads and BM25 indexes, or serve them.
Runs in the search venv (bm25s, PyStemmer, fastapi, uvicorn), not the training container.
  python scripts/wiki_search.py build --source $S/wikipedia --root $S --language sv
  python scripts/wiki_search.py serve --root $S --port 18090 [--preload en sv]
  python scripts/wiki_search.py query --root $S --language sv "Selma Lagerlöf född"
Layout under --root: leads/<lang>.parquet, index/<lang>/ (bm25s files + oellm_meta.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oellm_rlvr import wiki_search as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, required=True, help="snapshot of wikimedia/wikipedia (20231101.<lang>/)")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--language", required=True, choices=W.LANGUAGES)
    s = sub.add_parser("serve")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=18090)
    s.add_argument("--preload", nargs="*", default=[])
    q = sub.add_parser("query")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--language", default="en")
    q.add_argument("text")
    a = ap.parse_args()
    if a.cmd == "build":
        start = time.time()
        leads = a.root / "leads" / f"{a.language}.parquet"
        if not leads.exists():
            print(json.dumps(W.build_leads(a.source, a.language, leads)), flush=True)
        print(json.dumps(W.build_index(leads, a.language, a.root / "index" / a.language)),
              f"{time.time() - start:.0f}s", flush=True)
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run(W.create_app(a.root, a.preload), host=a.host, port=a.port, log_level="warning")
    else:
        print(W.format_results(W.Searcher(a.root).search(a.text, a.language)))


if __name__ == "__main__":
    main()
