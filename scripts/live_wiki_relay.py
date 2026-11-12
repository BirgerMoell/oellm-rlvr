#!/usr/bin/env python3
"""Live Wikipedia for the wiki_search / open_page / find_in_page tools, for small pilots (compute nodes are offline).

Two processes, standard library only (plus oellm_rlvr.wiki_search for page parts and the result text):

  relay    on a node with internet (Leonardo: lrd_all_serial). Serves the search server's API (POST /search, /open,
           /find, GET /health) on the cluster network behind a bearer token, and answers from the live MediaWiki API
           of <lang>.wikipedia.org (the ten tool languages, nothing else). Every outbound request counts against a hard
           budget (--budget, default 100) kept in <dir>/budget.json, so restarts do not reset it; once it is used up
           the tools answer "Live search budget used up." Every answer is cached in <dir>/cache.jsonl (replayable) and
           every request logged in <dir>/audit.jsonl.
  forward  on each rollout node, in place of the offline server (search_api.python points at a wrapper that runs
           this): 127.0.0.1:<port> -> the relay URL and token from <dir>/relay.url and <dir>/relay.token.

  python3 live_wiki_relay.py relay --dir <dir> [--budget 100] [--port 0]
  python3 live_wiki_relay.py forward --dir <dir> --port 18090
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oellm_rlvr.wiki_search import (LANGUAGES, PAGE_MAX_CHARS, find_passages, format_find, format_page,  # noqa: E402
                                    format_results, snippet, split_parts)

USER_AGENT = "OpenEuroLLM-RL-live-search-pilot/0.1 (research; small capped pilot; https://openeurollm.eu)"
BUDGET_USED_UP = "Live search budget used up."


class Live:
    """MediaWiki API client with a persistent request budget, an answer cache and an audit log."""

    def __init__(self, root: Path, budget: int, timeout: float = 15.0) -> None:
        self.root, self.budget, self.timeout = root, budget, timeout
        self.lock = threading.Lock()
        self.budget_file = root / "budget.json"
        self.used = json.loads(self.budget_file.read_text())["used"] if self.budget_file.exists() else 0
        self.cache: dict[str, dict] = {}
        cache_file = root / "cache.jsonl"
        if cache_file.exists():
            for line in cache_file.open(encoding="utf-8"):
                row = json.loads(line)
                self.cache[row["key"]] = row["value"]

    def _log(self, name: str, row: dict) -> None:
        with open(self.root / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _api(self, language: str, params: dict) -> dict | None:
        """One MediaWiki API call, or None when the budget is used up."""
        assert language in LANGUAGES
        with self.lock:
            if self.used >= self.budget:
                return None
            self.used += 1
            self.budget_file.write_text(json.dumps({"used": self.used, "budget": self.budget}) + "\n")
        url = f"https://{language}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {**params, "format": "json", "formatversion": 2, "utf8": 1})
        start = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self._log("audit.jsonl", {"t": time.time(), "url": url, "ok": True, "s": round(time.time() - start, 2),
                                      "n": self.used})
            return data
        except Exception as exc:  # network/API failure: the tool reports it, the budget stays spent
            self._log("audit.jsonl", {"t": time.time(), "url": url, "ok": False, "error": repr(exc)[:300],
                                      "n": self.used})
            return {"error": str(exc)[:200]}

    def _cached(self, key: str, compute):
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        value = compute()
        if value is not None and "error" not in value:
            with self.lock:
                self.cache[key] = value
                self._log("cache.jsonl", {"key": key, "value": value})
        return value

    def search(self, query: str, language: str, k: int) -> dict:
        def compute():
            data = self._api(language, {"action": "query", "generator": "search", "gsrsearch": query,
                                        "gsrlimit": k, "prop": "extracts", "exintro": 1, "explaintext": 1,
                                        "exlimit": k})
            if data is None or "error" in data:
                return data
            pages = sorted((data.get("query") or {}).get("pages", []), key=lambda p: p.get("index", 0))
            return {"results": [{"title": p["title"], "language": language,
                                 "snippet": snippet((p.get("extract") or "").strip()), "score": None, "match": "live"}
                                for p in pages]}
        return self._cached(f"search|{language}|{k}|{query}", compute)

    def page(self, title: str, language: str) -> dict | None:
        def compute():
            data = self._api(language, {"action": "query", "titles": title, "redirects": 1, "prop": "extracts",
                                        "explaintext": 1})
            if data is None or "error" in data:
                return data
            pages = (data.get("query") or {}).get("pages", [])
            if not pages or pages[0].get("missing") or not pages[0].get("extract"):
                return {"missing": True, "title": title}
            return {"title": pages[0]["title"], "text": pages[0]["extract"][:PAGE_MAX_CHARS]}
        return self._cached(f"page|{language}|{title.strip().casefold()}", compute)


def _lang(body: dict) -> str:
    lang = str(body.get("language") or "en")
    return lang if lang in LANGUAGES else "en"


def handle(live: Live, path: str, body: dict) -> dict:
    language = _lang(body)
    if path == "/search":
        res = live.search(str(body.get("query", ""))[:300], language, max(1, min(int(body.get("k") or 5), 10)))
        if res is None:
            return {"results": [], "text": BUDGET_USED_UP}
        if "error" in res:
            return {"results": [], "text": f"Search failed: {res['error']}"}
        return {**res, "text": format_results(res["results"])}
    title = str(body.get("title", ""))[:300]
    page = live.page(title, language)
    if page is None:
        return {"error": BUDGET_USED_UP, "text": BUDGET_USED_UP}
    if "error" in page or page.get("missing"):
        msg = f"No article titled {title!r} in {language}.wikipedia."
        return {"error": msg, "text": msg}
    if path == "/open":
        parts = split_parts(page["text"])
        part = max(1, min(int(body.get("part") or 1), len(parts)))
        out = {"title": page["title"], "language": language, "part": part, "parts": len(parts), "text": parts[part - 1]}
        return {**out, "text": format_page(out)}
    found = {"title": page["title"], "language": language,
             "passages": find_passages(page["text"], str(body.get("query", "")))}
    return {**found, "text": format_find(found)}


def _server(port: int, host: str, on_post, on_health):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, on_health(self)) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/search", "/open", "/find"):
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                code, obj = on_post(self, body)
            except Exception as exc:
                code, obj = 500, {"error": repr(exc)[:200], "text": "Tool failed: server error."}
            self._send(code, obj)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def relay(a) -> None:
    root = a.dir
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    token_file = root / "relay.token"
    if not token_file.exists():
        old = os.umask(0o077)
        token_file.write_text(secrets.token_urlsafe(32) + "\n")
        os.umask(old)
    token = token_file.read_text().strip()
    live = Live(root, a.budget)
    host = socket.gethostbyname(socket.gethostname())  # the node's cluster address (compute nodes reach it)

    def authorized(h) -> bool:
        return hmac.compare_digest(h.headers.get("Authorization", ""), f"Bearer {token}")

    def on_post(h, body):
        if not authorized(h):
            return 401, {"error": "unauthorized"}
        return 200, handle(live, h.path, body)

    def on_health(h):
        return {"ok": True, "used": live.used, "budget": live.budget, "cached": len(live.cache)}

    srv = _server(a.port, host, on_post, on_health)
    url = f"http://{host}:{srv.server_address[1]}"
    (root / "relay.url").write_text(url + "\n")
    print(f"relay on {url} ({socket.gethostname()}), budget {live.used}/{live.budget} used", flush=True)
    srv.serve_forever()


def forward(a) -> None:
    def relay_url() -> str:  # re-read per request: a restarted relay (4 h serial limit) gets a new port
        return (a.dir / "relay.url").read_text().strip()

    token = (a.dir / "relay.token").read_text().strip()

    def on_post(h, body):
        url = relay_url()
        req = urllib.request.Request(url + h.path, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as resp:
                return 200, json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return 200, {"error": repr(exc)[:200], "text": f"Tool failed: relay unreachable ({type(exc).__name__})."}

    def on_health(h):  # the launcher's readiness check counts quoted language codes
        return {"ok": True, "loaded": LANGUAGES, "pages": LANGUAGES, "relay": relay_url()}

    srv = _server(a.port, "127.0.0.1", on_post, on_health)
    print(f"forward 127.0.0.1:{a.port} -> {relay_url()}", flush=True)
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relay")
    r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--budget", type=int, default=100)
    r.add_argument("--port", type=int, default=0)
    f = sub.add_parser("forward")
    f.add_argument("--dir", type=Path, required=True)
    f.add_argument("--port", type=int, default=18090)
    f.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()
    {"relay": relay, "forward": forward}[a.cmd](a)


if __name__ == "__main__":
    main()
