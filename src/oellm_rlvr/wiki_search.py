"""Offline Wikipedia search for search-augmented RL and eval (hallucination plan, qwen35-posttrain
docs/OELLM_HALLUCINATION_PLAN.md).

Leonardo's compute nodes have no internet, so "web search" is a local BM25 index over the lead sections of
Wikipedia (wikimedia/wikipedia 20231101, CC BY-SA) in the ten eu_qa languages.

- ``build_leads``: title + lead (the first paragraphs, up to ~1,200 characters) of every article.
- ``build_index``: one BM25 index per language (bm25s; Snowball stemmer and stopwords where available).
- ``build_pages``: the full text of the same articles (capped at ``PAGE_MAX_CHARS``), row-aligned with the leads, as
  ``pages/<lang>.bin`` (UTF-8) + ``pages/<lang>.idx.npy`` (offsets), for browsing.
- ``create_app``: an HTTP server, ``POST /search {"query", "language", "k"}``, returning numbered results with
  title, language and snippet; with pages built also ``POST /open {"title", "language", "part"}`` (one part of an
  article) and ``POST /find {"title", "language", "query"}`` (the article's paragraphs that best match). Run it on the
  node next to the policy, like the code-execution API.
- The open-instruct tools ``wiki_search``, ``open_page`` and ``find_in_page`` call the server during rollouts. They
  are registered by ``compat.patch_wiki_search_tool`` (``OELLM_PATCH_WIKI_SEARCH_TOOL=1``).

The server needs bm25s and PyStemmer, which live in their own venv (not the training container); the tool only
needs open-instruct.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

LANGUAGES = ["en", "de", "fr", "es", "it", "nl", "sv", "pl", "pt", "fi"]
STEMMERS = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian", "nl": "dutch",
            "sv": "swedish", "pt": "portuguese", "fi": "finnish"}  # Snowball has no Polish stemmer
STOPWORDS = {"en": "en", "de": "german", "fr": "french", "es": "spanish", "it": "italian", "nl": "dutch",
             "sv": "swedish", "pt": "portuguese"}
LEAD_CHARS = 1200
SNIPPET_CHARS = 400
PAGE_CHARS = 2500          # one part of an opened article
PAGE_MAX_CHARS = 60000     # longer articles are cut (the tail is mostly lists and references)
FIND_RESULTS = 3
TOOL_DESCRIPTION = ("Search Wikipedia (offline snapshot, November 2023). Returns the best-matching articles with a "
                    "short excerpt. Use it to look up facts you are not sure about.")
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for, e.g. a name and what you want to know"},
        "language": {"type": "string", "enum": LANGUAGES,
                     "description": "Wikipedia edition to search (ISO code); default en"},
    },
    "required": ["query"],
}
OPEN_DESCRIPTION = ("Open a Wikipedia article (same offline snapshot) and read it, one part at a time. Use it when a "
                    "search excerpt does not contain the fact you need.")
OPEN_PARAMETERS = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Article title, as shown in the search results"},
        "language": {"type": "string", "enum": LANGUAGES, "description": "Wikipedia edition (ISO code); default en"},
        "part": {"type": "integer", "description": "Which part of the article to read, starting at 1; default 1"},
    },
    "required": ["title"],
}
FIND_DESCRIPTION = ("Find the passages of a Wikipedia article (same offline snapshot) that best match some words, "
                    "e.g. a date, a name or a keyword. Faster than reading the whole article.")
FIND_PARAMETERS = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Article title, as shown in the search results"},
        "query": {"type": "string", "description": "Words to look for in the article"},
        "language": {"type": "string", "enum": LANGUAGES, "description": "Wikipedia edition (ISO code); default en"},
    },
    "required": ["title", "query"],
}


def lead(text: str, max_chars: int = LEAD_CHARS) -> str:
    """The first paragraphs of an article (wikimedia/wikipedia plain text: paragraphs separated by blank lines)."""
    out: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if out and (len(para) < 40 and not para.endswith(".")):  # a section heading ends the lead
            break
        out.append(para)
        size += len(para)
        if size >= max_chars:
            break
    joined = "\n".join(out)
    return joined[:max_chars].rsplit(" ", 1)[0] if len(joined) > max_chars else joined


def build_leads(source_dir: str | Path, language: str, output: str | Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(source_dir).glob(f"20231101.{language}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {language} under {source_dir}")
    titles, leads, ids = [], [], []
    for path in files:
        table = pq.read_table(path, columns=["id", "title", "text"])
        for pid, title, text in zip(*(table.column(c).to_pylist() for c in ("id", "title", "text"))):
            body = lead(text or "")
            if len(body) < 40:
                continue
            ids.append(pid)
            titles.append(title)
            leads.append(body)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": ids, "title": titles, "lead": leads}), out, compression="zstd")
    return {"language": language, "articles": len(ids), "files": len(files)}


def _tokenizer_args(language: str) -> dict[str, Any]:
    import Stemmer

    return {"stopwords": STOPWORDS.get(language, []),
            "stemmer": Stemmer.Stemmer(STEMMERS[language]).stemWords if language in STEMMERS else None}


def build_index(leads_path: str | Path, language: str, output_dir: str | Path) -> dict[str, Any]:
    import bm25s
    import pyarrow.parquet as pq

    table = pq.read_table(leads_path, columns=["title", "lead"])
    # The title twice: queries name the entity, and leads of long articles bury it.
    corpus = [f"{t}. {t}. {body}" for t, body in zip(table.column("title").to_pylist(), table.column("lead").to_pylist())]
    tokens = bm25s.tokenize(corpus, show_progress=False, **_tokenizer_args(language))
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(output_dir))
    meta = {"language": language, "documents": len(corpus), "leads": str(leads_path)}
    Path(output_dir, "oellm_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def build_pages(source_dir: str | Path, language: str, leads_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Full text of the indexed articles, in the leads' row order (so a search hit's row is its page)."""
    import numpy as np
    import pyarrow.parquet as pq

    ids = pq.read_table(leads_path, columns=["id"]).column("id").to_pylist()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    offsets, pos, j = [0], 0, 0
    with open(out / f"{language}.bin", "wb") as fh:
        for path in sorted(Path(source_dir).glob(f"20231101.{language}/*.parquet")):
            table = pq.read_table(path, columns=["id", "text"])
            for pid, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
                if j < len(ids) and pid == ids[j]:
                    data = (text or "")[:PAGE_MAX_CHARS].encode("utf-8")
                    fh.write(data)
                    pos += len(data)
                    offsets.append(pos)
                    j += 1
    if j != len(ids):
        raise RuntimeError(f"{language}: matched {j} of {len(ids)} leads; the snapshot changed since the leads were built")
    np.save(out / f"{language}.idx.npy", np.asarray(offsets, dtype=np.int64))
    return {"language": language, "pages": j, "bytes": pos}


def split_parts(text: str, size: int = PAGE_CHARS) -> list[str]:
    """Article text in parts of about ``size`` characters, cut at paragraph boundaries where possible."""
    parts, cur = [], ""
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        while len(para) > size:  # one huge paragraph (a list or table flattened to text)
            if cur:
                parts.append(cur)
                cur = ""
            cut = para[:size].rsplit(" ", 1)[0] or para[:size]
            parts.append(cut)
            para = para[len(cut):].strip()
        if cur and len(cur) + len(para) + 1 > size:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts or [""]


def _terms(text: str) -> set[str]:
    """Lower-cased word prefixes (5 characters): crude stemming that works the same way in every language."""
    return {w.casefold()[:5] for w in _WORD.findall(text) if len(w) > 1 or w.isdigit()}


def find_passages(text: str, query: str, k: int = FIND_RESULTS) -> list[tuple[int, str]]:
    """(part number, paragraph) of the paragraphs sharing the most query terms; ties keep article order."""
    want = _terms(query)
    if not want:
        return []
    scored = []
    for part_no, part in enumerate(split_parts(text), 1):
        for para in part.split("\n"):
            hits = len(want & _terms(para))
            if hits:
                scored.append((-hits, part_no, para))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [(part_no, snippet(para, 700)) for _, part_no, para in scored[:k]]


_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


def norm_title(text: str) -> str:
    return " ".join(w.casefold() for w in _WORD.findall(text))


def title_keys(title: str) -> list[str]:
    """Exact key and, for "Name (qualifier)", the bare name as a weaker key."""
    keys = [norm_title(title)]
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    if bare != title:
        keys.append(norm_title(bare))
    return [k for k in keys if k]


def title_hits(query: str, exact: dict[str, list[int]], bare: dict[str, list[int]], max_hits: int = 2) -> list[int]:
    """Documents whose title occurs in the query as a whole word sequence, longest match first.

    BM25 alone ranks the article about an entity below shorter articles that mention it in their title (long leads
    are length-penalised), so a query naming the entity gets its article first. One-word titles count only if the
    word is capitalised in the query and at least 4 letters long, so common words do not match.
    """
    words = _WORD.findall(query)
    found: list[int] = []
    for n in range(min(6, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if n == 1 and (len(span[0]) < 4 or not span[0][:1].isupper()):
                continue
            key = " ".join(w.casefold() for w in span)
            for doc in exact.get(key, []) + bare.get(key, []):
                if doc not in found:
                    found.append(doc)
                if len(found) >= max_hits:
                    return found
    return found


class Searcher:
    """Loads one index per language on first use (memory-mapped) with its leads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._indexes: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}

    def _load(self, language: str):
        if language not in self._indexes:
            import bm25s
            import pyarrow.parquet as pq

            start = time.time()
            retriever = bm25s.BM25.load(str(self.root / "index" / language), mmap=True)
            # Leads stay an Arrow column (millions of Python strings take minutes to build); titles are needed as
            # Python strings once, for the exact-title maps.
            table = pq.read_table(self.root / "leads" / f"{language}.parquet", columns=["title", "lead"])
            titles = table.column("title").to_pylist()
            exact: dict[str, list[int]] = {}
            bare: dict[str, list[int]] = {}
            for doc, title in enumerate(titles):
                keys = title_keys(title)
                if not keys:  # titles without letters or digits ("!!!", "?")
                    continue
                exact.setdefault(keys[0], []).append(doc)
                for key in keys[1:]:
                    bare.setdefault(key, []).append(doc)
            self._indexes[language] = (retriever, titles, table.column("lead"), exact, bare)  # chunked: > 2 GB of text
            print(f"[wiki_search] loaded {language}: {len(titles)} articles in {time.time() - start:.0f}s", flush=True)
        return self._indexes[language]

    def search(self, query: str, language: str = "en", k: int = 5) -> list[dict[str, Any]]:
        import bm25s

        language = language if language in LANGUAGES else "en"
        retriever, titles, leads, exact, bare = self._load(language)
        picked = [(d, None) for d in title_hits(query, exact, bare)]
        tokens = bm25s.tokenize([query], show_progress=False, **_tokenizer_args(language))
        if tokens.vocab:
            docs, scores = retriever.retrieve(tokens, k=min(k + len(picked), len(titles)), show_progress=False)
            picked += [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        out, seen = [], set()
        for d, s in picked:
            if d in seen:
                continue
            seen.add(d)
            out.append({"title": titles[d], "language": language, "snippet": snippet(leads[d].as_py()),
                        "score": None if s is None else round(s, 3), "match": "title" if s is None else "bm25"})
            if len(out) >= k:
                break
        return out


    def has_pages(self, language: str) -> bool:
        return (self.root / "pages" / f"{language}.idx.npy").exists()

    def page_text(self, language: str, doc: int) -> str:
        import numpy as np

        if language not in self._pages:
            self._pages[language] = (np.load(self.root / "pages" / f"{language}.idx.npy", mmap_mode="r"),
                                     np.memmap(self.root / "pages" / f"{language}.bin", dtype=np.uint8, mode="r"))
        offsets, data = self._pages[language]
        return bytes(data[int(offsets[doc]):int(offsets[doc + 1])]).decode("utf-8", errors="replace")

    def resolve(self, title: str, language: str) -> int | None:
        """The article a title names: exact (case-insensitive) title, else the bare name, else the best search hit."""
        _, titles, _, exact, bare = self._load(language)
        keys = title_keys(title)
        if keys and exact.get(keys[0]):
            return exact[keys[0]][0]
        if keys and bare.get(keys[0]):
            return bare[keys[0]][0]
        hits = self.search(title, language, 1)
        if hits:
            return exact.get(title_keys(hits[0]["title"])[0], [None])[0]
        return None

    def open(self, title: str, language: str = "en", part: int = 1) -> dict[str, Any]:
        language = language if language in LANGUAGES else "en"
        doc = self.resolve(title, language) if self.has_pages(language) else None
        if doc is None:
            return {"error": f"No article titled {title!r} in {language}.wikipedia."}
        parts = split_parts(self.page_text(language, doc))
        part = max(1, min(int(part or 1), len(parts)))
        return {"title": self._indexes[language][1][doc], "language": language, "part": part, "parts": len(parts),
                "text": parts[part - 1]}

    def find(self, title: str, query: str, language: str = "en") -> dict[str, Any]:
        language = language if language in LANGUAGES else "en"
        doc = self.resolve(title, language) if self.has_pages(language) else None
        if doc is None:
            return {"error": f"No article titled {title!r} in {language}.wikipedia."}
        return {"title": self._indexes[language][1][doc], "language": language,
                "passages": find_passages(self.page_text(language, doc), query)}


def format_page(page: dict[str, Any]) -> str:
    if "error" in page:
        return page["error"]
    more = f" Next: part {page['part'] + 1}." if page["part"] < page["parts"] else ""
    return (f"{page['title']} ({page['language']}.wikipedia), part {page['part']} of {page['parts']}.{more}\n\n"
            f"{page['text']}")


def format_find(found: dict[str, Any]) -> str:
    if "error" in found:
        return found["error"]
    if not found["passages"]:
        return f"No passage in {found['title']} ({found['language']}.wikipedia) matches."
    return f"{found['title']} ({found['language']}.wikipedia):\n\n" + "\n\n".join(
        f"[part {n}] {para}" for n, para in found["passages"])


def snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(f"[{i}] {r['title']} ({r['language']}.wikipedia)\n{r['snippet']}" for i, r in enumerate(results, 1))


def create_app(root: str | Path, preload: list[str] | None = None):
    from fastapi import FastAPI, Request

    searcher = Searcher(root)
    app = FastAPI()
    if preload:  # load in the background; /health reports progress, requests for a loaded language work at once
        import threading

        def load_all() -> None:
            for language in preload:
                searcher._load(language)

        threading.Thread(target=load_all, daemon=True).start()

    # This module has postponed annotations, so FastAPI would see the string "Request" (imported only in this
    # function) and treat it as a query parameter (HTTP 422). Set the real class before registering the route.
    async def search(request):
        body = await request.json()
        start = time.time()
        results = searcher.search(str(body.get("query", "")), str(body.get("language") or "en"),
                                  max(1, min(int(body.get("k", 5)), 10)))
        return {"results": results, "text": format_results(results), "runtime": round(time.time() - start, 3)}

    search.__annotations__ = {"request": Request}
    app.post("/search")(search)

    async def open_page(request):
        body = await request.json()
        start = time.time()
        page = searcher.open(str(body.get("title", "")), str(body.get("language") or "en"), body.get("part") or 1)
        return {**page, "text": format_page(page), "runtime": round(time.time() - start, 3)}

    async def find_in_page(request):
        body = await request.json()
        start = time.time()
        found = searcher.find(str(body.get("title", "")), str(body.get("query", "")), str(body.get("language") or "en"))
        return {**found, "text": format_find(found), "runtime": round(time.time() - start, 3)}

    for name, fn in (("/open", open_page), ("/find", find_in_page)):
        fn.__annotations__ = {"request": Request}
        app.post(name)(fn)

    @app.get("/health")
    def health():
        return {"ok": True, "loaded": sorted(searcher._indexes),
                "pages": sorted(lang for lang in LANGUAGES if searcher.has_pages(lang))}

    return app


TOOLS = {
    # name: (description, parameters, required argument, server path, payload builder)
    "wiki_search": (TOOL_DESCRIPTION, TOOL_PARAMETERS, "query", "/search",
                    lambda a, n: {"query": a["query"], "language": a.get("language") or "en", "k": n}),
    "open_page": (OPEN_DESCRIPTION, OPEN_PARAMETERS, "title", "/open",
                  lambda a, n: {"title": a["title"], "language": a.get("language") or "en", "part": a.get("part") or 1}),
    "find_in_page": (FIND_DESCRIPTION, FIND_PARAMETERS, "query", "/find",
                     lambda a, n: {"title": a.get("title") or "", "query": a["query"],
                                   "language": a.get("language") or "en"}),
}


def endpoint_for(name: str, search_endpoint: str) -> str:
    """The server URL of a tool, from the ``/search`` URL of the same server."""
    return search_endpoint.rsplit("/", 1)[0] + TOOLS[name][3]


def browse_tool_class(name, base_tool, step_result, env_call_args_coercer, make_api_request, log_env_call):
    """Build one of the open-instruct tools in ``TOOLS`` against the installed open-instruct."""
    description, parameters, required, _, payload = TOOLS[name]

    class WikiTool(base_tool):
        config_name = name

        def __init__(self, call_name: str, api_endpoint: str, num_results: int = 5, timeout: int = 10) -> None:
            self.call_name = call_name
            self.api_endpoint = api_endpoint
            self.num_results = num_results
            self.timeout = timeout

        async def step(self, call):
            args = env_call_args_coercer(self.parameters, call.args)
            value = str(args.get(required) or "").strip()
            if not value:
                result = step_result(result=f"Missing {required}.", metadata={"error": f"empty {required}", "runtime": 0})
                log_env_call(self.call_name, "", result)
                return result
            args[required] = value
            start = time.time()
            response = await make_api_request(url=self.api_endpoint, timeout_seconds=self.timeout,
                                              json_payload=payload(args, self.num_results))
            if response.error:
                result = step_result(result=f"Tool failed: {response.error}",
                                     metadata={"error": response.error, "runtime": time.time() - start})
            else:
                result = step_result(result=response.data.get("text", "No results."),
                                     metadata={"error": "", "runtime": time.time() - start})
            log_env_call(self.call_name, value, result)
            return result

    WikiTool.description = description
    WikiTool.parameters = parameters
    return WikiTool


def search_tool_class(base_tool, step_result, env_call_args_coercer, make_api_request, log_env_call):
    """The ``wiki_search`` tool (kept for callers of the single-tool API)."""
    return browse_tool_class("wiki_search", base_tool, step_result, env_call_args_coercer, make_api_request,
                             log_env_call)


TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Hermes-style ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks (eval harness)."""
    calls = []
    for raw in TOOL_CALL.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("arguments", {}), dict):
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments") or {}})
    return calls
