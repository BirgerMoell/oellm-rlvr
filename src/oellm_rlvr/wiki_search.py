"""Offline Wikipedia search for search-augmented RL and eval (hallucination plan, qwen35-posttrain
docs/OELLM_HALLUCINATION_PLAN.md).

Leonardo's compute nodes have no internet, so "web search" is a local BM25 index over the lead sections of
Wikipedia (wikimedia/wikipedia 20231101, CC BY-SA) in the ten eu_qa languages.

- ``build_leads``: title + lead (the first paragraphs, up to ~1,200 characters) of every article.
- ``build_index``: one BM25 index per language (bm25s; Snowball stemmer and stopwords where available).
- ``create_app``: an HTTP server, ``POST /search {"query", "language", "k"}``, returning numbered results with
  title, language and snippet. Run it on the node next to the policy, like the code-execution API.
- ``WikiSearchTool``: the open-instruct tool that calls the server during rollouts. It is registered as
  ``wiki_search`` by ``compat.patch_wiki_search_tool`` (``OELLM_PATCH_WIKI_SEARCH_TOOL=1``).

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
            self._indexes[language] = (retriever, titles, table.column("lead").combine_chunks(), exact, bare)
            print(f"[wiki_search] loaded {language}: {len(titles)} articles in {time.time() - start:.0f}s", flush=True)
        return self._indexes[language]

    def search(self, query: str, language: str = "en", k: int = 5) -> list[dict[str, Any]]:
        import bm25s

        language = language if language in LANGUAGES else "en"
        retriever, titles, leads, exact, bare = self._load(language)
        picked = [(d, None) for d in title_hits(query, exact, bare)]
        tokens = bm25s.tokenize([query], show_progress=False, **_tokenizer_args(language))
        if tokens.vocab:
            docs, scores = retriever.retrieve(tokens, k=k + len(picked), show_progress=False)
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


def snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(f"[{i}] {r['title']} ({r['language']}.wikipedia)\n{r['snippet']}" for i, r in enumerate(results, 1))


def create_app(root: str | Path, preload: list[str] | None = None):
    from fastapi import FastAPI
    from pydantic import BaseModel

    searcher = Searcher(root)
    app = FastAPI()
    if preload:  # load in the background; /health reports progress, requests for a loaded language work at once
        import threading

        def load_all() -> None:
            for language in preload:
                searcher._load(language)

        threading.Thread(target=load_all, daemon=True).start()

    class Query(BaseModel):
        query: str
        language: str = "en"
        k: int = 5

    @app.post("/search")
    def search(q: Query) -> dict[str, Any]:
        start = time.time()
        results = searcher.search(q.query, q.language, max(1, min(q.k, 10)))
        return {"results": results, "text": format_results(results), "runtime": round(time.time() - start, 3)}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "loaded": sorted(searcher._indexes)}

    return app


def search_tool_class(base_tool, step_result, env_call_args_coercer, make_api_request, log_env_call):
    """Build the open-instruct tool class against the installed open-instruct (see compat.patch_wiki_search_tool)."""

    class WikiSearchTool(base_tool):
        config_name = "wiki_search"
        description = TOOL_DESCRIPTION
        parameters = TOOL_PARAMETERS

        def __init__(self, call_name: str, api_endpoint: str, num_results: int = 5, timeout: int = 10) -> None:
            self.call_name = call_name
            self.api_endpoint = api_endpoint
            self.num_results = num_results
            self.timeout = timeout

        async def step(self, call):
            args = env_call_args_coercer(self.parameters, call.args)
            query = (args.get("query") or "").strip()
            if not query:
                result = step_result(result="Empty query.", metadata={"error": "empty query", "runtime": 0})
                log_env_call(self.call_name, "", result)
                return result
            start = time.time()
            response = await make_api_request(url=self.api_endpoint, timeout_seconds=self.timeout,
                                              json_payload={"query": query, "language": args.get("language") or "en",
                                                            "k": self.num_results})
            if response.error:
                result = step_result(result=f"Search failed: {response.error}",
                                     metadata={"error": response.error, "runtime": time.time() - start})
            else:
                result = step_result(result=response.data.get("text", "No results."),
                                     metadata={"error": "", "runtime": time.time() - start})
            log_env_call(self.call_name, query, result)
            return result

    return WikiSearchTool


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
