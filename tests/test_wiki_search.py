import asyncio
import time
from dataclasses import asdict, dataclass
from types import ModuleType, SimpleNamespace

from oellm_rlvr import wiki_search as W
from oellm_rlvr.compat import patch_wiki_search_tool


def test_lead_stops_at_first_heading_and_caps_length():
    text = "Selma Lagerlöf (1858–1940) var en svensk författare.\n\nHon fick Nobelpriset 1909.\n\nBiografi\n\nUppväxt."
    assert W.lead(text) == "Selma Lagerlöf (1858–1940) var en svensk författare.\nHon fick Nobelpriset 1909."
    assert len(W.lead("Ord " * 2000)) <= W.LEAD_CHARS


def test_format_and_parse():
    out = W.format_results([{"title": "Selma Lagerlöf", "language": "sv", "snippet": "svensk författare", "score": 3.0}])
    assert out.startswith("[1] Selma Lagerlöf (sv.wikipedia)")
    assert W.format_results([]) == "No results."
    calls = W.parse_tool_calls('x <tool_call>{"name": "wiki_search", "arguments": {"query": "Lagerlöf"}}</tool_call>')
    assert calls == [{"name": "wiki_search", "arguments": {"query": "Lagerlöf"}}]
    assert W.parse_tool_calls("<tool_call>{bad json}</tool_call>") == []


def test_patch_registers_tool_that_calls_the_server():
    module = ModuleType("fake_tools")

    class Tool:
        pass

    @dataclass
    class BaseEnvConfig:
        pass

    class StepResult(SimpleNamespace):
        pass

    sent = {}

    async def make_api_request(url, timeout_seconds, json_payload):
        sent.update(url=url, payload=json_payload)
        return SimpleNamespace(error=None, data={"text": "[1] Selma Lagerlöf (sv.wikipedia)\n..."})

    module.Tool, module.BaseEnvConfig, module.StepResult = Tool, BaseEnvConfig, StepResult
    module.coerce_args = lambda params, args: dict(args)
    module.make_api_request = make_api_request
    module.log_env_call = lambda *a: None
    module.TOOL_REGISTRY = {}
    assert patch_wiki_search_tool(module) is True and patch_wiki_search_tool(module) is False
    config = module.TOOL_REGISTRY["wiki_search"](api_endpoint="http://127.0.0.1:1/search")
    assert set(asdict(config)) == {"api_endpoint", "num_results", "timeout"}  # kwargs open-instruct passes on
    tool = config.tool_class(call_name="wiki_search", api_endpoint=config.api_endpoint)
    result = asyncio.run(tool.step(SimpleNamespace(args={"query": "Selma Lagerlöf", "language": "sv"})))
    assert result.result.startswith("[1] Selma") and sent["payload"] == {"query": "Selma Lagerlöf", "language": "sv", "k": 5}
    empty = asyncio.run(tool.step(SimpleNamespace(args={"query": " "})))
    assert empty.metadata["error"] == "empty query"


def test_title_hits_put_the_named_entity_first():
    titles = ["Selma Lagerlöf", "Selma Lagerlöf (målning)", "Mårbacka", "År", "Selma"]
    exact, bare = {}, {}
    for doc, t in enumerate(titles):
        keys = W.title_keys(t)
        exact.setdefault(keys[0], []).append(doc)
        for key in keys[1:]:
            bare.setdefault(key, []).append(doc)
    assert W.title_hits("Vilket år föddes Selma Lagerlöf?", exact, bare) == [0, 1]
    assert W.title_hits("vilket år föddes hon", exact, bare) == []  # one-word, lower-case "år" does not match
    assert W.title_hits("Mårbacka gård", exact, bare) == [2]
    assert W.title_keys("!!!") == [] and W.title_keys("?") == []


def test_build_and_search_end_to_end(tmp_path):
    import pytest

    pytest.importorskip("bm25s")
    pytest.importorskip("Stemmer")
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "wikipedia" / "20231101.sv"
    src.mkdir(parents=True)
    rows = [
        {"id": "1", "title": "Selma Lagerlöf", "text": "Selma Lagerlöf, född 20 november 1858 på Mårbacka, var en svensk författare. " * 8 + "\n\nBiografi\n\nMer."},
        {"id": "2", "title": "Selma Lagerlöf (målning)", "text": "Selma Lagerlöf är ett porträtt av Selma Lagerlöf från 1908, målat av Carl Larsson."},
        {"id": "3", "title": "!!!", "text": "!!! är ett amerikanskt rockband som bildades 1996 i Sacramento."},
        {"id": "4", "title": "Mårbacka", "text": "Mårbacka är en herrgård i Sunne kommun i Värmland, känd som Selma Lagerlöfs hem."},
    ]
    # Two row groups / chunks, like the multi-file real snapshot.
    pq.write_table(pa.Table.from_pylist(rows[:2]), src / "train-00000-of-00002.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), src / "train-00001-of-00002.parquet")
    root = tmp_path / "search"
    assert W.build_leads(tmp_path / "wikipedia", "sv", root / "leads" / "sv.parquet")["articles"] == 4
    W.build_index(root / "leads" / "sv.parquet", "sv", root / "index" / "sv")
    results = W.Searcher(root).search("Vilket år föddes Selma Lagerlöf?", "sv", k=3)
    assert results[0]["title"] == "Selma Lagerlöf" and results[0]["match"] == "title"
    assert "1858" in results[0]["snippet"]
    assert W.Searcher(root).search("rockband Sacramento", "sv", k=2)[0]["title"] == "!!!"

    from fastapi.testclient import TestClient

    app = W.create_app(root, preload=["sv"])
    client = TestClient(app)
    for _ in range(100):
        if client.get("/health").json()["loaded"] == ["sv"]:
            break
        time.sleep(0.05)
    response = client.post("/search", json={"query": "Mårbacka", "language": "sv", "k": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"].startswith("[1] Mårbacka (sv.wikipedia)")
