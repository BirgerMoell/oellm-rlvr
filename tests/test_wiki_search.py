import asyncio
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
