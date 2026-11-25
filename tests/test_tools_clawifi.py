from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi, ClawifiAPIError
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.clawifi import ClawifiToolSpec

SEARCH = {
    "success": True,
    "results": [
        {"url": "https://example.com/a", "title": "A", "snippet": "first", "rank": 1, "data": {}},
        {"url": "https://example.com/b", "title": "B", "snippet": None, "rank": 2, "data": {}},
    ],
    "errors": [],
}
FETCH = {
    "success": True,
    "content_status": "ok",
    "errors": [],
    "data": {"url": "https://example.com", "title": "Example", "content_markdown": "# Hi\n\nBody"},
}


class Recorder:
    def __init__(self, routes: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, payload = self.routes[request.url.path]
        return httpx.Response(status, json=payload)

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def spec_for(rec: Recorder, **kwargs: Any) -> ClawifiToolSpec:
    transport = httpx.MockTransport(rec)
    return ClawifiToolSpec(
        client=Clawifi(api_key="k", transport=transport, max_retries=0),
        async_client=AsyncClawifi(api_key="k", transport=transport, max_retries=0),
        **kwargs,
    )


def test_class() -> None:
    assert BaseToolSpec.__name__ in [b.__name__ for b in ClawifiToolSpec.__mro__]


def test_tool_list_has_sync_and_async() -> None:
    tools = ClawifiToolSpec().to_tool_list()
    names = [t.metadata.name for t in tools]
    assert names == ["search", "fetch"]
    for tool in tools:
        assert tool.fn is not None and tool.async_fn is not None
        assert "query" in tool.metadata.fn_schema_str or "url" in tool.metadata.fn_schema_str


def test_search_documents() -> None:
    rec = Recorder({"/search/free": (200, SEARCH)})
    docs = spec_for(rec, max_results=4).search("mcp", time_range="month")
    assert rec.body() == {"query": "mcp", "max_results": 4, "time_range": "month"}
    assert docs[0].text == "first"
    assert docs[0].metadata == {"url": "https://example.com/a", "title": "A", "rank": 1}
    assert docs[1].text == "B"  # falls back to the title when there is no snippet


def test_search_failure_raises() -> None:
    rec = Recorder({"/search/free": (200, {"success": False, "results": [], "errors": ["x"]})})
    with pytest.raises(RuntimeError):
        spec_for(rec).search("mcp")


def test_api_error_propagates() -> None:
    rec = Recorder({"/fetch": (401, {"error": {"code": "unauthorized", "message": "bad key"}})})
    with pytest.raises(ClawifiAPIError):
        spec_for(rec).fetch("https://example.com")


def test_fetch_document_and_query() -> None:
    rec = Recorder({"/fetch": (200, FETCH)})
    docs = spec_for(rec).fetch("https://example.com", query="what is it")
    assert rec.body() == {"url": "https://example.com", "query": "what is it"}
    assert docs[0].text == "# Hi\n\nBody"
    assert docs[0].metadata["title"] == "Example"
    assert docs[0].metadata["content_status"] == "ok"


def test_fetch_truncates() -> None:
    long = {**FETCH, "data": {**FETCH["data"], "content_markdown": "y" * 30}}
    rec = Recorder({"/fetch": (200, long)})
    docs = spec_for(rec, max_chars=5).fetch("https://example.com")
    assert docs[0].text == "yyyyy" and docs[0].metadata["truncated"] is True


async def test_async_tools() -> None:
    rec = Recorder({"/search/free": (200, SEARCH), "/fetch": (200, FETCH)})
    spec = spec_for(rec)
    tools = {t.metadata.name: t for t in spec.to_tool_list()}
    out = await tools["search"].acall(query="mcp")
    assert "first" in str(out)
    docs = await spec.afetch("https://example.com")
    assert docs[0].metadata["url"] == "https://example.com"


def test_keyless_and_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ClawifiToolSpec().client.config.api_key is None
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    assert ClawifiToolSpec().client.config.api_key == "env-key"
    assert ClawifiToolSpec(api_key="explicit").client.config.api_key == "explicit"
