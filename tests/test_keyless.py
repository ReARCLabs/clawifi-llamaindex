from __future__ import annotations

import httpx
import pytest

from llama_index.tools.clawifi import ClawifiToolSpec
from llama_index.tools.clawifi._keyless import (
    AsyncKeylessClient,
    ClawifiKeylessError,
    KeylessClient,
)
from tests.fake_mcp import FakeMcpServer


def keyless_spec(server: FakeMcpServer, **kwargs) -> ClawifiToolSpec:
    transport = httpx.MockTransport(server)
    return ClawifiToolSpec(
        keyless_client=KeylessClient(transport=transport),
        async_keyless_client=AsyncKeylessClient(transport=transport),
        **kwargs,
    )


def test_search_keyless_caps_and_notes() -> None:
    server = FakeMcpServer()
    docs = keyless_spec(server).search("mcp", max_results=10, time_range="month")
    assert server.methods() == ["tools/call"]  # round 5: one POST per call
    assert server.calls() == [{"name": "search_free", "arguments": {"query": "mcp"}}]
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3]
    note = docs[0].metadata["keyless_note"]
    assert "Not applied: time_range." in note and "Asked for 10 results; capped at 3." in note
    assert {"url", "title", "rank", "keyless_note"} == set(docs[0].metadata)


def test_search_keyless_smaller_limit() -> None:
    docs = keyless_spec(FakeMcpServer(), max_results=2).search("mcp")
    assert len(docs) == 2 and "capped" not in docs[0].metadata["keyless_note"]


def test_fetch_keyless_sse_same_fields() -> None:
    server = FakeMcpServer(sse=True)
    docs = keyless_spec(server).fetch("https://example.com", query="what")
    assert server.calls()[0] == {
        "name": "fetch_url",
        "arguments": {"url": "https://example.com", "query": "what"},
    }
    assert docs[0].text == "Body"
    assert docs[0].metadata == {
        "url": "https://example.com/",
        "title": "Example Domain",
        "content_status": "ok",
    }


def test_rate_limit_and_tool_error() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "4"}
    )
    spec = keyless_spec(server)
    with pytest.raises(ClawifiKeylessError) as info:
        spec.search("mcp")
    assert info.value.status == 429 and info.value.retry_after == 4
    server.tool_error = "Sign in to use this tool"
    with pytest.raises(ClawifiKeylessError, match="Sign in"):
        spec.fetch("https://example.com")


async def test_async_keyless_through_tool_list() -> None:
    server = FakeMcpServer(structured=True)
    tools = {t.metadata.name: t for t in keyless_spec(server).to_tool_list()}
    out = await tools["search"].acall(query="mcp")
    assert "keyless_note" in str(out.raw_output[0].metadata)
    docs = await keyless_spec(FakeMcpServer(sse=True)).afetch("https://example.com")
    assert docs[0].metadata["title"] == "Example Domain"


def test_key_selects_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ClawifiToolSpec()._keyless() is True
    assert ClawifiToolSpec(api_key="k")._keyless() is False
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    assert ClawifiToolSpec()._keyless() is False
