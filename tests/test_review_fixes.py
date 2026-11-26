"""Regression tests for the 2026-11-26 code review findings (no network)."""

from __future__ import annotations

import asyncio
import inspect
import tomllib
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi, ClawifiError

import llama_index.tools.clawifi._keyless as keyless_module
from llama_index.tools.clawifi import ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import KeylessClient, retry_after_seconds
from tests.fake_mcp import FETCH_PAYLOAD, SEARCH_PAYLOAD, FakeMcpServer


class Router:
    def __init__(self, mcp: FakeMcpServer) -> None:
        self.mcp = mcp
        self.rest: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp":
            return self.mcp(request)
        self.rest.append(request)
        payload = SEARCH_PAYLOAD if request.url.path == "/search/free" else FETCH_PAYLOAD
        return httpx.Response(200, json=payload)


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Router:
    """Route self-built httpx clients to fakes; async clients fail across event loops."""
    route = Router(FakeMcpServer())
    real_async, real_sync = httpx.AsyncClient, httpx.Client

    class LoopBoundAsyncClient(real_async):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, transport: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, transport=transport or httpx.MockTransport(route), **kwargs)
            self.loop_seen: asyncio.AbstractEventLoop | None = None

        async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            loop = asyncio.get_running_loop()
            if self.loop_seen is None:
                self.loop_seen = loop
            elif self.loop_seen is not loop:
                raise RuntimeError("Event loop is closed")
            return await super().send(request, **kwargs)

    class RoutedClient(real_sync):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, transport: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, transport=transport or httpx.MockTransport(route), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", LoopBoundAsyncClient)
    monkeypatch.setattr(httpx, "Client", RoutedClient)
    return route


# Finding 1 ----------------------------------------------------------------------


def test_keyless_async_works_across_two_event_loops(router: Router) -> None:
    spec = ClawifiToolSpec()
    for _ in range(2):
        docs = asyncio.run(spec.asearch("mcp"))
        assert len(docs) == 3
    assert router.mcp.methods().count("tools/call") == 2


def test_rest_async_works_across_two_event_loops(router: Router) -> None:
    spec = ClawifiToolSpec(api_key="k")
    for _ in range(2):
        docs = asyncio.run(spec.afetch("https://example.com"))
        assert docs[0].metadata["title"] == "Example Domain"
    assert [r.url.path for r in router.rest] == ["/fetch", "/fetch"]


# Finding 2 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="data: {bad\n\n", headers={"content-type": "text/event-stream"}),
        httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"}),
        httpx.Response(200, json={"jsonrpc": "2.0", "id": 2, "error": "Unknown tool"}),
        httpx.Response(200, json={"jsonrpc": "2.0", "id": 2, "result": [1, 2]}),
    ],
)
def test_unreadable_replies_raise_keyless_error(response: httpx.Response) -> None:
    server = FakeMcpServer()
    server.fail_next = response
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    with pytest.raises(ClawifiKeylessError) as info:
        spec.search("mcp")
    assert isinstance(info.value, ClawifiError)


# Finding 3 ----------------------------------------------------------------------


def test_sync_client_only_then_async_call_uses_rest(router: Router) -> None:
    spec = ClawifiToolSpec(client=Clawifi(api_key="sync-key"))
    docs = asyncio.run(spec.asearch("mcp"))
    assert router.mcp.requests == []
    assert router.rest[0].headers["authorization"] == "Bearer sync-key"
    assert "keyless_note" not in docs[0].metadata


def test_async_client_only_then_sync_call_uses_rest(router: Router) -> None:
    spec = ClawifiToolSpec(async_client=AsyncClawifi(api_key="async-key"))
    docs = spec.fetch("https://example.com")
    assert docs[0].text == "Body"
    assert router.mcp.requests == []
    assert router.rest[0].headers["authorization"] == "Bearer async-key"


def test_keyless_only_without_key_and_without_clients() -> None:
    assert ClawifiToolSpec()._keyless() is True
    assert ClawifiToolSpec(client=Clawifi())._keyless() is False
    assert ClawifiToolSpec(async_client=AsyncClawifi())._keyless() is False


# Finding 6 ----------------------------------------------------------------------


def test_note_only_mentions_cap_for_user_requests() -> None:
    server = FakeMcpServer()
    transport = httpx.MockTransport(server)

    def note(spec: ClawifiToolSpec, **kwargs: Any) -> str:
        return spec.search("mcp", **kwargs)[0].metadata["keyless_note"]

    default = ClawifiToolSpec(keyless_client=KeylessClient(transport=transport))
    assert default.max_results == 5
    assert "Asked for" not in note(default)
    assert "Asked for 9 results; capped at 3." in note(default, max_results=9)
    configured = ClawifiToolSpec(max_results=7, keyless_client=KeylessClient(transport=transport))
    assert "Asked for 7 results; capped at 3." in note(configured)


# Finding 8 ----------------------------------------------------------------------


def test_empty_keyless_search_keeps_the_note() -> None:
    empty = {**SEARCH_PAYLOAD, "results": []}
    server = FakeMcpServer(tool_payloads={"search_free": empty, "fetch_url": FETCH_PAYLOAD})
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    docs = spec.search("nothing matches")
    assert len(docs) == 1
    assert docs[0].metadata["result_count"] == 0
    assert docs[0].text == docs[0].metadata["keyless_note"]
    assert "Keyless trial" in docs[0].text


# Finding 9 ----------------------------------------------------------------------


def test_httpx_is_a_declared_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert any(dep.replace(" ", "").startswith("httpx>=") for dep in deps)


def test_no_private_clawifi_module_is_imported() -> None:
    assert "clawifi.errors" not in inspect.getsource(keyless_module)


def test_retry_after_seconds() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    assert retry_after_seconds("4") == 4
    assert (
        retry_after_seconds(format_datetime(now + timedelta(seconds=45), usegmt=True), now=now)
        == 45
    )
    assert retry_after_seconds("later") is None
