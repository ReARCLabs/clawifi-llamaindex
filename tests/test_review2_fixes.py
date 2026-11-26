"""Regression tests for the second code review round (2026-11-26, no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import Clawifi

from llama_index.tools.clawifi import ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import (
    AsyncKeylessClient,
    KeylessClient,
    retry_after_seconds,
)
from tests.conftest import Router
from tests.fake_mcp import FakeMcpServer


def keyless_spec(server: FakeMcpServer, **kwargs) -> ClawifiToolSpec:
    transport = httpx.MockTransport(server)
    return ClawifiToolSpec(
        keyless_client=KeylessClient(transport=transport),
        async_keyless_client=AsyncKeylessClient(transport=transport),
        **kwargs,
    )


# Finding 1: reading spec.client does not turn keyless off -----------------------


def test_reading_client_keeps_keyless(router: Router) -> None:
    spec = ClawifiToolSpec()
    assert spec.client is spec.client  # the spec keeps the client it built
    assert spec._keyless() is True
    docs = spec.search("mcp")
    assert "keyless_note" in docs[0].metadata
    assert router.rest == []  # no keyless REST call (which would answer 401)
    assert router.mcp.methods().count("tools/call") == 1


def test_user_client_still_selects_rest() -> None:
    assert ClawifiToolSpec(client=Clawifi())._keyless() is False


# Finding 2: one MCP session per loop, ended with DELETE ---------------------------


def test_async_spec_reuses_one_session_and_aclose_ends_it() -> None:
    server = FakeMcpServer()
    spec = keyless_spec(server)

    async def run() -> None:
        await spec.asearch("a")
        await spec.afetch("https://example.com")
        await spec.asearch("b")
        await spec.aclose()

    asyncio.run(run())
    assert server.methods().count("initialize") == 1
    assert server.methods().count("tools/call") == 3
    assert server.open_sessions() == 0


def test_sync_spec_close_ends_session() -> None:
    server = FakeMcpServer()
    spec = keyless_spec(server)
    spec.search("a")
    spec.fetch("https://example.com")
    spec.close()
    assert server.methods().count("initialize") == 1
    assert server.deleted_sessions == ["sess-1-1"]


# Finding 7: SSE pings are skipped ----------------------------------------------


def test_sse_ping_events_are_skipped() -> None:
    server = FakeMcpServer(sse=True)
    server.sse_ping = True
    docs = keyless_spec(server).fetch("https://example.com")
    assert docs[0].metadata["title"] == "Example Domain"


# Finding 9: only finite, non-negative Retry-After ------------------------------


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-2"])
def test_retry_after_rejects_bad_values(value: str) -> None:
    assert retry_after_seconds(value) is None


def test_inf_retry_after_is_not_reported() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "inf"}
    )
    with pytest.raises(ClawifiKeylessError) as info:
        keyless_spec(server).search("mcp")
    assert info.value.retry_after is None


# Finding 10: max_results changed after construction is tracked --------------------


def test_max_results_set_later_is_a_user_request() -> None:
    server = FakeMcpServer()
    spec = keyless_spec(server)
    assert "Asked for" not in spec.search("mcp")[0].metadata["keyless_note"]
    spec.max_results = 10
    assert spec.max_results == 10
    assert "Asked for 10 results; capped at 3." in spec.search("mcp")[0].metadata["keyless_note"]
    spec.max_results = None  # back to the default, not a user request
    assert spec.max_results == 5
    assert "Asked for" not in spec.search("mcp")[0].metadata["keyless_note"]
