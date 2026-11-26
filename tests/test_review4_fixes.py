"""Regression tests for the fourth review round: per-call keyless sessions (no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi

from llama_index.tools.clawifi import ClawifiClientConfigError, ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import AsyncKeylessClient, KeylessClient
from tests.conftest import Router
from tests.fake_mcp import FakeMcpServer


def async_spec(server: FakeMcpServer) -> ClawifiToolSpec:
    return ClawifiToolSpec(
        async_keyless_client=AsyncKeylessClient(transport=httpx.MockTransport(server))
    )


def test_two_concurrent_searches_open_two_sessions_and_end_both() -> None:
    server = FakeMcpServer()
    spec = async_spec(server)

    async def both() -> list:
        return await asyncio.gather(spec.asearch("a"), spec.asearch("b"))

    first, second = asyncio.run(both())
    assert first and second
    assert server.sessions_opened == 2
    assert sorted(server.deleted_sessions) == ["sess-1-1", "sess-1-2"]
    assert server.open_sessions() == 0


def test_failed_call_still_deletes_its_session() -> None:
    server = FakeMcpServer()
    server.tool_error = "Sign in to use this tool"
    with pytest.raises(ClawifiKeylessError):
        asyncio.run(async_spec(server).afetch("https://example.com"))
    assert server.deleted_sessions == ["sess-1-1"]


def test_invalid_url_is_wrapped() -> None:
    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        ClawifiToolSpec(api_base="http://[bad").search("mcp")


def test_close_then_search_works() -> None:
    server = FakeMcpServer()
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    spec.search("a")
    spec.close()
    spec.search("b")
    assert server.open_sessions() == 0 and server.sessions_opened == 2


def test_key_and_only_async_client_sync_call_uses_the_key(router: Router) -> None:
    spec = ClawifiToolSpec(api_key="k", async_client=AsyncClawifi(api_key="async"))
    spec.fetch("https://example.com")
    assert router.rest[0].headers["authorization"] == "Bearer k"


def test_env_key_and_only_sync_client_async_call_uses_the_key(
    router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", "env-key")
    spec = ClawifiToolSpec(client=Clawifi(api_key="sync"))
    asyncio.run(spec.afetch("https://example.com"))
    assert router.rest[0].headers["authorization"] == "Bearer env-key"


def test_no_key_and_one_client_still_fails_clearly() -> None:
    spec = ClawifiToolSpec(async_client=AsyncClawifi(api_key="async"))
    with pytest.raises(ClawifiClientConfigError, match="Pass client too"):
        spec.search("mcp")
