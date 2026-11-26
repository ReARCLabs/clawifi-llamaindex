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


def test_invalid_url_is_wrapped() -> None:
    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        ClawifiToolSpec(api_base="http://exa\tmple").search("mcp")


def test_close_then_search_works() -> None:
    server = FakeMcpServer()
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    spec.search("a")
    spec.close()
    spec.search("b")
    assert server.methods() == ["tools/call", "tools/call"]


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
