"""Regression tests for the third code review round: the simplification (no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi

import llama_index.tools.clawifi._keyless as keyless_module
from llama_index.tools.clawifi import ClawifiClientConfigError, ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import AsyncKeylessClient, KeylessClient
from tests.conftest import Router
from tests.fake_mcp import FakeMcpServer

# 1. No mirroring ---------------------------------------------------------------


def test_only_sync_client_then_async_call_fails_clearly(router: Router) -> None:
    spec = ClawifiToolSpec(client=Clawifi(api_key="k"))
    with pytest.raises(ClawifiClientConfigError, match="Pass async_client too"):
        asyncio.run(spec.afetch("https://example.com"))
    assert router.rest == [] and router.mcp.requests == []


def test_only_async_client_then_sync_call_fails_clearly(router: Router) -> None:
    spec = ClawifiToolSpec(async_client=AsyncClawifi(api_key="k"))
    with pytest.raises(ClawifiClientConfigError, match="Pass client too"):
        spec.search("mcp")
    assert router.rest == [] and router.mcp.requests == []


def test_api_key_builds_both_modes(router: Router) -> None:
    spec = ClawifiToolSpec(api_key="k")
    spec.fetch("https://example.com")
    asyncio.run(spec.afetch("https://example.com"))
    assert [r.headers["authorization"] for r in router.rest] == ["Bearer k", "Bearer k"]


def test_mirroring_helpers_are_gone() -> None:
    assert not hasattr(keyless_module, "http_settings_copyable")
    assert not hasattr(keyless_module, "require_copyable")


# 3. close() then another sync call works ------------------------------------------


def test_sync_call_after_close_works() -> None:
    server = FakeMcpServer()
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    spec.search("a")
    spec.close()
    assert spec.search("b")[0].metadata["keyless_note"]
    spec.close()
    assert server.open_sessions() == 0


# 5. Transport failures are wrapped ----------------------------------------------


@pytest.mark.parametrize(
    "error", [RuntimeError("client has been closed"), httpx.ReadTimeout("slow")]
)
def test_transport_failures_are_wrapped(error: Exception) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(broken))).fetch(
            "https://example.com"
        )
    spec = ClawifiToolSpec(
        async_keyless_client=AsyncKeylessClient(transport=httpx.MockTransport(broken))
    )
    with pytest.raises(ClawifiKeylessError, match="unreachable"):
        asyncio.run(spec.asearch("a"))
