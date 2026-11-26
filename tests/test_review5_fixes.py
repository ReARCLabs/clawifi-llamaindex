"""Regression tests for the fifth review round: the single-POST model (no network)."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from clawifi import AsyncClawifi, Clawifi

import llama_index.tools.clawifi._keyless as keyless_module
from llama_index.tools.clawifi import ClawifiClientConfigError, ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import AsyncKeylessClient, KeylessClient
from tests.conftest import Router
from tests.fake_mcp import FakeMcpServer


def spec_for(server: FakeMcpServer) -> ClawifiToolSpec:
    return ClawifiToolSpec(
        keyless_client=KeylessClient(transport=httpx.MockTransport(server)),
        async_keyless_client=AsyncKeylessClient(transport=httpx.MockTransport(server)),
    )


def test_single_post_success_sync_and_async() -> None:
    server = FakeMcpServer()
    spec = spec_for(server)
    assert spec.search("a")
    assert asyncio.run(spec.afetch("https://example.com"))[0].text == "Body"
    assert server.methods() == ["tools/call", "tools/call"]
    assert server.delete_requests == []


@pytest.mark.parametrize("style", ["rpc-code", "rpc-message", "http400"])
def test_not_initialized_falls_back_and_deletes_the_session(style: str) -> None:
    server = FakeMcpServer(require_init=True, init_error=style)
    spec = spec_for(server)
    spec.search("a")
    asyncio.run(spec.asearch("b"))
    assert server.methods().count("initialize") == 2
    assert server.deleted_sessions == ["sess-1-1", "sess-1-2"]


def test_no_delete_without_a_session_id() -> None:
    server = FakeMcpServer(require_init=True, session_id=None)
    spec_for(server).fetch("https://example.com")
    assert server.delete_requests == []


def test_429_details_survive_a_failing_close(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_aclose(self: httpx.AsyncClient) -> None:
        raise httpx.ReadError("close failed")

    monkeypatch.setattr(httpx.AsyncClient, "aclose", failing_aclose)
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        429, json={"error": {"message": "busy"}}, headers={"Retry-After": "3"}
    )
    with pytest.raises(ClawifiKeylessError) as info:
        asyncio.run(spec_for(server).asearch("a"))
    assert info.value.status == 429 and info.value.retry_after == 3


def test_other_runtime_errors_propagate() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("a bug somewhere else")

    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(broken)))
    with pytest.raises(RuntimeError, match="a bug somewhere else"):
        spec.search("a")


def test_concurrent_first_sync_calls_build_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    real_client = httpx.Client

    class SlowClient(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            time.sleep(0.05)
            built.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(keyless_module.httpx, "Client", SlowClient)
    server = FakeMcpServer()
    client = KeylessClient(transport=httpx.MockTransport(server))
    threads = [
        threading.Thread(target=client.call_tool, args=("search_free", {"query": "a"}))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1


def test_built_clients_copy_the_given_api_base(router: Router) -> None:
    ClawifiToolSpec(
        api_key="k", async_client=AsyncClawifi(api_key="a", api_base="https://alt.example")
    ).fetch("https://example.com")
    asyncio.run(
        ClawifiToolSpec(
            api_key="k", client=Clawifi(api_key="s", api_base="https://alt2.example")
        ).afetch("https://example.com")
    )
    asyncio.run(
        ClawifiToolSpec(
            api_key="k",
            client=Clawifi(api_key="s", api_base="https://alt2.example"),
            api_base="https://own.example",
        ).afetch("https://example.com")
    )
    assert [r.url.host for r in router.rest] == ["alt.example", "alt2.example", "own.example"]
    assert all(r.headers["authorization"] == "Bearer k" for r in router.rest)


def test_whitespace_keys_count_as_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ClawifiToolSpec(api_key="   ")._keyless() is True
    monkeypatch.setenv("CLAWIFI_API_KEY", "  ")
    assert ClawifiToolSpec()._keyless() is True
    with pytest.raises(ClawifiClientConfigError):
        ClawifiToolSpec(api_key=" ", async_client=AsyncClawifi(api_key="a")).search("a")
