"""Test setup: no test may reach the network, and no real key is read."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from tests.fake_mcp import FETCH_PAYLOAD, SEARCH_PAYLOAD, FakeMcpServer


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access is disabled in unit tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.delenv("CLAWIFI_API_KEY", raising=False)
    monkeypatch.delenv("CLAWIFI_API_BASE", raising=False)


class Router:
    """Sends /mcp to the fake MCP server and every other path to a REST recorder."""

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
    """Route the default httpx transports to in-memory fakes (no network).

    Patching the transports, not the clients, keeps every client the code builds
    a default one. The async transport also behaves like a real connection pool:
    used from a second event loop it fails with "Event loop is closed".
    """
    route = Router(FakeMcpServer())

    def handle_request(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        request.read()
        return route(request)

    async def handle_async_request(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        loop = asyncio.get_running_loop()
        seen = getattr(self, "_test_loop", None)
        if seen is None:
            self._test_loop = loop  # type: ignore[attr-defined]
        elif seen is not loop:
            raise RuntimeError("Event loop is closed")
        await request.aread()
        return route(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    return route
