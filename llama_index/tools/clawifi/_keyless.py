"""Keyless trial calls through Clawifi's hosted MCP server.

Clawifi's REST API needs a key. Without one, the hosted MCP server at
``https://api.clawifi.dev/mcp`` still answers a few trial tools, among them
``search_free`` and ``fetch_url`` (12 calls a minute and 50 a day per address,
at most 3 search results, no domain or time filters). This module speaks just
enough MCP over streamable HTTP to call those two tools: ``initialize``, then
``notifications/initialized``, then ``tools/call``, carrying ``Mcp-Session-Id``
and reading either a JSON or an SSE (``text/event-stream``) reply.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import threading
from typing import Any

import httpx

from clawifi import ClawifiError
from clawifi.errors import parse_retry_after

DEFAULT_API_BASE = "https://api.clawifi.dev"
PROTOCOL_VERSION = "2025-06-18"
KEYLESS_MAX_RESULTS = 3
KEYLESS_SEARCH_TOOL = "search_free"
KEYLESS_FETCH_TOOL = "fetch_url"
CLIENT_INFO = {"name": "clawifi-python-integration", "version": "0.1.0"}


class ClawifiKeylessError(ClawifiError):
    """A keyless trial call failed: HTTP error, JSON-RPC error or tool error."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: Any = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after


def mcp_url(api_base: str | None = None) -> str:
    base = (api_base or os.getenv("CLAWIFI_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    return f"{base}/mcp"


def keyless_note(dropped_filters: list[str], requested: int | None) -> str:
    note = (
        "Keyless trial through Clawifi's hosted MCP server: at most "
        f"{KEYLESS_MAX_RESULTS} results, no domain or time filters, "
        "12 calls a minute and 50 a day per address. Set CLAWIFI_API_KEY for full search."
    )
    if dropped_filters:
        note += f" Not applied: {', '.join(dropped_filters)}."
    if requested is not None and requested > KEYLESS_MAX_RESULTS:
        note += f" Asked for {requested} results; capped at {KEYLESS_MAX_RESULTS}."
    return note


# -- message helpers (shared by the sync and async clients) --------------------


def _headers(session_id: str | None, protocol: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if protocol:
        headers["MCP-Protocol-Version"] = protocol
    return headers


def _http_error(response: httpx.Response) -> ClawifiKeylessError:
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    message = ""
    code: Any = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
            code = error.get("code")
        elif isinstance(error, str):
            code = error
            message = str(payload.get("message") or payload.get("detail") or error)
        else:
            message = str(payload.get("message") or payload.get("detail") or "")
    if not message:
        message = response.text.strip()[:300] or response.reason_phrase
    text = f"Clawifi keyless call failed (HTTP {response.status_code}): {message}"
    if retry_after:
        text += f" Retry after {retry_after:g} s."
    return ClawifiKeylessError(
        text, status=response.status_code, code=code, retry_after=retry_after
    )


def _messages(response: httpx.Response) -> list[dict[str, Any]]:
    """Decode a JSON or SSE reply into JSON-RPC messages."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        messages: list[dict[str, Any]] = []
        data: list[str] = []
        for line in [*response.text.splitlines(), ""]:
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif line == "" and data:
                messages.extend(_as_list(json.loads("\n".join(data))))
                data = []
        return messages
    if not response.content:
        return []
    return _as_list(response.json())


def _as_list(payload: Any) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else [payload]
    return [item for item in items if isinstance(item, dict)]


def _result(response: httpx.Response, request_id: int) -> dict[str, Any]:
    for message in _messages(response):
        if message.get("id") != request_id:
            continue
        if "error" in message:
            error = message["error"] or {}
            raise ClawifiKeylessError(
                f"Clawifi keyless call failed: {error.get('message', error)}",
                code=error.get("code"),
            )
        return message.get("result") or {}
    raise ClawifiKeylessError("Clawifi MCP server sent no reply to the request.")


def _tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Turn a tools/call result into the tool's JSON object."""
    texts = [
        block.get("text", "")
        for block in result.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if result.get("isError"):
        raise ClawifiKeylessError(
            "Clawifi keyless call failed: " + (" ".join(texts).strip() or "tool error")
        )
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        inner = structured.get("result")
        if len(structured) == 1 and isinstance(inner, dict):
            return inner
        return structured
    for text in texts:
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ClawifiKeylessError("Clawifi MCP tool returned no JSON result.")


def _initialize_request(request_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        },
    }


_INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _call_request(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


# -- clients -------------------------------------------------------------------


class KeylessClient:
    """Synchronous keyless MCP client. One session, opened on the first call."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = mcp_url(api_base)
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.session_id: str | None = None
        self.protocol: str | None = None
        self._ready = False

    def close(self) -> None:
        self._http.close()

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._http.post(
                self.url, json=body, headers=_headers(self.session_id, self.protocol)
            )
        except httpx.HTTPError as exc:
            raise ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}") from exc

    def _initialize(self) -> None:
        self.session_id = None
        self.protocol = None
        request_id = next(self._ids)
        response = self._post(_initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        result = _result(response, request_id)
        self.session_id = response.headers.get("mcp-session-id")
        self.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        notified = self._post(_INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)
        self._ready = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._ready:
                self._initialize()
            request_id = next(self._ids)
            response = self._post(_call_request(request_id, name, arguments))
            if response.status_code == 404 and self.session_id:
                # The server dropped the session: open a new one and retry once.
                self._initialize()
                request_id = next(self._ids)
                response = self._post(_call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))


class AsyncKeylessClient:
    """Asynchronous keyless MCP client. One session, opened on the first call."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = mcp_url(api_base)
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._ids = itertools.count(1)
        self._lock: asyncio.Lock | None = None
        self.session_id: str | None = None
        self.protocol: str | None = None
        self._ready = False

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._http.post(
                self.url, json=body, headers=_headers(self.session_id, self.protocol)
            )
        except httpx.HTTPError as exc:
            raise ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}") from exc

    async def _initialize(self) -> None:
        self.session_id = None
        self.protocol = None
        request_id = next(self._ids)
        response = await self._post(_initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        result = _result(response, request_id)
        self.session_id = response.headers.get("mcp-session-id")
        self.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        notified = await self._post(_INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)
        self._ready = True

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if not self._ready:
                await self._initialize()
            request_id = next(self._ids)
            response = await self._post(_call_request(request_id, name, arguments))
            if response.status_code == 404 and self.session_id:
                await self._initialize()
                request_id = next(self._ids)
                response = await self._post(_call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))


def search_arguments(query: str) -> dict[str, Any]:
    return {"query": query}


def fetch_arguments(url: str, query: str | None) -> dict[str, Any]:
    return {"url": url, "query": query} if query else {"url": url}
