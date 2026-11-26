"""Keyless trial calls through Clawifi's hosted MCP server.

Clawifi's REST API needs a key. Without one, the hosted MCP server at
``https://api.clawifi.dev/mcp`` still answers a few trial tools, among them
``search_free`` and ``fetch_url`` (12 calls a minute and 50 a day per address,
at most 3 search results, no domain or time filters). This module speaks just
enough MCP over streamable HTTP to call those two tools: ``initialize``, then
``notifications/initialized``, then ``tools/call``, carrying ``Mcp-Session-Id``
and reading either a JSON or an SSE (``text/event-stream``) reply.

Every failure, including a reply that cannot be decoded, is raised as
:class:`ClawifiKeylessError`, a subclass of ``clawifi.ClawifiError``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import threading
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from clawifi import ClawifiError

DEFAULT_API_BASE = "https://api.clawifi.dev"
PROTOCOL_VERSION = "2025-06-18"
KEYLESS_MAX_RESULTS = 3
KEYLESS_SEARCH_TOOL = "search_free"
KEYLESS_FETCH_TOOL = "fetch_url"
CLIENT_INFO = {"name": "clawifi-python-integration", "version": "0.1.0"}


class ClawifiKeylessError(ClawifiError):
    """A keyless trial call failed: HTTP error, JSON-RPC error, tool error or bad reply."""

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


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header: delay seconds or an HTTP date.

    Only finite, non-negative delays are accepted; ``nan``, ``inf`` and negative
    numbers give ``None``. A date in the past means "now" (0 seconds).
    """
    if not value or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max((when - current).total_seconds(), 0.0)


def keyless_note(dropped_filters: list[str], requested: int | None) -> str:
    """The note added to keyless search results.

    ``requested`` is the result count the user asked for, or ``None`` when they
    did not ask for one; the cap is mentioned only when they asked for more.
    """
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
    retry_after = retry_after_seconds(response.headers.get("Retry-After"))
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


def _bad_reply(response: httpx.Response, detail: str) -> ClawifiKeylessError:
    return ClawifiKeylessError(
        f"Clawifi MCP server sent a reply that could not be read: {detail}",
        status=response.status_code,
    )


def _loads(text: str, response: httpx.Response) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise _bad_reply(response, f"invalid JSON ({exc})") from exc


def _sse_events(response: httpx.Response) -> tuple[list[dict[str, Any]], int]:
    """Decode the JSON events of an SSE reply.

    Returns the JSON-RPC messages and how many events could not be decoded.
    Undecodable events (a keep-alive ping, say) are skipped, not fatal.
    """
    messages: list[dict[str, Any]] = []
    unreadable = 0
    data: list[str] = []
    for line in [*response.text.splitlines(), ""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line == "" and data:
            try:
                messages.extend(_as_list(json.loads("\n".join(data))))
            except ValueError:
                unreadable += 1
            data = []
    return messages, unreadable


def _messages(response: httpx.Response) -> tuple[list[dict[str, Any]], int]:
    """Decode a JSON or SSE reply into JSON-RPC messages (plus unreadable SSE events)."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return _sse_events(response)
    if not response.content:
        return [], 0
    return _as_list(_loads(response.text, response)), 0


def _as_list(payload: Any) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else [payload]
    return [item for item in items if isinstance(item, dict)]


def _rpc_error(error: Any) -> ClawifiKeylessError:
    if isinstance(error, dict):
        return ClawifiKeylessError(
            f"Clawifi keyless call failed: {error.get('message') or error}",
            code=error.get("code"),
        )
    return ClawifiKeylessError(f"Clawifi keyless call failed: {error}")


def _result(response: httpx.Response, request_id: int) -> dict[str, Any]:
    messages, unreadable = _messages(response)
    for message in messages:
        if message.get("id") != request_id:
            continue
        if "error" in message:
            raise _rpc_error(message["error"])
        result = message.get("result")
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise _bad_reply(response, "the JSON-RPC result is not an object")
        return result
    if unreadable:
        raise _bad_reply(
            response, f"no JSON-RPC reply among its events ({unreadable} could not be decoded)"
        )
    raise ClawifiKeylessError("Clawifi MCP server sent no reply to the request.")


def _tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Turn a tools/call result into the tool's JSON object."""
    content = result.get("content")
    texts = [
        str(block.get("text", ""))
        for block in (content if isinstance(content, list) else [])
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
#
# An MCP session is opened once and reused for later calls. close()/aclose()
# end it with an HTTP DELETE, as the streamable HTTP transport specifies. If the
# server drops a session (HTTP 404), a new one is opened and the call retried once.


class _Session:
    """MCP session state shared by the sync and async clients."""

    def __init__(self) -> None:
        self.ids = itertools.count(1)
        self.session_id: str | None = None
        self.protocol: str | None = None
        self.ready = False

    def headers(self) -> dict[str, str]:
        return _headers(self.session_id, self.protocol)

    def opened(self, response: httpx.Response, result: dict[str, Any]) -> None:
        self.session_id = response.headers.get("mcp-session-id")
        self.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)

    def reset(self) -> None:
        self.session_id = None
        self.protocol = None
        self.ready = False


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
        self._lock = threading.Lock()
        self._session = _Session()

    @property
    def session_id(self) -> str | None:
        return self._session.session_id

    def __enter__(self) -> KeylessClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """End the MCP session (HTTP DELETE) and close the HTTP client."""
        with self._lock:
            if self._session.session_id and not self._http.is_closed:
                try:
                    self._http.delete(self.url, headers=self._session.headers())
                except httpx.HTTPError:
                    pass  # best effort: the server expires idle sessions too
            self._session.reset()
            self._http.close()

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._http.post(self.url, json=body, headers=self._session.headers())
        except httpx.HTTPError as exc:
            raise ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}") from exc

    def _initialize(self) -> None:
        self._session.reset()
        request_id = next(self._session.ids)
        response = self._post(_initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        self._session.opened(response, _result(response, request_id))
        notified = self._post(_INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)
        self._session.ready = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._session.ready:
                self._initialize()
            request_id = next(self._session.ids)
            response = self._post(_call_request(request_id, name, arguments))
            if response.status_code == 404 and self._session.session_id:
                # The server dropped the session: open a new one and retry once.
                self._initialize()
                request_id = next(self._session.ids)
                response = self._post(_call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))


class _LoopState:
    """The HTTP client, lock and MCP session of one event loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.loop = loop
        self.http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self.lock = asyncio.Lock()
        self.session = _Session()


class AsyncKeylessClient:
    """Asynchronous keyless MCP client.

    The HTTP client and MCP session belong to the event loop they were opened
    in and are reused by later calls in that loop. When a call comes from a new
    loop (a second ``asyncio.run``, say), a fresh HTTP client and session are
    opened there, and the previous session is ended with a DELETE from the new
    loop, so no session is left open on the server and nothing is used across
    loops. ``aclose()`` ends the current session.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = mcp_url(api_base)
        self._timeout = timeout
        self._transport = transport
        self._state: _LoopState | None = None

    @property
    def session_id(self) -> str | None:
        return self._state.session.session_id if self._state else None

    async def __aenter__(self) -> AsyncKeylessClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def _delete(self, http: httpx.AsyncClient, session: _Session) -> None:
        if not session.session_id:
            return
        try:
            await http.delete(self.url, headers=session.headers())
        except httpx.HTTPError:
            pass  # best effort: the server expires idle sessions too

    async def _close_http(self, http: httpx.AsyncClient) -> None:
        # A transport passed in by the caller belongs to the caller: leave it open.
        if self._transport is None:
            await http.aclose()

    async def aclose(self) -> None:
        """End the MCP session (HTTP DELETE) and close the HTTP client.

        Call it from the event loop that made the calls.
        """
        state, self._state = self._state, None
        if state is None:
            return
        if state.loop is asyncio.get_running_loop():
            await self._delete(state.http, state.session)
            await self._close_http(state.http)
        elif state.session.session_id:
            await self._end_orphan(state.session)

    async def _end_orphan(self, session: _Session) -> None:
        """End a session opened in a loop that is gone, from the current loop."""
        http = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        try:
            await self._delete(http, session)
        finally:
            await self._close_http(http)

    async def _current(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        state = self._state
        if state is not None and state.loop is loop:
            return state
        fresh = _LoopState(loop, self._timeout, self._transport)
        self._state = fresh
        if state is not None and state.session.session_id:
            # The old loop's client cannot be used here; end its session from this loop.
            await self._delete(fresh.http, state.session)
        return fresh

    async def _post(self, state: _LoopState, body: dict[str, Any]) -> httpx.Response:
        try:
            return await state.http.post(self.url, json=body, headers=state.session.headers())
        except httpx.HTTPError as exc:
            raise ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}") from exc

    async def _initialize(self, state: _LoopState) -> None:
        state.session.reset()
        request_id = next(state.session.ids)
        response = await self._post(state, _initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        state.session.opened(response, _result(response, request_id))
        notified = await self._post(state, _INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)
        state.session.ready = True

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = await self._current()
        async with state.lock:
            if not state.session.ready:
                await self._initialize(state)
            request_id = next(state.session.ids)
            response = await self._post(state, _call_request(request_id, name, arguments))
            if response.status_code == 404 and state.session.session_id:
                await self._initialize(state)
                request_id = next(state.session.ids)
                response = await self._post(state, _call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))


# -- building the other mode's REST client ---------------------------------------


class ClawifiClientConfigError(ClawifiError):
    """Only one of ``client`` / ``async_client`` was given and it cannot be mirrored."""


_DEFAULT_POOLS = ("ConnectionPool", "AsyncConnectionPool")


def _mount_signature(http: Any) -> list[tuple[str, str]]:
    signature = []
    for pattern, transport in (getattr(http, "_mounts", None) or {}).items():
        pool = getattr(transport, "_pool", None)
        proxy = getattr(pool, "_proxy_url", None)
        signature.append((str(getattr(pattern, "pattern", pattern)), repr(proxy)))
    return sorted(signature)


def http_settings_copyable(sdk_client: Any) -> bool:
    """True when the SDK client's HTTP layer is the default one (no custom transport or proxy).

    Only then can the client for the other mode (sync/async) be rebuilt from
    ``.config`` without losing anything. Environment proxies count as default,
    since a new client picks them up too.
    """
    http = getattr(sdk_client, "_client", None)
    transport = getattr(http, "_transport", None)
    if http is None or transport is None or not hasattr(http, "_mounts"):
        return False
    if type(transport) not in (httpx.HTTPTransport, httpx.AsyncHTTPTransport):
        return False
    if type(getattr(transport, "_pool", None)).__name__ not in _DEFAULT_POOLS:
        return False
    reference = httpx.Client(trust_env=getattr(http, "trust_env", True))
    try:
        return _mount_signature(http) == _mount_signature(reference)
    finally:
        reference.close()


def require_copyable(sdk_client: Any, given: str, missing: str) -> None:
    if not http_settings_copyable(sdk_client):
        raise ClawifiClientConfigError(
            f"Only {given} was given, and its HTTP settings (a custom transport or proxy) "
            f"cannot be copied to {missing}. Pass {missing}= as well."
        )


def search_arguments(query: str) -> dict[str, Any]:
    return {"query": query}


def fetch_arguments(url: str, query: str | None) -> dict[str, Any]:
    return {"url": url, "query": query} if query else {"url": url}
