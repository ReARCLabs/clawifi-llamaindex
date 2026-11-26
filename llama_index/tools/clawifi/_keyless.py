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
# The MCP session ID is a plain string, not tied to any connection or event loop.
# Both clients keep it on the object and reuse it; if the server no longer knows
# it (HTTP 404), they initialize once more and retry. A transport passed in by the
# caller is never closed by these clients.

_TRANSPORT_ERRORS = (httpx.HTTPError, RuntimeError)


class _Session:
    """MCP session state: the session ID, the negotiated protocol and request IDs."""

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
        self.ready = True

    def reset(self) -> None:
        self.session_id = None
        self.protocol = None
        self.ready = False


def _unreachable(exc: BaseException) -> ClawifiKeylessError:
    return ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}")


class KeylessClient:
    """Synchronous keyless MCP client.

    The HTTP client is built on first use. ``close()`` ends the MCP session (HTTP
    DELETE, errors ignored) and drops the HTTP client; the next call builds a new
    one. A ``transport`` passed in belongs to the caller and is never closed.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = mcp_url(api_base)
        self._timeout = timeout
        self._transport = transport
        self._http: httpx.Client | None = None
        self._lock = threading.Lock()
        self._session = _Session()

    @property
    def session_id(self) -> str | None:
        return self._session.session_id

    def __enter__(self) -> KeylessClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self._timeout, transport=self._transport)
        return self._http

    def close(self) -> None:
        """End the MCP session (HTTP DELETE, errors ignored) and drop the HTTP client."""
        with self._lock:
            if self._session.session_id:
                try:
                    self._client().delete(self.url, headers=self._session.headers())
                except _TRANSPORT_ERRORS:
                    pass  # best effort: the server expires idle sessions too
            self._session.reset()
            http, self._http = self._http, None
            if http is not None and self._transport is None:  # the caller's transport stays open
                http.close()

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client().post(self.url, json=body, headers=self._session.headers())
        except _TRANSPORT_ERRORS as exc:
            raise _unreachable(exc) from exc

    def _initialize(self) -> None:
        self._session.reset()
        request_id = next(self._session.ids)
        response = self._post(_initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        self._session.opened(response, _result(response, request_id))
        notified = self._post(_INITIALIZED)
        if notified.status_code >= 400:
            self._session.reset()
            raise _http_error(notified)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._session.ready:
                self._initialize()
            request_id = next(self._session.ids)
            response = self._post(_call_request(request_id, name, arguments))
            if response.status_code == 404 and self._session.session_id:
                # The server no longer knows the session: initialize once more and retry.
                self._initialize()
                request_id = next(self._session.ids)
                response = self._post(_call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))


class AsyncKeylessClient:
    """Asynchronous keyless MCP client.

    Every call opens a short-lived ``httpx.AsyncClient`` in the running event
    loop, so nothing is ever used across loops (repeated ``asyncio.run`` works).
    The MCP session ID is kept on the object and reused by later calls, whatever
    loop they run in. Two concurrent first calls may each open a session; the
    last one to finish is kept. ``aclose()`` ends the session with an HTTP DELETE
    (errors ignored). A ``transport`` passed in belongs to the caller and is
    never closed.
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
        self._session = _Session()

    @property
    def session_id(self) -> str | None:
        return self._session.session_id

    async def __aenter__(self) -> AsyncKeylessClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def _with_client(self, work: Any) -> Any:
        http = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        try:
            return await work(http)
        finally:
            if self._transport is None:  # the caller's transport stays open
                await http.aclose()

    async def aclose(self) -> None:
        """End the MCP session with an HTTP DELETE (errors ignored)."""
        session_id = self._session.session_id
        headers = self._session.headers()
        self._session.reset()
        if not session_id:
            return

        async def delete(http: httpx.AsyncClient) -> None:
            try:
                await http.delete(self.url, headers=headers)
            except _TRANSPORT_ERRORS:
                pass  # best effort: the server expires idle sessions too

        await self._with_client(delete)

    async def _post(self, http: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
        try:
            return await http.post(self.url, json=body, headers=self._session.headers())
        except _TRANSPORT_ERRORS as exc:
            raise _unreachable(exc) from exc

    async def _initialize(self, http: httpx.AsyncClient) -> None:
        self._session.reset()
        request_id = next(self._session.ids)
        response = await self._post(http, _initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        self._session.opened(response, _result(response, request_id))
        notified = await self._post(http, _INITIALIZED)
        if notified.status_code >= 400:
            self._session.reset()
            raise _http_error(notified)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async def call(http: httpx.AsyncClient) -> dict[str, Any]:
            if not self._session.ready:
                await self._initialize(http)
            request_id = next(self._session.ids)
            response = await self._post(http, _call_request(request_id, name, arguments))
            if response.status_code == 404 and self._session.session_id:
                await self._initialize(http)
                request_id = next(self._session.ids)
                response = await self._post(http, _call_request(request_id, name, arguments))
            if response.status_code >= 400:
                raise _http_error(response)
            return _tool_payload(_result(response, request_id))

        return await self._with_client(call)


# -- one given client, the other mode called -------------------------------------


class ClawifiClientConfigError(ClawifiError):
    """Only one of ``client`` / ``async_client`` was given and the other mode was called."""


def missing_client_error(given: str, missing: str) -> ClawifiClientConfigError:
    """The error for a call in the mode whose client was not given."""
    mode = "async" if missing == "async_client" else "sync"
    return ClawifiClientConfigError(
        f"Only {given} was given, so there is no client for this {mode} call. "
        f"Pass {missing} too, or pass api_key instead of clients."
    )


def search_arguments(query: str) -> dict[str, Any]:
    return {"query": query}


def fetch_arguments(url: str, query: str | None) -> dict[str, Any]:
    return {"url": url, "query": query} if query else {"url": url}
