"""Keyless trial calls through Clawifi's hosted MCP server.

Clawifi's REST API needs a key. Without one, the hosted MCP server at
``https://api.clawifi.dev/mcp`` still answers a few trial tools, among them
``search_free`` and ``fetch_url`` (12 calls a minute and 50 a day per address,
at most 3 search results, no domain or time filters). The hosted server is
stateless, so each call is one ``tools/call`` POST over streamable HTTP; the
reply may be JSON or SSE (``text/event-stream``). See the clients section for
the initialization fallback and for which errors become
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


def _find_message(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """The JSON-RPC message that answers ``request_id``.

    A server that cannot tell which request failed answers with an error whose
    ``id`` is null; when nothing matches ``request_id``, that error is the answer.
    """
    messages, unreadable = _messages(response)
    for message in messages:
        if message.get("id") == request_id:
            return message
    for message in messages:
        if "id" in message and message["id"] is None and "error" in message:
            return message
    if unreadable:
        raise _bad_reply(
            response, f"no JSON-RPC reply among its events ({unreadable} could not be decoded)"
        )
    raise ClawifiKeylessError("Clawifi MCP server sent no reply to the request.")


def _message_result(response: httpx.Response, message: dict[str, Any]) -> dict[str, Any]:
    if "error" in message:
        raise _rpc_error(message["error"])
    result = message.get("result")
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise _bad_reply(response, "the JSON-RPC result is not an object")
    return result


def _result(response: httpx.Response, request_id: int) -> dict[str, Any]:
    return _message_result(response, _find_message(response, request_id))


# A stateless server answers tools/call directly. A stateful one refuses it until
# the client initializes: with JSON-RPC error -32002, or with a message that
# says so explicitly ("not initialized", "missing session", "Mcp-Session-Id"),
# sometimes as HTTP 400. Any other error goes to the caller as it is.
_INIT_HINTS = ("not initialized", "missing session", "mcp-session-id")
_NOT_INITIALIZED_CODE = -32002


def _asks_for_initialize(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _INIT_HINTS)


def _rpc_asks_for_initialize(error: Any) -> bool:
    if isinstance(error, dict):
        if error.get("code") == _NOT_INITIALIZED_CODE:
            return True
        return _asks_for_initialize(str(error.get("message") or ""))
    return _asks_for_initialize(str(error))


def _call_outcome(response: httpx.Response, request_id: int) -> dict[str, Any] | None:
    """The tool's JSON result, or None when the server asks for initialization first."""
    if response.status_code == 400 and _asks_for_initialize(response.text):
        return None
    if response.status_code >= 400:
        raise _http_error(response)
    message = _find_message(response, request_id)
    if "error" in message and _rpc_asks_for_initialize(message["error"]):
        return None
    return _tool_payload(_message_result(response, message))


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
# Clawifi's hosted MCP server is stateless and answers in JSON, so a keyless
# call is a single POST of ``tools/call`` (the trial limits count tool calls).
# Only if a server explicitly asks for initialization does the call fall back to
# initialize -> notifications/initialized -> tools/call; if that server returned
# an ``Mcp-Session-Id``, the session is ended with a short DELETE afterwards
# (skipped on cancellation, errors ignored).
#
# Wrapped as ClawifiKeylessError: HTTP status errors, bad replies, and while
# sending a request any httpx.HTTPError, httpx.InvalidURL or httpx.StreamError,
# plus httpx's "client has been closed" RuntimeError. Other errors propagate.

_HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError)
DELETE_TIMEOUT = 2.0

# One module-level lock guards the (rare) lazy creation of HTTP clients. Keeping
# no lock on the instances keeps them copyable and picklable.
CLIENT_LOCK = threading.Lock()


def _is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, _HTTPX_ERRORS):
        return True
    return isinstance(exc, RuntimeError) and "client has been closed" in str(exc)


def _unreachable(exc: BaseException) -> ClawifiKeylessError:
    return ClawifiKeylessError(f"Clawifi MCP server unreachable: {exc}")


class _BorrowedTransport(httpx.BaseTransport):
    """Wraps a caller's transport so closing our client never closes theirs."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._inner.handle_request(request)

    def close(self) -> None:
        pass  # the caller owns the transport


class _BorrowedAsyncTransport(httpx.AsyncBaseTransport):
    """Wraps a caller's async transport so closing our client never closes theirs."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        pass  # the caller owns the transport


class _Session:
    """One call's MCP state: request IDs, the protocol version and any session ID."""

    def __init__(self) -> None:
        self.ids = itertools.count(1)
        self.session_id: str | None = None
        self.protocol: str = PROTOCOL_VERSION

    def headers(self) -> dict[str, str]:
        return _headers(self.session_id, self.protocol)

    def opened(self, response: httpx.Response, result: dict[str, Any]) -> None:
        self.session_id = response.headers.get("mcp-session-id")
        self.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)


_STILL_NOT_INITIALIZED = "Clawifi MCP server still asks for initialization after initialize."


class KeylessClient:
    """Synchronous keyless MCP client: one ``tools/call`` POST per call.

    One ``httpx.Client`` is built lazily (under a module-level lock) and reused;
    ``close()`` closes it and the next call builds a new one. Copies and pickles
    of the client leave the HTTP client out and build their own. A ``transport`` passed in
    belongs to the caller and is never closed.
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
        self._transport = _BorrowedTransport(transport) if transport is not None else None
        self._http: httpx.Client | None = None

    def __getstate__(self) -> dict[str, Any]:
        # The HTTP client is not copyable; a copy builds its own on first use.
        state = self.__dict__.copy()
        state["_http"] = None
        return state

    def __enter__(self) -> KeylessClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client; the next call builds a new one."""
        with CLIENT_LOCK:
            http, self._http = self._http, None
        if http is not None:
            http.close()

    def _client(self) -> httpx.Client:
        with CLIENT_LOCK:
            if self._http is None or self._http.is_closed:
                self._http = httpx.Client(timeout=self._timeout, transport=self._transport)
            return self._http

    def _post(self, http: httpx.Client, session: _Session, body: dict[str, Any]) -> httpx.Response:
        try:
            return http.post(self.url, json=body, headers=session.headers())
        except Exception as exc:
            if _is_transport_error(exc):
                raise _unreachable(exc) from exc
            raise

    def _send_call(
        self, http: httpx.Client, session: _Session, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        request_id = next(session.ids)
        response = self._post(http, session, _call_request(request_id, name, arguments))
        return _call_outcome(response, request_id)

    def _initialize(self, http: httpx.Client, session: _Session) -> None:
        request_id = next(session.ids)
        response = self._post(http, session, _initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        session.opened(response, _result(response, request_id))
        notified = self._post(http, session, _INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)

    def _end(self, http: httpx.Client, session: _Session) -> None:
        if not session.session_id:
            return
        try:
            http.delete(self.url, headers=session.headers(), timeout=DELETE_TIMEOUT)
        except Exception:  # noqa: BLE001 - best-effort cleanup, never hides the result
            pass

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            http = self._client()
        except Exception as exc:
            if _is_transport_error(exc):
                raise _unreachable(exc) from exc
            raise
        payload = self._send_call(http, _Session(), name, arguments)
        if payload is not None:
            return payload
        # The server asked for initialization: do it once, then call again.
        session = _Session()
        try:
            self._initialize(http, session)
            payload = self._send_call(http, session, name, arguments)
            if payload is None:
                raise ClawifiKeylessError(_STILL_NOT_INITIALIZED)
            return payload
        finally:
            self._end(http, session)


class AsyncKeylessClient:
    """Asynchronous keyless MCP client: one ``tools/call`` POST per call.

    Each call opens its own ``httpx.AsyncClient`` and closes it before
    returning, so concurrent calls and repeated ``asyncio.run`` are safe. An
    error while closing never hides an error raised by the call itself. A
    ``transport`` passed in belongs to the caller and is never closed.
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
        self._transport = _BorrowedAsyncTransport(transport) if transport is not None else None

    async def __aenter__(self) -> AsyncKeylessClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Nothing to close: every call closes its own client."""

    async def _post(
        self, http: httpx.AsyncClient, session: _Session, body: dict[str, Any]
    ) -> httpx.Response:
        try:
            return await http.post(self.url, json=body, headers=session.headers())
        except Exception as exc:
            if _is_transport_error(exc):
                raise _unreachable(exc) from exc
            raise

    async def _send_call(
        self, http: httpx.AsyncClient, session: _Session, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        request_id = next(session.ids)
        response = await self._post(http, session, _call_request(request_id, name, arguments))
        return _call_outcome(response, request_id)

    async def _initialize(self, http: httpx.AsyncClient, session: _Session) -> None:
        request_id = next(session.ids)
        response = await self._post(http, session, _initialize_request(request_id))
        if response.status_code >= 400:
            raise _http_error(response)
        session.opened(response, _result(response, request_id))
        notified = await self._post(http, session, _INITIALIZED)
        if notified.status_code >= 400:
            raise _http_error(notified)

    async def _end(self, http: httpx.AsyncClient, session: _Session) -> None:
        if not session.session_id:
            return
        try:
            await http.delete(self.url, headers=session.headers(), timeout=DELETE_TIMEOUT)
        except Exception:  # noqa: BLE001 - best-effort cleanup, never hides the result
            pass

    async def _call(
        self, http: httpx.AsyncClient, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        payload = await self._send_call(http, _Session(), name, arguments)
        if payload is not None:
            return payload
        # The server asked for initialization: do it once, then call again.
        session = _Session()
        try:
            await self._initialize(http, session)
            payload = await self._send_call(http, session, name, arguments)
            if payload is None:
                raise ClawifiKeylessError(_STILL_NOT_INITIALIZED)
        except Exception:
            await self._end(http, session)  # errors in _end are swallowed
            raise
        # A cancellation (BaseException) skips the DELETE above and below.
        await self._end(http, session)
        return payload

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        http = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        try:
            payload = await self._call(http, name, arguments)
        except BaseException:
            try:
                await http.aclose()
            except Exception:  # noqa: BLE001 - never hide the call's own error
                pass
            raise
        try:
            await http.aclose()
        except Exception as exc:
            if _is_transport_error(exc):
                raise _unreachable(exc) from exc
            raise
        return payload


# -- API keys and the REST clients ------------------------------------------------


def clean_key(value: str | None) -> str | None:
    """A usable API key, or None: a key of only whitespace counts as no key."""
    if value is None:
        return None
    return value.strip() or None


def resolve_key(explicit: str | None) -> str | None:
    """The API key to use: the explicit one, else ``CLAWIFI_API_KEY``; blank means none."""
    return clean_key(explicit) or clean_key(os.getenv("CLAWIFI_API_KEY"))


class ClawifiClientConfigError(ClawifiError):
    """No key, only one of ``client`` / ``async_client`` given, and the other mode called."""


def missing_client_error(given: str, missing: str) -> ClawifiClientConfigError:
    """The error for a call in the mode whose client was not given."""
    mode = "async" if missing == "async_client" else "sync"
    return ClawifiClientConfigError(
        f"Only {given} was given and there is no API key, so there is no client for this "
        f"{mode} call. Pass {missing} too, or set an API key."
    )


def search_arguments(query: str) -> dict[str, Any]:
    return {"query": query}


def fetch_arguments(url: str, query: str | None) -> dict[str, Any]:
    return {"url": url, "query": query} if query else {"url": url}
