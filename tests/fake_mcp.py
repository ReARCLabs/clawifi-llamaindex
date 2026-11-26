"""A fake of Clawifi's hosted MCP server for keyless unit tests (no network).

By default it behaves like the live server: stateless, JSON replies, and a
``tools/call`` is answered without any initialization. With ``require_init`` it
behaves like a stateful server that refuses calls until the client initializes,
in one of three styles (``init_error``):

- ``"rpc-code"``: JSON-RPC error -32002 "Server not initialized";
- ``"rpc-message"``: JSON-RPC error -32600 "Bad Request: Server not initialized";
- ``"http400"``: HTTP 400 "Bad Request: Missing session ID".
"""

from __future__ import annotations

import json
from typing import Any

import httpx

SEARCH_PAYLOAD = {
    "success": True,
    "query": "mcp",
    "provider": "free_auto",
    "results": [
        {"url": f"https://example.com/{i}", "title": f"T{i}", "snippet": f"s{i}", "rank": i}
        for i in range(1, 6)
    ],
    "errors": [],
}
FETCH_PAYLOAD = {
    "success": True,
    "content_status": "ok",
    "errors": [],
    "data": {"url": "https://example.com/", "title": "Example Domain", "content_markdown": "Body"},
}


class FakeMcpServer:
    def __init__(
        self,
        *,
        sse: bool = False,
        session_id: str | None = "sess-1",
        structured: bool = False,
        tool_payloads: dict[str, dict[str, Any]] | None = None,
        require_init: bool = False,
        init_error: str = "rpc-code",
    ) -> None:
        self.sse = sse
        self.session_id = session_id  # prefix of the IDs handed out on initialize
        self.structured = structured
        self.tool_payloads = tool_payloads or {
            "search_free": SEARCH_PAYLOAD,
            "fetch_url": FETCH_PAYLOAD,
        }
        self.require_init = require_init
        self.init_error = init_error
        self.requests: list[tuple[httpx.Request, dict[str, Any]]] = []
        self.fail_next: httpx.Response | None = None
        self.tool_error: str | None = None
        self.rpc_error: dict[str, Any] | None = None
        self.sessions_opened = 0
        self.deleted_sessions: list[str | None] = []
        self.delete_requests: list[httpx.Request] = []
        self.active: set[str] = set()  # sessions opened and not yet ended
        self.initialized = False  # for a stateful server without session IDs
        self.sse_ping = False

    def methods(self) -> list[str]:
        return [body.get("method") for _, body in self.requests]

    def open_sessions(self) -> int:
        """Sessions opened and not yet ended with DELETE."""
        return len(self.active)

    def calls(self) -> list[dict[str, Any]]:
        return [body["params"] for _, body in self.requests if body.get("method") == "tools/call"]

    def _reply(
        self, message: dict[str, Any], headers: dict[str, str] | None = None
    ) -> httpx.Response:
        headers = dict(headers or {})
        if self.sse:
            note = {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info"},
            }
            text = "".join(
                f"event: message\ndata: {json.dumps(item)}\n\n" for item in (note, message)
            )
            if self.sse_ping:
                text = "event: ping\ndata: ping\n\n" + text + ": keep-alive\n\n"
            headers["content-type"] = "text/event-stream"
            return httpx.Response(200, text=text, headers=headers)
        return httpx.Response(200, json=message, headers=headers)

    def _not_initialized(self, body: dict[str, Any]) -> httpx.Response:
        if self.init_error == "http400":
            return httpx.Response(400, text="Bad Request: Missing session ID")
        code, message = (
            (-32002, "Server not initialized")
            if self.init_error == "rpc-code"
            else (-32600, "Bad Request: Server not initialized")
        )
        return self._reply(
            {"jsonrpc": "2.0", "id": body["id"], "error": {"code": code, "message": message}}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mcp"
        assert "authorization" not in request.headers
        if request.method == "DELETE":
            session = request.headers.get("mcp-session-id")
            self.delete_requests.append(request)
            self.deleted_sessions.append(session)
            self.active.discard(session or "")
            return httpx.Response(204)
        assert "text/event-stream" in request.headers["accept"]
        assert "application/json" in request.headers["accept"]
        body = json.loads(request.content)
        self.requests.append((request, body))
        method = body.get("method")
        if method == "initialize":
            self.sessions_opened += 1
            self.initialized = True
            headers = {}
            if self.session_id:
                new_session = f"{self.session_id}-{self.sessions_opened}"
                self.active.add(new_session)
                headers = {"mcp-session-id": new_session}
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "clawifi"},
            }
            return self._reply({"jsonrpc": "2.0", "id": body["id"], "result": result}, headers)
        assert request.headers.get("mcp-protocol-version") == "2025-06-18"
        sent_session = request.headers.get("mcp-session-id")
        if sent_session is not None:
            assert sent_session in self.active
        if method == "notifications/initialized":
            return httpx.Response(202)
        assert method == "tools/call"
        if self.require_init:
            ready = sent_session in self.active if self.session_id else self.initialized
            if not ready:
                return self._not_initialized(body)
        if self.fail_next is not None:
            response, self.fail_next = self.fail_next, None
            return response
        if self.rpc_error is not None:
            return self._reply({"jsonrpc": "2.0", "id": body["id"], "error": self.rpc_error})
        if self.tool_error is not None:
            result = {"content": [{"type": "text", "text": self.tool_error}], "isError": True}
            return self._reply({"jsonrpc": "2.0", "id": body["id"], "result": result})
        payload = self.tool_payloads[body["params"]["name"]]
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "isError": False,
        }
        if self.structured:
            result = {
                "content": [{"type": "text", "text": "see structured"}],
                "structuredContent": payload,
            }
        return self._reply({"jsonrpc": "2.0", "id": body["id"], "result": result})
