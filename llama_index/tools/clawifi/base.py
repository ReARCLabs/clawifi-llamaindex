"""Clawifi tool spec for LlamaIndex: web search and page reading."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from clawifi import AsyncClawifi, Clawifi
from llama_index.tools.clawifi._keyless import (
    KEYLESS_FETCH_TOOL,
    KEYLESS_MAX_RESULTS,
    KEYLESS_SEARCH_TOOL,
    AsyncKeylessClient,
    KeylessClient,
    fetch_arguments,
    keyless_note,
    missing_client_error,
    search_arguments,
)

DEFAULT_MAX_CHARS = 20_000
DEFAULT_MAX_RESULTS = 5


class ClawifiToolSpec(BaseToolSpec):
    """Clawifi web search and page reading.

    With an API key (``api_key`` or ``CLAWIFI_API_KEY``), ``search`` calls
    ``POST /search/free`` and ``fetch`` calls ``POST /fetch`` through the official
    ``clawifi`` SDK. Without a key they call the keyless trial tools
    (``search_free``, ``fetch_url``) of Clawifi's hosted MCP server: at most 3
    search results, no time filter, 12 calls a minute and 50 a day per address.
    """

    spec_functions = [("search", "asearch"), ("fetch", "afetch")]

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        max_results: Optional[int] = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        api_base: Optional[str] = None,
        client: Optional[Clawifi] = None,
        async_client: Optional[AsyncClawifi] = None,
        keyless_client: Optional[KeylessClient] = None,
        async_keyless_client: Optional[AsyncKeylessClient] = None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        # None means "not set by the user": the default of 5 is used, and a keyless
        # note never claims the user asked for more than the trial's 3 results.
        self._max_results: Optional[int] = max_results
        self.max_chars = max_chars
        # Clients the user passed in. They alone decide REST over keyless.
        self._client = client
        self._async_client = async_client
        # A sync client this spec built itself (lazily, on first use).
        self._sync: Optional[Clawifi] = None
        self._keyless_client = keyless_client
        self._async_keyless_client = async_keyless_client

    @property
    def max_results(self) -> int:
        """Default result count; 5 unless set. Setting it counts as the user's request."""
        return self._max_results if self._max_results is not None else DEFAULT_MAX_RESULTS

    @max_results.setter
    def max_results(self, value: Optional[int]) -> None:
        self._max_results = value

    @property
    def client(self) -> Clawifi:
        """The sync REST client: the one passed in, else one built (and kept) by the spec.

        Reading it never switches the spec from keyless to REST.
        """
        if self._client is not None:
            return self._client
        if self._sync is None:
            if self._async_client is not None:
                raise missing_client_error("async_client", "client")
            self._sync = Clawifi(api_key=self._api_key, api_base=self._api_base)
        return self._sync

    @asynccontextmanager
    async def _async_rest(self) -> AsyncIterator[AsyncClawifi]:
        """Yield an async REST client bound to the running event loop.

        A client passed as ``async_client`` is used as is. Otherwise a new one is
        opened for this call and closed after it, so nothing stays bound to an
        event loop that ``asyncio.run`` has already closed.
        """
        if self._async_client is not None:
            yield self._async_client
            return
        if self._client is not None:
            raise missing_client_error("client", "async_client")
        client = AsyncClawifi(api_key=self._api_key, api_base=self._api_base)
        try:
            yield client
        finally:
            await client.aclose()

    @property
    def keyless_client(self) -> KeylessClient:
        if self._keyless_client is None:
            self._keyless_client = KeylessClient(api_base=self._api_base)
        return self._keyless_client

    @property
    def async_keyless_client(self) -> AsyncKeylessClient:
        if self._async_keyless_client is None:
            self._async_keyless_client = AsyncKeylessClient(api_base=self._api_base)
        return self._async_keyless_client

    def close(self) -> None:
        """End the keyless MCP session (if any) and close the clients this spec built."""
        if self._keyless_client is not None:
            self._keyless_client.close()
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        """End the async keyless MCP session (if any), from any event loop."""
        if self._async_keyless_client is not None:
            await self._async_keyless_client.aclose()

    def _has_key(self) -> bool:
        return bool(self._api_key or (os.getenv("CLAWIFI_API_KEY") or "").strip())

    def _keyless(self) -> bool:
        """Use the keyless trial only with no key and no pre-built REST client at all."""
        return self._client is None and self._async_client is None and not self._has_key()

    def _requested(self, max_results: Optional[int]) -> Optional[int]:
        """The result count the user asked for: per call, or set on the spec; else None."""
        if max_results is not None:
            return max_results
        return self._max_results

    # -- search -------------------------------------------------------------

    def _search_body(self, max_results: Optional[int], time_range: Optional[str]) -> dict:
        body: dict[str, Any] = {"max_results": max_results or self.max_results}
        if time_range:
            body["time_range"] = time_range
        return body

    @staticmethod
    def _search_docs(response: dict, note: Optional[str] = None) -> List[Document]:
        if not response.get("success", True) and not response.get("results"):
            raise RuntimeError(f"Clawifi search failed: {response.get('errors') or []}")
        docs = []
        for item in response.get("results") or []:
            if not isinstance(item, dict):
                continue
            metadata = {
                k: item[k]
                for k in ("url", "title", "rank", "published_at")
                if item.get(k) is not None
            }
            if note:
                metadata["keyless_note"] = note
            docs.append(
                Document(text=item.get("snippet") or item.get("title") or "", metadata=metadata)
            )
        return docs

    def _keyless_docs(self, response: dict, body: dict, requested: Optional[int]) -> List[Document]:
        limit = min(requested or KEYLESS_MAX_RESULTS, KEYLESS_MAX_RESULTS)
        dropped = ["time_range"] if body.get("time_range") else []
        note = keyless_note(dropped, requested)
        docs = self._search_docs(response, note)[:limit]
        if not docs:
            # Keep the note even when the trial search finds nothing.
            return [Document(text=note, metadata={"keyless_note": note, "result_count": 0})]
        return docs

    def search(
        self, query: str, max_results: Optional[int] = None, time_range: Optional[str] = None
    ) -> List[Document]:
        """
        Search the web and return results with title, url and snippet.

        Args:
            query: The search query.
            max_results: How many results to return.
            time_range: Optional recent period: "day", "week", "month" or "year".

        """
        body = self._search_body(max_results, time_range)
        if self._keyless():
            response = self.keyless_client.call_tool(KEYLESS_SEARCH_TOOL, search_arguments(query))
            return self._keyless_docs(response, body, self._requested(max_results))
        return self._search_docs(self.client.search_free(query, **body))

    async def asearch(
        self, query: str, max_results: Optional[int] = None, time_range: Optional[str] = None
    ) -> List[Document]:
        """
        Search the web and return results with title, url and snippet.

        Args:
            query: The search query.
            max_results: How many results to return.
            time_range: Optional recent period: "day", "week", "month" or "year".

        """
        body = self._search_body(max_results, time_range)
        if self._keyless():
            response = await self.async_keyless_client.call_tool(
                KEYLESS_SEARCH_TOOL, search_arguments(query)
            )
            return self._keyless_docs(response, body, self._requested(max_results))
        async with self._async_rest() as client:
            return self._search_docs(await client.search_free(query, **body))

    # -- fetch --------------------------------------------------------------

    def _fetch_doc(self, response: dict) -> List[Document]:
        if not response.get("success", True):
            raise RuntimeError(
                f"Clawifi could not read the page ({response.get('content_status')}): "
                f"{response.get('errors') or []}"
            )
        data = response.get("data") or {}
        content = data.get("content_markdown") or ""
        metadata = {
            "url": data.get("url") or data.get("source_url"),
            "title": data.get("title"),
            "content_status": response.get("content_status"),
        }
        if len(content) > self.max_chars:
            metadata["truncated"] = True
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return [Document(text=content[: self.max_chars], metadata=metadata)]

    def fetch(self, url: str, query: Optional[str] = None) -> List[Document]:
        """
        Read a web page or PDF and return its main content as Markdown.

        Args:
            url: The URL of the page or PDF to read.
            query: Optional question; when set, only the passages that answer it are returned.

        """
        if self._keyless():
            response = self.keyless_client.call_tool(
                KEYLESS_FETCH_TOOL, fetch_arguments(url, query)
            )
        else:
            response = self.client.fetch(url, **({"query": query} if query else {}))
        return self._fetch_doc(response)

    async def afetch(self, url: str, query: Optional[str] = None) -> List[Document]:
        """
        Read a web page or PDF and return its main content as Markdown.

        Args:
            url: The URL of the page or PDF to read.
            query: Optional question; when set, only the passages that answer it are returned.

        """
        if self._keyless():
            response = await self.async_keyless_client.call_tool(
                KEYLESS_FETCH_TOOL, fetch_arguments(url, query)
            )
        else:
            async with self._async_rest() as client:
                response = await client.fetch(url, **({"query": query} if query else {}))
        return self._fetch_doc(response)
