"""Clawifi tool spec for LlamaIndex: web search and page reading."""

from __future__ import annotations

import os
from typing import Any, List, Optional

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
    search_arguments,
)

DEFAULT_MAX_CHARS = 20_000


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
        max_results: int = 5,
        max_chars: int = DEFAULT_MAX_CHARS,
        api_base: Optional[str] = None,
        client: Optional[Clawifi] = None,
        async_client: Optional[AsyncClawifi] = None,
        keyless_client: Optional[KeylessClient] = None,
        async_keyless_client: Optional[AsyncKeylessClient] = None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self.max_results = max_results
        self.max_chars = max_chars
        self._client = client
        self._async_client = async_client
        self._keyless_client = keyless_client
        self._async_keyless_client = async_keyless_client

    @property
    def client(self) -> Clawifi:
        if self._client is None:
            self._client = Clawifi(api_key=self._api_key, api_base=self._api_base)
        return self._client

    @property
    def async_client(self) -> AsyncClawifi:
        if self._async_client is None:
            self._async_client = AsyncClawifi(api_key=self._api_key, api_base=self._api_base)
        return self._async_client

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

    def _has_key(self) -> bool:
        return bool(self._api_key or (os.getenv("CLAWIFI_API_KEY") or "").strip())

    def _keyless(self, *, is_async: bool) -> bool:
        prebuilt = self._async_client if is_async else self._client
        return prebuilt is None and not self._has_key()

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

    def _keyless_docs(self, response: dict, body: dict) -> List[Document]:
        requested = body.get("max_results")
        limit = min(requested or KEYLESS_MAX_RESULTS, KEYLESS_MAX_RESULTS)
        dropped = ["time_range"] if body.get("time_range") else []
        note = keyless_note(dropped, requested)
        return self._search_docs(response, note)[:limit]

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
        if self._keyless(is_async=False):
            response = self.keyless_client.call_tool(KEYLESS_SEARCH_TOOL, search_arguments(query))
            return self._keyless_docs(response, body)
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
        if self._keyless(is_async=True):
            response = await self.async_keyless_client.call_tool(
                KEYLESS_SEARCH_TOOL, search_arguments(query)
            )
            return self._keyless_docs(response, body)
        return self._search_docs(await self.async_client.search_free(query, **body))

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
        if self._keyless(is_async=False):
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
        if self._keyless(is_async=True):
            response = await self.async_keyless_client.call_tool(
                KEYLESS_FETCH_TOOL, fetch_arguments(url, query)
            )
        else:
            response = await self.async_client.fetch(url, **({"query": query} if query else {}))
        return self._fetch_doc(response)
