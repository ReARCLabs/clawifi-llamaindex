"""Clawifi tool spec for LlamaIndex: web search and page reading."""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from clawifi import AsyncClawifi, Clawifi

DEFAULT_MAX_CHARS = 20_000


class ClawifiToolSpec(BaseToolSpec):
    """Clawifi web search and page reading.

    ``search`` calls ``POST /search/free`` and ``fetch`` calls ``POST /fetch`` through
    the official ``clawifi`` SDK. The API key comes from ``api_key`` or
    ``CLAWIFI_API_KEY``; without one, calls go out keyless on Clawifi's free trial
    limits.
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
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base
        self.max_results = max_results
        self.max_chars = max_chars
        self._client = client
        self._async_client = async_client

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

    # -- search -------------------------------------------------------------

    def _search_body(self, max_results: Optional[int], time_range: Optional[str]) -> dict:
        body: dict[str, Any] = {"max_results": max_results or self.max_results}
        if time_range:
            body["time_range"] = time_range
        return body

    @staticmethod
    def _search_docs(response: dict) -> List[Document]:
        if not response.get("success", True) and not response.get("results"):
            raise RuntimeError(f"Clawifi search failed: {response.get('errors') or []}")
        docs = []
        for item in response.get("results") or []:
            metadata = {
                k: item[k]
                for k in ("url", "title", "rank", "published_at")
                if item.get(k) is not None
            }
            docs.append(
                Document(text=item.get("snippet") or item.get("title") or "", metadata=metadata)
            )
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
        return self._search_docs(
            self.client.search_free(query, **self._search_body(max_results, time_range))
        )

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
        response = await self.async_client.search_free(
            query, **self._search_body(max_results, time_range)
        )
        return self._search_docs(response)

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
        return self._fetch_doc(self.client.fetch(url, **({"query": query} if query else {})))

    async def afetch(self, url: str, query: Optional[str] = None) -> List[Document]:
        """
        Read a web page or PDF and return its main content as Markdown.

        Args:
            url: The URL of the page or PDF to read.
            query: Optional question; when set, only the passages that answer it are returned.

        """
        response = await self.async_client.fetch(url, **({"query": query} if query else {}))
        return self._fetch_doc(response)
