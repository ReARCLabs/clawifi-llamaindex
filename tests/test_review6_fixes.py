"""Regression tests for the sixth review round (no network)."""

from __future__ import annotations

import asyncio
import copy
import pickle

import httpx
import pytest

from llama_index.tools.clawifi import ClawifiKeylessError, ClawifiToolSpec
from llama_index.tools.clawifi._keyless import KeylessClient
from tests.conftest import Router
from tests.fake_mcp import FakeMcpServer

COPIES = [
    pytest.param(copy.deepcopy, id="deepcopy"),
    pytest.param(lambda spec: pickle.loads(pickle.dumps(spec)), id="pickle"),
]


@pytest.mark.parametrize("make_copy", COPIES)
def test_fresh_spec_copies(make_copy) -> None:  # noqa: ANN001
    clone = make_copy(ClawifiToolSpec(max_results=7))
    assert clone.max_results == 7


@pytest.mark.parametrize("make_copy", COPIES)
def test_spec_copies_after_calls_and_the_copy_works(make_copy, router: Router) -> None:  # noqa: ANN001
    keyless = ClawifiToolSpec()
    keyless.search("a")
    asyncio.run(keyless.asearch("b"))
    clone = make_copy(keyless)
    assert clone.search("c") and asyncio.run(clone.asearch("d"))

    rest = ClawifiToolSpec(api_key="k")
    rest.fetch("https://example.com")
    rest_clone = make_copy(rest)
    assert rest_clone._sync is None
    assert rest_clone.fetch("https://example.com")[0].text == "Body"


def test_unrelated_session_message_does_not_initialize() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(400, text="invalid session parameter")
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    with pytest.raises(ClawifiKeylessError, match="invalid session parameter"):
        spec.search("a")
    assert server.methods() == ["tools/call"]


def test_null_id_error_is_raised() -> None:
    server = FakeMcpServer()
    server.fail_next = httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
    )
    spec = ClawifiToolSpec(keyless_client=KeylessClient(transport=httpx.MockTransport(server)))
    with pytest.raises(ClawifiKeylessError, match="Parse error") as info:
        spec.search("a")
    assert info.value.code == -32700
