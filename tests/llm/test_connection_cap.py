"""The model client opens as many connections as it is asked to, unless the operator sets a cap."""

from __future__ import annotations

import pytest

from tuvl import llm
from tuvl.core.config import settings


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}

    class Client:
        is_closed = False

        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(llm.httpx, "AsyncClient", Client)
    llm._clients.clear()
    yield seen
    llm._clients.clear()


async def test_by_default_there_is_no_cap(captured, monkeypatch) -> None:
    monkeypatch.setattr(settings, "tuvl_llm_max_connections", None)
    llm._client()
    assert captured["limits"].max_connections is None


async def test_an_operator_can_cap_the_connections(captured, monkeypatch) -> None:
    monkeypatch.setattr(settings, "tuvl_llm_max_connections", 64)
    llm._client()
    assert captured["limits"].max_connections == 64
