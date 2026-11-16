"""What a loop agent holds the model to: ``tool_choice`` on each turn, the named ``finish`` on the repair
turn, and what the ``llm_turn`` event records about a reply."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.runtime import test_loop_engine as base
from tests.runtime.test_loop_engine import (  # noqa: F401 - the autouse fixture and the helpers
    agents,
    call,
    events,
    finish,
    start,
    workflows,
)
from tuvl import llm as llm_client
from tuvl.runtime.loop import LoopState
from tuvl.runtime.runner import workflow_hash

FINISH_NAMED = {"type": "function", "function": {"name": "finish"}}


class Recording(base.Script):
    """Scripted turns that note the ``tool_choice`` each request carried; a request can be refused."""

    def __init__(
        self,
        *turns: Any,
        reject: Callable[[Any], bool] | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        super().__init__(*turns)
        self.tool_choices: list[Any] = []
        self.reject = reject
        self.reasons = list(reasons or [])

    async def __call__(self, **kwargs: Any) -> Any:
        choice = kwargs.get("tool_choice")
        self.tool_choices.append(choice)
        if self.reject is not None and self.reject(choice):
            raise llm_client.BadRequestError("tool_choice is not supported here", status_code=400)
        response = await super().__call__(**kwargs)
        if self.reasons:
            response.choices[0].finish_reason = self.reasons.pop(0)
        return response


def install(monkeypatch: pytest.MonkeyPatch, *turns: Any, **options: Any) -> Recording:
    script = Recording(*turns, **options)
    monkeypatch.setattr(llm_client, "acompletion", script)
    return script


async def test_the_repair_turn_names_finish(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, "I think we are done.", finish())
    run = await start(pg_engine, workflows())
    assert run.status == "completed" and run.output == {"status": 200, "body": "done"}
    assert script.tool_choices == ["auto", FINISH_NAMED]
    turns = await events(pg_engine, run.id, "llm_turn")
    assert [t["tool_choice"] for t in turns] == ["auto", "finish"]


async def test_a_model_that_still_answers_in_text_ends_as_before(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, "done, I think", "really done")
    run = await start(pg_engine, workflows())
    done = (await events(pg_engine, run.id, "agent_completed"))[0]
    assert (done["signal"], done["error_type"], done["message"]) == (
        "error",
        "parse_error",
        "the model never called finish",
    )
    assert script.tool_choices == ["auto", FINISH_NAMED]  # one repair, no more


async def test_a_finish_that_does_not_fit_is_repaired_with_a_named_finish(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, [call("finish", outcome="nonsense", resolution="x")], finish())
    run = await start(pg_engine, workflows())
    assert run.output == {"status": 200, "body": "done"}
    assert script.tool_choices == ["auto", FINISH_NAMED]


async def test_a_provider_that_refuses_the_named_choice_is_asked_with_the_configured_one(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, "text first", finish(), reject=lambda c: isinstance(c, dict))
    run = await start(pg_engine, workflows())
    assert run.output == {"status": 200, "body": "done"}
    assert script.tool_choices == ["auto", FINISH_NAMED, "auto"]
    turns = await events(pg_engine, run.id, "llm_turn")
    assert [(t["tool_choice"], t.get("tool_choice_rejected")) for t in turns] == [
        ("auto", None),
        ("auto", True),
    ]


async def test_required_holds_every_turn_and_the_repair_still_names_finish(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, [call("lookup_order", order_id="o-1")], "text", finish())
    run = await start(pg_engine, workflows(loop_params={"tool_choice": "required"}))
    assert run.output == {"status": 200, "body": "done"}
    assert script.tool_choices == ["required", "required", FINISH_NAMED]
    turns = await events(pg_engine, run.id, "llm_turn")
    assert [t["tool_choice"] for t in turns] == ["required", "required", "finish"]


async def test_a_refused_required_is_an_error_not_a_fallback(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = install(monkeypatch, finish(), reject=lambda c: c == "required")
    run = await start(pg_engine, workflows(loop_params={"tool_choice": "required"}))
    done = (await events(pg_engine, run.id, "agent_completed"))[0]
    assert (done["signal"], done["error_type"]) == ("error", "llm_error")
    assert script.tool_choices == ["required"]  # the loop's own choice is not quietly dropped


async def test_llm_turn_says_why_a_reply_ended_and_how_much_text_it_held(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = "cut off mid-sentence"
    install(
        monkeypatch,
        [call("lookup_order", order_id="o-1")],
        cut,
        finish(),
        reasons=["tool_calls", "length", "tool_calls"],
    )
    run = await start(pg_engine, workflows())
    turns = await events(pg_engine, run.id, "llm_turn")
    assert [(t["finish_reason"], t["text_chars"]) for t in turns] == [
        ("tool_calls", 0),
        ("length", len(cut)),
        ("tool_calls", 0),
    ]


async def test_a_provider_that_reports_no_reason_leaves_the_field_out(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, finish())
    run = await start(pg_engine, workflows())
    turn = (await events(pg_engine, run.id, "llm_turn"))[0]
    assert "finish_reason" not in turn and turn["text_chars"] == 0


def test_a_saved_loop_state_keeps_a_pending_repair_and_an_older_one_still_loads() -> None:
    pending = LoopState(repair_pending=True)
    assert LoopState(**dataclasses.asdict(pending)).repair_pending is True
    older = dataclasses.asdict(LoopState())
    older.pop("repair_pending")  # paused before the field existed
    assert LoopState(**older).repair_pending is False


def test_the_default_choice_leaves_the_workflow_hash_as_it_was() -> None:
    plain = workflows()["wf"]
    assert "tool_choice" not in plain.model_dump(mode="json")["spec"]["agents"][0]["loop"]
    required = workflows(loop_params={"tool_choice": "required"})["wf"]
    dumped = required.model_dump(mode="json")["spec"]["agents"][0]["loop"]
    assert dumped["tool_choice"] == "required"
    assert workflow_hash(required) != workflow_hash(plain)
