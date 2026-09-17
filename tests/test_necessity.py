from __future__ import annotations

from datetime import UTC, datetime

import pytest

from context_router.domain import RawEvent
from context_router.evaluation.necessity import (
    AblationVerdict,
    build_ablation_prompt,
    context_material,
    control_failures,
    parse_verdict,
    required_from_ablations,
    run_ablations,
)
from context_router.providers.openai_compatible import AnswerResult


def event(event_id: str, sequence: int, content: str) -> RawEvent:
    return RawEvent.create(
        event_id=event_id,
        session_id="real-1",
        sequence=sequence,
        occurred_at=datetime(2026, 9, 1, sequence, tzinfo=UTC),
        ingested_at=datetime(2026, 9, 1, sequence, tzinfo=UTC),
        actor="user",
        kind="message",
        content=content,
    )


class ScriptedJudge:
    """Replies from a script so the ablation loop can be tested without a call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def answer(self, *, query: str, working_context: str, instructions: str) -> AnswerResult:
        del query, instructions
        self.prompts.append(working_context)
        text = self.replies.pop(0) if self.replies else ""
        return AnswerResult(
            text=text,
            model="stub",
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            stop_reason="end_turn",
            raw_response={},
        )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("YES", True),
        ("yes", True),
        ("NO", False),
        ("No.", False),
        ("NO, the file is missing", False),
        ("", None),
        ("maybe", None),
    ],
)
def test_verdict_parsing_is_strict(reply: str, expected: bool | None) -> None:
    assert parse_verdict(reply) == expected


def test_material_prefers_the_events_the_query_matches() -> None:
    """Ablation shows retrieved evidence, not the descriptor, or it tests the wrong thing."""

    events = [
        event("e1", 1, "无关的闲聊内容"),
        event("e2", 2, "Picard 迭代次数不够"),
        event("e3", 3, "更多无关内容"),
    ]

    material = context_material(events, "Picard 迭代怎么调")

    assert "Picard" in material
    assert "闲聊" not in material


def test_material_falls_back_to_recent_turns_without_any_term_overlap() -> None:
    """A context can be required for continuity rather than for a term match."""

    events = [event(f"e{i}", i, f"第{i}轮记录") for i in range(1, 6)]

    material = context_material(events, "完全不相干的提问", limit=2)

    assert "第4轮记录" in material and "第5轮记录" in material


def test_the_prompt_always_carries_the_recent_window() -> None:
    """Required means needed *in addition to* the window, which the real system always has."""

    prompt = build_ablation_prompt(
        query="回到 Picard 那个问题",
        recent_window="用户: 继续这个\n助手: 好的",
        materials={"ctx-a": "材料 A"},
        context_names={"ctx-a": "Arctic sea ice"},
    )

    assert "继续这个" in prompt
    assert "材料 A" in prompt
    assert "Arctic sea ice" in prompt
    assert prompt.rstrip().endswith("回到 Picard 那个问题")


def test_ablation_removes_exactly_one_context_per_call() -> None:
    judge = ScriptedJudge(["YES", "NO"])
    materials = {"ctx-a": "MATERIAL-AAA", "ctx-b": "MATERIAL-BBB"}

    run = run_ablations(
        sample_id="s-1",
        query="q",
        recent_window="WINDOW-TEXT",
        materials=materials,
        context_names={"ctx-a": "Alpha", "ctx-b": "Beta"},
        control_context_ids=set(),
        judge=judge,
    )

    assert len(judge.prompts) == 2
    first, second = judge.prompts
    assert "MATERIAL-AAA" not in first, "the ablated context's material must be hidden"
    assert "MATERIAL-BBB" in first, "every surviving context must still be shown"
    assert "MATERIAL-BBB" not in second
    assert "MATERIAL-AAA" in second
    assert "WINDOW-TEXT" in first and "WINDOW-TEXT" in second
    assert required_from_ablations(run.verdicts) == ["ctx-b"]


def test_an_unreadable_reply_is_counted_and_does_not_become_a_requirement() -> None:
    """A judge nobody can read is a broken instrument, not evidence of necessity."""

    judge = ScriptedJudge(["maybe", "YES"])

    run = run_ablations(
        sample_id="s-1",
        query="q",
        recent_window="w",
        materials={"ctx-a": "A", "ctx-b": "B"},
        context_names={},
        control_context_ids=set(),
        judge=judge,
    )

    assert run.unreadable == 1
    assert [v.ablated_context_id for v in run.verdicts] == ["ctx-b"]


def test_controls_flag_a_judge_that_says_no_to_irrelevant_contexts() -> None:
    """A context nothing depends on should survive its own ablation. If not, the judge is off."""

    verdicts = [
        AblationVerdict(
            sample_id="s",
            ablated_context_id="ctx-irrelevant",
            answerable_without=False,
            raw_reply="NO",
            is_control=True,
        ),
        AblationVerdict(
            sample_id="s",
            ablated_context_id="ctx-needed",
            answerable_without=False,
            raw_reply="NO",
            is_control=False,
        ),
    ]

    assert control_failures(verdicts) == ["ctx-irrelevant"]
    assert required_from_ablations(verdicts) == ["ctx-irrelevant", "ctx-needed"]
