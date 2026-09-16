from __future__ import annotations

import pytest

from context_router.evaluation.judge import (
    CoverageJudge,
    JudgeOutcome,
    JudgePair,
    LLMJudge,
    Verdict,
    build_judge_prompt,
    parse_verdict,
    run_pairwise_judging,
    summarise_wins,
)
from context_router.providers.openai_compatible import AnswerResult


def pair(
    answer_a: str, answer_b: str, *, arm_a: str = "hybrid_router", arm_b: str = "full_history"
) -> JudgePair:
    return JudgePair(
        sample_id="s-01",
        query="回到 SQLite migration，锁升级怎么定？",
        requirements=["覆盖 migration.py 的「锁升级」结论"],
        arm_a=arm_a,
        arm_b=arm_b,
        answer_a=answer_a,
        answer_b=answer_b,
    )


BETTER = "migration.py 的锁升级需要先收敛约束。"
WORSE = "不太确定。"


def test_judge_prompt_never_leaks_the_arm_names() -> None:
    """Blinding that depends on the caller remembering to omit something is not blinding."""

    secret_arm = "hybrid_router_with_trained_ranker"
    prompt = build_judge_prompt(query="q", requirements=["r"], answer_a="a", answer_b="b")
    assert secret_arm not in prompt
    assert "hybrid_router" not in prompt
    assert "full_history" not in prompt


def test_judge_prompt_carries_the_query_the_rubric_and_both_answers() -> None:
    prompt = build_judge_prompt(
        query="回到 SQLite migration",
        requirements=["覆盖 migration.py 的「锁升级」结论"],
        answer_a="answer one",
        answer_b="answer two",
    )

    assert "回到 SQLite migration" in prompt
    assert "migration.py" in prompt
    assert "answer one" in prompt
    assert "answer two" in prompt


def test_coverage_judge_prefers_the_answer_that_meets_the_rubric() -> None:
    judge = CoverageJudge()

    assert (
        judge.compare(
            query="q",
            requirements=["覆盖 migration.py 的「锁升级」结论"],
            answer_a=BETTER,
            answer_b=WORSE,
        )
        == "a"
    )
    assert (
        judge.compare(
            query="q",
            requirements=["覆盖 migration.py 的「锁升级」结论"],
            answer_a=WORSE,
            answer_b=BETTER,
        )
        == "b"
    )
    assert (
        judge.compare(
            query="q",
            requirements=["覆盖 migration.py 的「锁升级」结论"],
            answer_a=WORSE,
            answer_b=WORSE,
        )
        == "tie"
    )


class _PositionBiasedJudge:
    """Always picks whichever answer it is shown first. A swap must catch this."""

    model_version = "position-biased"

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query, requirements, answer_a, answer_b
        return "a"


class _ConsistentJudge:
    model_version = "consistent"

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query, requirements
        return "a" if len(answer_a) > len(answer_b) else "b"


def test_swapping_catches_a_position_biased_judge() -> None:
    outcomes = run_pairwise_judging(_PositionBiasedJudge(), [pair("long answer text", "x")])

    assert len(outcomes) == 1
    assert outcomes[0].swapped is True
    assert outcomes[0].agreement is False
    assert outcomes[0].winner == "tie", "a verdict about position is not a verdict"


def test_swapping_agrees_for_a_judge_that_ignores_order() -> None:
    outcomes = run_pairwise_judging(_ConsistentJudge(), [pair("long answer text", "x")])

    assert outcomes[0].agreement is True
    assert outcomes[0].winner == "a"


def test_verdicts_are_resolved_onto_arms_not_positions() -> None:
    """A verdict of 'a' in the reversed order belongs to the second arm."""

    outcomes = run_pairwise_judging(_ConsistentJudge(), [pair("x", "much longer answer")])

    assert outcomes[0].winner == "b"
    assert outcomes[0].arm_a == "hybrid_router"
    assert outcomes[0].arm_b == "full_history"


def test_without_swap_agreement_is_unknown() -> None:
    outcomes = run_pairwise_judging(_ConsistentJudge(), [pair("long", "x")], swap=False)

    assert outcomes[0].swapped is False
    assert outcomes[0].agreement is None


def test_summary_counts_wins_and_reports_order_agreement() -> None:
    outcomes = [
        JudgeOutcome(
            sample_id="s-01",
            arm_a="hybrid_router",
            arm_b="full_history",
            winner="a",
            swapped=True,
            agreement=True,
        ),
        JudgeOutcome(
            sample_id="s-02",
            arm_a="hybrid_router",
            arm_b="full_history",
            winner="tie",
            swapped=True,
            agreement=False,
        ),
    ]

    tally = summarise_wins(outcomes)

    assert tally["hybrid_router"]["wins"] == 1
    assert tally["hybrid_router"]["ties"] == 1
    assert tally["hybrid_router"]["win_rate"] == 0.5
    assert tally["full_history"]["losses"] == 1
    assert tally["full_history"]["order_agreement"] == 0.5


class _ScriptedModel:
    """Returns canned judge replies so the parsing and counting can be tested offline."""

    model_version = "scripted-judge-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def run(self, *, prompt: str, instructions: str) -> AnswerResult:
        del instructions
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else ""
        return AnswerResult(
            text=text,
            model=self.model_version,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            stop_reason="end_turn",
            raw_response={},
        )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"winner": "a", "reason": "covers more"}', "a"),
        ('```json\n{"winner": "b", "reason": "x"}\n```', "b"),
        ('Here is my verdict: {"winner": "tie", "reason": "equivalent"}', "tie"),
        ("winner: a", "a"),
        ("Winner: B", "b"),
        ("  a  ", "a"),
        ("The better answer is B.", None),
        ("", None),
    ],
)
def test_verdict_parsing_tolerates_real_reply_shapes(reply: str, expected: str | None) -> None:
    assert parse_verdict(reply) == expected


def test_an_unparseable_reply_is_counted_not_silently_called_a_tie() -> None:
    """An unreadable judge is a broken instrument, not agreement."""

    judge = LLMJudge(_ScriptedModel(["I refuse to pick."]))

    assert judge.compare(query="q", requirements=["r"], answer_a="A", answer_b="B") == "tie"
    assert judge.parse_failures == 1


def test_the_llm_judge_forwards_the_blinded_prompt_and_accumulates_usage() -> None:
    model = _ScriptedModel(['{"winner": "a", "reason": "x"}'])
    judge = LLMJudge(model)

    assert judge.compare(query="q", requirements=["r"], answer_a="AAA", answer_b="BBB") == "a"
    assert judge.model_version == "llm-judge:scripted-judge-model"
    assert "AAA" in model.prompts[0] and "BBB" in model.prompts[0]
    assert "hybrid_router" not in model.prompts[0]
    assert judge.input_tokens == 10
    assert judge.output_tokens == 5


def test_a_reply_cut_off_mid_json_still_yields_its_winner() -> None:
    """Output caps truncate JSON. The winner is already on the wire, so read it.

    Anchored on the field name: a reply that merely mentions "a" or "b" in prose must not
    be read as a verdict, because a spurious winner is worse than an honest tie.
    """

    assert parse_verdict('{"winner":"a","reason":"the answer is a bit long and then it sto') == "a"
    assert parse_verdict('{"winner": "b", "reason": ') == "b"
    assert parse_verdict("Answer A mentions the file, answer B does not.") is None


def test_unparseable_replies_are_kept_for_diagnosis() -> None:
    """The first judge run lost 40 replies with no way to see what they said."""

    model = _ScriptedModel(["garbage one", "still garbage"])
    judge = LLMJudge(model, retries=1)

    assert judge.compare(query="q", requirements=["r"], answer_a="A", answer_b="B") == "tie"
    assert judge.replies == ["garbage one", "still garbage"]
    assert judge.parse_failures == 1


def test_an_empty_reply_is_retried_rather_than_losing_the_pair() -> None:
    """7 of 40 judge calls returned nothing at all; a retry is cheaper than a lost pair.

    The model spends its output budget on a thinking block first, so when the thinking runs
    long it emits no text block and the reply is empty.
    """

    model = _ScriptedModel(["", '{"winner": "b", "reason": "x"}'])
    judge = LLMJudge(model, retries=1)

    assert judge.compare(query="q", requirements=["r"], answer_a="A", answer_b="B") == "b"
    assert judge.parse_failures == 0
    assert judge.replies == ["", '{"winner": "b", "reason": "x"}']


def test_the_coverage_judge_is_order_invariant_unlike_a_model() -> None:
    """The control's value is that it has no positional noise, so any disagreement with a
    model judge is the model's noise or the model's signal -- never the protocol's."""

    pairs = [
        pair("migration.py 的锁升级需要先收敛", "不太确定。"),
        pair("不太确定。", "migration.py 的锁升级需要先收敛"),
    ]

    outcomes = run_pairwise_judging(CoverageJudge(), pairs, swap=True)

    assert [o.agreement for o in outcomes] == [True, True]
    assert [o.winner for o in outcomes] == ["a", "b"], "resolved onto arms, not positions"
