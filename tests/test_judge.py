from __future__ import annotations

from context_router.evaluation.judge import (
    CoverageJudge,
    JudgeOutcome,
    JudgePair,
    Verdict,
    build_judge_prompt,
    run_pairwise_judging,
    summarise_wins,
)


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
