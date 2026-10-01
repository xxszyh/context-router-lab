from __future__ import annotations

import pytest

from context_router.evaluation.judge import (
    JUDGE_INSTRUCTIONS,
    JUDGE_INSTRUCTIONS_ANCHORED,
    JUDGE_STYLES,
    CoverageJudge,
    JudgeOutcome,
    JudgePair,
    LLMJudge,
    RefusalGatedJudge,
    Verdict,
    build_judge_prompt,
    judge_instructions,
    parse_verdict,
    run_pairwise_judging,
    summarise_judge_strata,
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


class _CountingJudge:
    model_version = "counting"

    def __init__(self, verdict: Verdict = "a") -> None:
        self.verdict = verdict
        self.calls = 0

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query, requirements, answer_a, answer_b
        self.calls += 1
        return self.verdict


class _ConsistentJudge:
    model_version = "consistent"

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query, requirements
        return "a" if len(answer_a) > len(answer_b) else "b"


class _FlakyJudge:
    """Decides by answer length, so it is position-independent, but abstains on chosen calls.

    Position-independence is what makes it a stand-in for a real judge rather than a stub that
    ignores its input: the same comparison shown in the other order must come back flipped, or
    the swap is measuring the stub. It abstains rather than reversing on the flaky calls, which
    is the shape the project's runs actually show -- 42% of their ties are the orders
    contradicting each other.
    """

    model_version = "flaky"

    def __init__(self, abstains_on: set[int]) -> None:
        self.abstains_on = abstains_on
        self.calls = 0

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query, requirements
        index = self.calls
        self.calls += 1
        if index in self.abstains_on:
            return "tie"
        return "a" if len(answer_a) >= len(answer_b) else "b"


def test_repeats_take_a_strict_majority_over_both_orders() -> None:
    """Six calls, five of them 'a': a decision, and `consistency` says how contested it was."""

    outcomes = run_pairwise_judging(
        _FlakyJudge(abstains_on={3}), [pair("long answer text", "x")], repeats=3
    )

    assert outcomes[0].winner == "a"
    assert outcomes[0].verdicts == ["a", "a", "a", "tie", "a", "a"]
    assert outcomes[0].consistency == pytest.approx(5 / 6)
    assert outcomes[0].agreement is False, "not unanimous, so the coarser flag is False"


def test_an_even_split_is_a_tie_rather_than_whichever_verdict_came_first() -> None:
    """Three of six each way is not a majority, and must not be decided by ordering.

    `Counter.most_common` breaks ties by insertion order, so a three-three split would silently
    become whatever verdict happened to be counted first. The strict `most * 2 > len` test is
    what stops that.
    """

    outcomes = run_pairwise_judging(
        _FlakyJudge(abstains_on={0, 1, 2}), [pair("long answer text", "x")], repeats=3
    )

    assert outcomes[0].verdicts == ["tie", "tie", "tie", "a", "a", "a"]
    assert outcomes[0].winner == "tie"
    assert outcomes[0].consistency == pytest.approx(0.5)


def test_repetition_does_not_invent_a_decision_the_judge_does_not_have() -> None:
    """A judge that always says tie keeps saying tie, however many times it is asked.

    This is the property that separates the two kinds of tie. 42% of this project's ties are the
    orders contradicting each other, which repetition removes; the other 58% are the judge
    finding the answers equal, which it must not remove.
    """

    class _AlwaysTies:
        model_version = "always-ties"

        def compare(
            self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
        ) -> Verdict:
            del query, requirements, answer_a, answer_b
            return "tie"

    outcomes = run_pairwise_judging(_AlwaysTies(), [pair("x", "y")], repeats=4)

    assert outcomes[0].winner == "tie"
    assert outcomes[0].verdicts == ["tie"] * 8
    assert outcomes[0].consistency == 1.0, "unanimous, and unanimous on a tie"


def test_one_repeat_reproduces_the_old_two_order_behaviour() -> None:
    """The default must not change what every earlier run measured."""

    consistent = run_pairwise_judging(_ConsistentJudge(), [pair("long answer text", "x")])
    assert consistent[0].winner == "a"
    assert consistent[0].agreement is True
    assert consistent[0].consistency == 1.0

    biased = run_pairwise_judging(_PositionBiasedJudge(), [pair("long answer text", "x")])
    assert biased[0].winner == "tie"
    assert biased[0].agreement is False
    assert biased[0].consistency == 0.5


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


def test_refusal_gate_ties_two_refusals_without_calling_the_delegate() -> None:
    delegate = _CountingJudge()
    judge = RefusalGatedJudge(delegate)

    verdict = judge.compare(
        query="q",
        requirements=["explain the implementation"],
        answer_a="上下文不足，无法回答。",
        answer_b="The provided context does not contain that information.",
    )

    assert verdict == "tie"
    assert delegate.calls == 0
    assert judge.gate_hits == 1


def test_refusal_gate_does_not_decide_when_only_one_answer_refuses() -> None:
    delegate = _CountingJudge(verdict="b")
    judge = RefusalGatedJudge(delegate)

    verdict = judge.compare(
        query="q",
        requirements=["explain the implementation"],
        answer_a="上下文不足，无法回答。",
        answer_b="The implementation uses an append-only event store.",
    )

    assert verdict == "b"
    assert delegate.calls == 1
    assert judge.gate_hits == 0


def test_refusal_gate_skips_both_swapped_calls_for_a_double_refusal_pair() -> None:
    delegate = _CountingJudge()
    judge = RefusalGatedJudge(delegate)

    outcomes = run_pairwise_judging(
        judge,
        [pair("无法据此确定。", "There is not enough information to answer.")],
    )

    assert outcomes[0].winner == "tie"
    assert outcomes[0].swapped is False
    assert outcomes[0].agreement is None
    assert outcomes[0].decision_source == "refusal_gate"
    assert delegate.calls == 0
    assert judge.gate_hits == 1


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


def test_summary_reports_unmeasured_agreement_as_none() -> None:
    outcome = JudgeOutcome(
        sample_id="gated",
        arm_a="hybrid_router",
        arm_b="full_history",
        winner="tie",
        swapped=False,
        agreement=None,
        decision_source="refusal_gate",
    )

    tally = summarise_wins([outcome])

    assert tally["hybrid_router"]["order_agreement"] is None
    assert tally["full_history"]["order_agreement"] is None


def test_judge_summary_separates_checkpoint_and_response_refusal_strata() -> None:
    pairs = [
        pair("无法回答。", "No information is available.").model_copy(
            update={"sample_id": "must-refuse", "expects_refusal": True}
        ),
        pair("无法回答。", BETTER).model_copy(update={"sample_id": "one-refusal"}),
        pair(BETTER, WORSE).model_copy(update={"sample_id": "no-refusal"}),
    ]
    outcomes = [
        JudgeOutcome(
            sample_id="must-refuse",
            arm_a="hybrid_router",
            arm_b="full_history",
            winner="tie",
            swapped=True,
            agreement=True,
        ),
        JudgeOutcome(
            sample_id="one-refusal",
            arm_a="hybrid_router",
            arm_b="full_history",
            winner="b",
            swapped=True,
            agreement=True,
        ),
        JudgeOutcome(
            sample_id="no-refusal",
            arm_a="hybrid_router",
            arm_b="full_history",
            winner="a",
            swapped=True,
            agreement=False,
        ),
    ]

    summary = summarise_judge_strata(pairs, outcomes)

    assert summary["checkpoint"]["must_refuse"]["pairs"] == 1
    assert summary["checkpoint"]["answerable"]["pairs"] == 2
    assert summary["response"]["both_refuse"]["ties"] == 1
    assert summary["response"]["both_refuse"]["judged_pairs"] == 1
    assert summary["response"]["one_refuses"]["arm_b_wins"] == 1
    assert summary["response"]["neither_refuses"]["disagreed"] == 1


class _ScriptedModel:
    """Returns canned judge replies so the parsing and counting can be tested offline."""

    model_version = "scripted-judge-model"

    def __init__(self, replies: list[str], *, stop_reason: str | None = "end_turn") -> None:
        self.replies = list(replies)
        self.stop_reason = stop_reason
        self.prompts: list[str] = []
        #: Recorded rather than discarded so a test can prove which prompt variant reached the
        #: model. The prompt is a parameter of the verdict and was unrecorded for the whole
        #: history of this project's judge numbers.
        self.instructions: list[str] = []

    def run(self, *, prompt: str, instructions: str) -> AnswerResult:
        self.instructions.append(instructions)
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else ""
        return AnswerResult(
            text=text,
            model=self.model_version,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            stop_reason=self.stop_reason,
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
    assert [c.text for c in judge.calls] == ["garbage one", "still garbage"]
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
    assert [c.text for c in judge.calls] == ["", '{"winner": "b", "reason": "x"}']


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


def test_the_last_verdict_mention_wins_over_an_earlier_one() -> None:
    """The judge is told to end with its verdict, so an earlier mention is reasoning."""

    reasoning_then_verdict = (
        "Requirement 1: A addresses it, B does not, so on that basis winner: a.\n"
        "Requirement 2: neither addresses it.\n"
        "WINNER: b"
    )

    assert parse_verdict(reasoning_then_verdict) == "b"


def test_the_judge_is_told_to_anchor_its_verdict_in_the_requirements() -> None:
    """20% order disagreement means the verdict was not anchored in content.

    The instructions must define a tie, forbid counting an echoed requirement as satisfied,
    and require a final verdict line -- each of which gives the decision something to rest
    on other than the position the answers happened to appear in.
    """

    lowered = JUDGE_INSTRUCTIONS.lower()
    assert "tie" in lowered and "same requirements" in lowered
    assert "does not satisfy" in lowered, "the echoed-refusal loophole must be closed"
    assert "winner: <a|b|tie>" in lowered
    assert "order" in lowered and "swapped" in lowered


@pytest.mark.parametrize("style", sorted(JUDGE_STYLES))
def test_every_style_keeps_the_rubric_the_verdict_rests_on(style: str) -> None:
    """A style is a presentation of the rubric, never a different rubric.

    If a variant drops the tie definition or the echoed-refusal rule, its numbers stop being
    comparable with the other variant's, and the comparison the two exist to support is void.
    """

    lowered = judge_instructions(style).lower()
    assert "tie" in lowered and "same requirements" in lowered
    assert "does not satisfy" in lowered
    assert "winner: <a|b|tie>" in lowered
    assert "order" in lowered and "swapped" in lowered


def test_the_anchored_style_asks_for_the_quote_the_terse_one_forbids() -> None:
    """This difference is the whole experiment, so it is pinned rather than described.

    The terse prompt tells the judge not to restate either answer, which leaves the
    satisfied/not decision resting on nothing quotable. The anchored prompt requires the
    quote. If a later edit softens either line, the two styles stop being a contrast.
    """

    assert "do not restate either answer" in JUDGE_INSTRUCTIONS.lower()
    assert "quote" in JUDGE_INSTRUCTIONS_ANCHORED.lower()
    assert "do not restate" not in JUDGE_INSTRUCTIONS_ANCHORED.lower()


def test_the_chosen_style_is_what_reaches_the_model() -> None:
    """The default must stay the historical prompt, so old numbers stay reproducible."""

    default_model = _ScriptedModel(["WINNER: a"])
    LLMJudge(default_model).compare(query="q", requirements=["r"], answer_a="A", answer_b="B")
    assert default_model.instructions == [JUDGE_INSTRUCTIONS]

    anchored_model = _ScriptedModel(["WINNER: a"])
    LLMJudge(anchored_model, instructions=JUDGE_INSTRUCTIONS_ANCHORED).compare(
        query="q", requirements=["r"], answer_a="A", answer_b="B"
    )
    assert anchored_model.instructions == [JUDGE_INSTRUCTIONS_ANCHORED]


def test_the_prompt_digest_travels_with_the_judge_and_tracks_the_text() -> None:
    """An artefact has to be able to say which instrument produced it.

    The style name is readable and the digest is not; recording both is what stops a prompt
    edit from silently re-labelling every number already published under the old one.
    """

    terse = LLMJudge(_ScriptedModel(["WINNER: a"]))
    anchored = LLMJudge(_ScriptedModel(["WINNER: a"]), instructions=JUDGE_INSTRUCTIONS_ANCHORED)

    assert terse.instructions_sha256 != anchored.instructions_sha256
    assert len(terse.instructions_sha256) == 64
    # Stable across instances, or it identifies nothing.
    assert terse.instructions_sha256 == LLMJudge(_ScriptedModel([])).instructions_sha256


def test_an_unknown_style_is_rejected_rather_than_defaulted() -> None:
    """Silently falling back to the default would run the old prompt under a new name."""

    with pytest.raises(ValueError, match="unknown judge style"):
        judge_instructions("anvhored")


def test_the_instructions_are_in_the_blinded_judge_prompt_path() -> None:
    """The procedure must reach the model without the prompt builder learning about arms."""

    model = _ScriptedModel(["WINNER: a"])
    judge = LLMJudge(model)

    assert judge.compare(query="q", requirements=["r"], answer_a="A", answer_b="B") == "a"
    assert "hybrid_router" not in model.prompts[0]


def test_a_cut_off_reply_is_distinguishable_from_one_never_produced() -> None:
    """The two need different fixes: a bigger budget, versus a different prompt.

    Not having this field is why the judge had to be re-run twice to find out which of the
    two was happening.
    """

    model = _ScriptedModel(["", '{"winner": "a", "reason": "x"}'])
    judge = LLMJudge(model, retries=1)
    judge.compare(query="q", requirements=["r"], answer_a="A", answer_b="B")

    assert [call.stop_reason for call in judge.calls] == ["end_turn", "end_turn"]
    assert [call.output_tokens for call in judge.calls] == [5, 5]
