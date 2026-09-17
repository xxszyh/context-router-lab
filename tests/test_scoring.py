from __future__ import annotations

import pytest

from context_router.datasets.synthetic import REFUSAL_REQUIREMENT
from context_router.evaluation.scoring import (
    deterministic_coverage,
    is_refusal,
    is_refusal_requirement,
    normalize_for_matching,
    requirement_satisfied,
    requirement_terms,
)


def test_terms_are_the_identifier_and_the_quoted_topic() -> None:
    assert requirement_terms("覆盖 migration.py 的「锁升级」结论") == ["migration.py", "锁升级"]


def test_one_character_identifiers_are_not_terms() -> None:
    """A single letter matches almost any answer, so it is a liability rather than a check.

    Real examples: `A` in 「方案 A」, `C` in 「ρ(C)」, `J` in 「λJ₁(λ)=Bi·J₀(λ)」, `r` in
    「白色内孔椭圆（r）」. None of them carries content -- the quoted span does -- and each
    becomes the *only* thing matching once the reading is loosened.
    """

    assert requirement_terms("给出「方案 A」与「方案 B」两种处理方式") == ["方案 A", "方案 B"]
    assert requirement_terms("回答题目条件「不多余」：「ρ(C)」是热物性") == ["不多余", "ρ(C)"]
    # An identifier inside a quoted span is extracted twice, which is harmless: requiring it
    # twice is requiring it once. Only the one-character ones are dropped.
    assert requirement_terms("指出只有「输出时」才需要换算，域外「填 NaN」") == [
        "NaN",
        "输出时",
        "填 NaN",
    ]


def test_a_lone_single_character_leaves_nothing_to_match_on() -> None:
    """Which is the correct verdict: `必须改 A` cannot be checked against an answer."""

    assert requirement_terms("必须改 A") == []
    assert not requirement_satisfied("必须改 A", "A 已经改了")


def test_requirement_is_satisfied_only_when_both_terms_appear() -> None:
    requirement = "覆盖 migration.py 的「锁升级」结论"

    assert requirement_satisfied(requirement, "锁升级 需要在 migration.py 里先收敛。")
    assert not requirement_satisfied(requirement, "需要在 migration.py 里先收敛。")
    assert not requirement_satisfied(requirement, "锁升级 需要先收敛。")


def test_coverage_is_the_fraction_of_requirements_met() -> None:
    requirements = ["覆盖 migration.py 的「锁升级」结论", "覆盖 routing.py 的「RRF 融合」结论"]

    assert deterministic_coverage(requirements, "migration.py 锁升级；routing.py RRF 融合") == 1.0
    assert deterministic_coverage(requirements, "migration.py 锁升级") == 0.5
    assert deterministic_coverage(requirements, "完全无关的回答") == 0.0
    assert deterministic_coverage([], "anything") == 1.0


@pytest.mark.parametrize(
    "answer",
    ["上下文中没有相关信息，无法回答。", "The context does not contain the answer."],
)
def test_refusal_requirement_is_scored_as_a_refusal(answer: str) -> None:
    assert is_refusal_requirement(REFUSAL_REQUIREMENT)
    assert deterministic_coverage([REFUSAL_REQUIREMENT], answer) == 1.0


def test_fabricating_an_answer_fails_the_refusal_requirement() -> None:
    """The whole point of an unanswerable checkpoint is that guessing must not score."""

    fabricated = "根据上下文，部署密钥是 sk-abc123，在 migration.py 里。"
    assert deterministic_coverage([REFUSAL_REQUIREMENT], fabricated) == 0.0


def test_an_answer_that_echoes_the_rubric_while_declining_is_detectable() -> None:
    """The lexical check alone is fooled by this, and it flatters whoever writes fluently.

    Observed in the first real run: the answer quoted the required identifier and sub-topic
    and then said it could not answer, and scored full marks. Coverage must be reportable
    separately from declining so the difference is visible instead of averaged away.
    """

    requirement = "覆盖 migration.py 的「锁升级」结论"
    echo_then_decline = (
        "回答中没有说明 migration.py 的锁升级结论。它只提到 WAL，未给出两者结合的依据。"
    )

    assert deterministic_coverage([requirement], echo_then_decline) == 1.0
    assert is_refusal(echo_then_decline) is True, "and this is what must be surfaced"


def test_a_real_answer_is_not_mistaken_for_a_refusal() -> None:
    real = "migration.py 的锁升级需要先收敛事务范围，若仍不稳定就回退到上一个可用版本。"
    assert is_refusal(real) is False


@pytest.mark.parametrize(
    "answer",
    [
        "内部运动规律对 τ₄ **完全无影响**。",
        "内部运动规律对 τ₄ *完全无影响*。",
    ],
)
def test_markdown_emphasis_inside_a_quoted_span_still_matches(answer: str) -> None:
    """Every one of the eight misses on the first real label set was this.

    A model that emits markdown would otherwise score below one that does not, on identical
    content -- and this scorer's whole job is comparing arms.
    """

    requirement = "指出「内部运动规律对 τ₄ 完全无影响」"
    assert requirement_satisfied(requirement, answer)


def test_underscore_emphasis_is_deliberately_not_normalised() -> None:
    """A known limitation, kept because the cure costs more than the disease.

    ``__bold__`` is markdown, but ``_`` is also an identifier character in this benchmark's
    own vocabulary -- ``required_context_ids``, ``answer_requirements``, ``__init__.py``.
    Stripping every underscore would make those collide with each other, which is a worse
    failure than missing a rarer emphasis style. ``**`` and backticks are not identifier
    characters, so they are stripped.
    """

    assert not requirement_satisfied(
        "指出「内部运动规律对 τ₄ 完全无影响」", "内部运动规律对 τ₄ __完全无影响__。"
    )


def test_backticks_and_escaped_asterisks_do_not_break_a_quoted_span() -> None:
    assert requirement_satisfied("给出「R* = 1.2112 cm」", "令其 = 760 → **R\\* = 1.2112 cm**")
    assert requirement_satisfied("找到「最初的 T1.py」", "最后一轮审核针对的是最初的 `T1.py`")


def test_spacing_inside_a_quoted_term_does_not_break_it() -> None:
    assert requirement_satisfied("给出「λJ₁(λ)=Bi·J₀(λ)」", "特征方程 **λJ₁(λ) = Bi·J₀(λ)**")


def test_normalisation_does_not_make_unrelated_answers_match() -> None:
    """The loosening must stay inside the quoted span, not turn every pair into a match."""

    requirement = "指出「内部运动规律对 τ₄ 完全无影响」"
    assert not requirement_satisfied(requirement, "内部运动规律**有**影响。")
    assert not requirement_satisfied(requirement, "内部运动规律对 τ₃ 完全无影响。")


def test_identifier_characters_survive_normalisation() -> None:
    assert normalize_for_matching("T2_CN.py") == "t2_cn.py"
