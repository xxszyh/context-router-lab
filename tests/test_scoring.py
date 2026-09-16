from __future__ import annotations

import pytest

from context_router.datasets.synthetic import REFUSAL_REQUIREMENT
from context_router.evaluation.scoring import (
    deterministic_coverage,
    is_refusal_requirement,
    requirement_satisfied,
    requirement_terms,
)


def test_terms_are_the_identifier_and_the_quoted_topic() -> None:
    assert requirement_terms("覆盖 migration.py 的「锁升级」结论") == ["migration.py", "锁升级"]


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
