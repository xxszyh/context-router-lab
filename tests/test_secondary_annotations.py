from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from context_router.datasets.real_replay import AnnotatedCheckpoint, RealReplayAnnotation
from context_router.datasets.secondary_annotations import (
    AnswerAdjudicationRecord,
    AnswerAdjudicationSet,
    SecondaryAnswerAnnotationSet,
    SecondaryAnswerRecord,
    audit_answer_adjudication,
    audit_secondary_answer_annotations,
)
from context_router.domain import FlatContext


def secondary(*records: SecondaryAnswerRecord) -> SecondaryAnswerAnnotationSet:
    return SecondaryAnswerAnnotationSet(
        dataset_id="real-1",
        annotator_id="annotator-2",
        annotated_at=date(2026, 9, 18),
        independence="user_reported",
        source_file_name="blind.xlsx",
        source_workbook_sha256="a" * 64,
        selection_seed="seed-v1",
        records=list(records),
    )


def record(
    requirement: str = "指出「alpha.py」",
    *,
    blind_id: str = "B01",
    sample_id: str = "q-1",
) -> SecondaryAnswerRecord:
    return SecondaryAnswerRecord(
        blind_id=blind_id,
        sample_id=sample_id,
        answer_status="answerable",
        answer_requirements=[requirement],
    )


def primary() -> RealReplayAnnotation:
    return RealReplayAnnotation(
        source_session_id="real-1",
        contexts=[
            FlatContext(
                context_id="ctx-1",
                name="one",
                goal="one",
                summary="one",
                entities=[],
                lexical_terms=[],
                status="active",
                created_at_event="event-1",
                last_active_sequence=1,
                version=1,
            )
        ],
        checkpoints=[
            AnnotatedCheckpoint(
                sample_id="q-1",
                query_event_id="event-2",
                as_of_sequence=2,
                query_type="continue",
                relation_label="continue",
                answer_requirements=["回答包含「alpha.py」", "回答包含「fixed」"],
                annotator="primary",
            )
        ],
    )


def test_secondary_annotation_requires_scoreable_requirements() -> None:
    with pytest.raises(ValidationError, match="nothing to match on"):
        record("只写无法被评分的中文概括")


def test_no_substantive_answer_cannot_carry_requirements() -> None:
    with pytest.raises(ValidationError, match="cannot carry"):
        SecondaryAnswerRecord(
            blind_id="B01",
            sample_id="q-1",
            answer_status="no_substantive_answer",
            answer_requirements=["提到「alpha.py」"],
        )


def test_secondary_set_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ValidationError, match="sample ids must be unique"):
        secondary(record(), record(blind_id="B02"))


def test_committed_secondary_annotation_is_valid() -> None:
    path = Path("datasets/real-replay/secondary-annotator-2-2026-09-18.json")
    annotation = SecondaryAnswerAnnotationSet.model_validate_json(path.read_text(encoding="utf-8"))

    assert len(annotation.records) == 6
    assert sum(len(item.answer_requirements) for item in annotation.records) == 11
    assert annotation.annotator_type == "human"
    assert annotation.independence == "user_reported"


def test_audit_preserves_disagreement_instead_of_adjudicating_it() -> None:
    audit = audit_secondary_answer_annotations(
        secondary(record("指出「alpha.py」")),
        primary(),
        {"q-1": "alpha.py was fixed"},
    )

    assert audit.records == 1
    assert audit.status_matches == 1
    assert audit.requirements == 1
    assert audit.supported_requirements == 1
    assert audit.primary_requirements == 2
    assert audit.primary_requirements_recalled == 1
    assert audit.secondary_requirements_recalled_by_primary == 1


def test_audit_rejects_an_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="unknown sample"):
        audit_secondary_answer_annotations(
            secondary(record(sample_id="q-missing")),
            primary(),
            {"q-missing": "alpha.py"},
        )


def test_committed_adjudication_is_valid_and_preserves_source_contradiction() -> None:
    adjudication_path = Path("datasets/real-replay/adjudication-2026-09-18.json")
    primary_path = Path("datasets/real-replay/claude-d22f2593.json")
    secondary_path = Path("datasets/real-replay/secondary-annotator-2-2026-09-18.json")
    adjudication = AnswerAdjudicationSet.model_validate_json(
        adjudication_path.read_text(encoding="utf-8")
    )
    primary_labels = RealReplayAnnotation.model_validate_json(
        primary_path.read_text(encoding="utf-8")
    )
    secondary_labels = SecondaryAnswerAnnotationSet.model_validate_json(
        secondary_path.read_text(encoding="utf-8")
    )
    answers = {
        record.sample_id: " ".join(record.final_answer_requirements)
        for record in adjudication.records
    }
    audit = audit_answer_adjudication(adjudication, primary_labels, secondary_labels, answers)

    assert audit.records == 6
    assert audit.requirements == 13
    assert audit.supported_requirements == 13
    assert audit.gold_labels_changed == 0
    assert audit.decision_inconsistencies == 0
    corrected = next(detail for detail in audit.details if detail.blind_id == "B06")
    assert corrected.final_matches == "primary"
    assert corrected.decision_consistent


def test_no_answer_adjudication_cannot_carry_final_requirements() -> None:
    with pytest.raises(ValidationError, match="cannot carry"):
        AnswerAdjudicationRecord(
            blind_id="B01",
            sample_id="q-1",
            source_decision="confirm_no_substantive_answer",
            final_answer_requirements=["提到「alpha.py」"],
        )


def test_adjudication_set_rejects_duplicate_sample_ids() -> None:
    item = AnswerAdjudicationRecord(
        blind_id="B01",
        sample_id="q-1",
        source_decision="use_primary",
        final_answer_requirements=["提到「alpha.py」"],
    )
    with pytest.raises(ValidationError, match="sample ids must be unique"):
        AnswerAdjudicationSet(
            dataset_id="real-1",
            adjudicator_id="adjudicator-1",
            source_file_name="adjudication.xlsx",
            source_workbook_sha256="b" * 64,
            primary_annotation_file="primary.json",
            secondary_annotation_file="secondary.json",
            records=[item, item.model_copy(update={"blind_id": "B02"})],
        )
