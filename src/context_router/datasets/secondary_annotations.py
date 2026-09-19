"""Independent answer-requirement annotations kept separate from adjudicated gold labels."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from context_router.datasets.real_replay import RealReplayAnnotation
from context_router.domain import Contract
from context_router.evaluation.scoring import requirement_satisfied, requirement_terms

AnswerStatus = Literal["answerable", "no_substantive_answer"]
AnnotatorType = Literal["human", "model", "unknown"]
Independence = Literal["confirmed", "user_reported", "unknown"]
AdjudicationDecision = Literal[
    "use_primary",
    "use_secondary",
    "merge_rewrite",
    "equivalent",
    "confirm_no_substantive_answer",
]
FinalMatch = Literal["primary", "secondary", "both", "neither", "no_answer"]


class SecondaryAnswerRecord(Contract):
    """One blinded checkpoint labelled without overwriting the benchmark's gold label."""

    blind_id: str
    sample_id: str
    answer_status: AnswerStatus
    answer_requirements: list[str] = Field(default_factory=list, max_length=3)
    notes: str = ""

    @model_validator(mode="after")
    def validate_requirements(self) -> SecondaryAnswerRecord:
        if self.answer_status == "answerable" and not self.answer_requirements:
            raise ValueError("an answerable record needs at least one answer requirement")
        if self.answer_status == "no_substantive_answer" and self.answer_requirements:
            raise ValueError("a no-substantive-answer record cannot carry answer requirements")
        for requirement in self.answer_requirements:
            if not requirement_terms(requirement):
                raise ValueError(f"requirement has nothing to match on: {requirement!r}")
        return self


class SecondaryAnswerAnnotationSet(Contract):
    """A provenance-preserving label set produced from a blinded worksheet."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    annotator_id: str
    annotator_type: AnnotatorType = "unknown"
    annotated_at: date | None = None
    independence: Independence = "unknown"
    source_file_name: str
    source_workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_seed: str
    provenance_note: str = ""
    records: list[SecondaryAnswerRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_records(self) -> SecondaryAnswerAnnotationSet:
        blind_ids = [record.blind_id for record in self.records]
        sample_ids = [record.sample_id for record in self.records]
        if len(blind_ids) != len(set(blind_ids)):
            raise ValueError("blind ids must be unique")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample ids must be unique")
        return self


class AnswerAdjudicationRecord(Contract):
    """The final answer label plus the adjudicator's unmodified worksheet decision."""

    blind_id: str
    sample_id: str
    source_decision: AdjudicationDecision
    final_answer_requirements: list[str] = Field(default_factory=list, max_length=3)
    notes: str = ""

    @model_validator(mode="after")
    def validate_requirements(self) -> AnswerAdjudicationRecord:
        no_answer = self.source_decision == "confirm_no_substantive_answer"
        if no_answer and self.final_answer_requirements:
            raise ValueError("a confirmed no-answer record cannot carry final requirements")
        if not no_answer and not self.final_answer_requirements:
            raise ValueError("an answer adjudication needs at least one final requirement")
        for requirement in self.final_answer_requirements:
            if not requirement_terms(requirement):
                raise ValueError(f"final requirement has nothing to match on: {requirement!r}")
        return self


class AnswerAdjudicationSet(Contract):
    """A provenance-preserving import of a completed adjudication worksheet."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    adjudicator_id: str
    adjudicator_type: AnnotatorType = "unknown"
    adjudicated_at: date | None = None
    source_file_name: str
    source_workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_annotation_file: str
    secondary_annotation_file: str
    provenance_note: str = ""
    records: list[AnswerAdjudicationRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_records(self) -> AnswerAdjudicationSet:
        blind_ids = [record.blind_id for record in self.records]
        sample_ids = [record.sample_id for record in self.records]
        if len(blind_ids) != len(set(blind_ids)):
            raise ValueError("blind ids must be unique")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample ids must be unique")
        return self


class SecondaryRecordAudit(Contract):
    blind_id: str
    sample_id: str
    status_matches_reference: bool
    supported_requirements: list[bool]
    primary_requirements_recalled: list[bool]
    secondary_requirements_recalled_by_primary: list[bool]


class SecondaryAnnotationAudit(Contract):
    records: int
    status_matches: int
    requirements: int
    supported_requirements: int
    primary_requirements: int
    primary_requirements_recalled: int
    secondary_requirements_recalled_by_primary: int
    details: list[SecondaryRecordAudit]


class AdjudicationRecordAudit(Contract):
    blind_id: str
    sample_id: str
    final_matches: FinalMatch
    decision_consistent: bool
    status_matches_reference: bool
    supported_requirements: list[bool]


class AnswerAdjudicationAudit(Contract):
    records: int
    requirements: int
    supported_requirements: int
    status_matches: int
    decision_inconsistencies: int
    gold_labels_changed: int
    details: list[AdjudicationRecordAudit]


def audit_secondary_answer_annotations(
    secondary: SecondaryAnswerAnnotationSet,
    primary: RealReplayAnnotation,
    answers: Mapping[str, str],
) -> SecondaryAnnotationAudit:
    """Compare two raw label sets without silently adjudicating either one.

    The cross-label fields are deliberately lexical diagnostics, not semantic agreement.
    They identify wording and coverage disagreements that need adjudication; they do not decide
    which annotator is correct.
    """

    checkpoints = {checkpoint.sample_id: checkpoint for checkpoint in primary.checkpoints}
    details: list[SecondaryRecordAudit] = []
    for record in secondary.records:
        if record.sample_id not in checkpoints:
            raise ValueError(f"secondary annotation names unknown sample {record.sample_id}")
        if record.sample_id not in answers:
            raise ValueError(f"missing reference answer for {record.sample_id}")
        checkpoint = checkpoints[record.sample_id]
        answer = answers[record.sample_id]
        answer_present = bool(answer.strip())
        status_matches = (record.answer_status == "answerable" and answer_present) or (
            record.answer_status == "no_substantive_answer" and not answer_present
        )
        secondary_text = " ".join(record.answer_requirements)
        primary_text = " ".join(checkpoint.answer_requirements)
        details.append(
            SecondaryRecordAudit(
                blind_id=record.blind_id,
                sample_id=record.sample_id,
                status_matches_reference=status_matches,
                supported_requirements=[
                    requirement_satisfied(requirement, answer)
                    for requirement in record.answer_requirements
                ],
                primary_requirements_recalled=[
                    requirement_satisfied(requirement, secondary_text)
                    for requirement in checkpoint.answer_requirements
                ],
                secondary_requirements_recalled_by_primary=[
                    requirement_satisfied(requirement, primary_text)
                    for requirement in record.answer_requirements
                ],
            )
        )

    return SecondaryAnnotationAudit(
        records=len(details),
        status_matches=sum(detail.status_matches_reference for detail in details),
        requirements=sum(len(detail.supported_requirements) for detail in details),
        supported_requirements=sum(sum(detail.supported_requirements) for detail in details),
        primary_requirements=sum(len(detail.primary_requirements_recalled) for detail in details),
        primary_requirements_recalled=sum(
            sum(detail.primary_requirements_recalled) for detail in details
        ),
        secondary_requirements_recalled_by_primary=sum(
            sum(detail.secondary_requirements_recalled_by_primary) for detail in details
        ),
        details=details,
    )


def audit_answer_adjudication(
    adjudication: AnswerAdjudicationSet,
    primary: RealReplayAnnotation,
    secondary: SecondaryAnswerAnnotationSet,
    answers: Mapping[str, str],
) -> AnswerAdjudicationAudit:
    """Validate final labels while preserving contradictions in worksheet decisions.

    The final-requirement columns are the adjudication output. ``source_decision`` remains
    untouched provenance: if it points at a label set different from the final text, the
    record is usable but the audit reports the inconsistency instead of silently rewriting it.
    """

    primary_by_id = {checkpoint.sample_id: checkpoint for checkpoint in primary.checkpoints}
    secondary_by_id = {record.sample_id: record for record in secondary.records}
    details: list[AdjudicationRecordAudit] = []
    for record in adjudication.records:
        if record.sample_id not in primary_by_id:
            raise ValueError(f"adjudication names unknown primary sample {record.sample_id}")
        if record.sample_id not in secondary_by_id:
            raise ValueError(f"adjudication names unknown secondary sample {record.sample_id}")
        if record.sample_id not in answers:
            raise ValueError(f"missing reference answer for {record.sample_id}")

        primary_requirements = primary_by_id[record.sample_id].answer_requirements
        secondary_record = secondary_by_id[record.sample_id]
        secondary_requirements = secondary_record.answer_requirements
        final = record.final_answer_requirements
        if not final:
            final_matches: FinalMatch = "no_answer"
        elif final == primary_requirements and final == secondary_requirements:
            final_matches = "both"
        elif final == primary_requirements:
            final_matches = "primary"
        elif final == secondary_requirements:
            final_matches = "secondary"
        else:
            final_matches = "neither"

        decision_consistent = {
            "use_primary": final_matches in {"primary", "both"},
            "use_secondary": final_matches in {"secondary", "both"},
            "merge_rewrite": bool(final),
            "equivalent": final_matches == "both",
            "confirm_no_substantive_answer": final_matches == "no_answer",
        }[record.source_decision]
        answer_present = bool(answers[record.sample_id].strip())
        status_matches = (bool(final) and answer_present) or (not final and not answer_present)
        details.append(
            AdjudicationRecordAudit(
                blind_id=record.blind_id,
                sample_id=record.sample_id,
                final_matches=final_matches,
                decision_consistent=decision_consistent,
                status_matches_reference=status_matches,
                supported_requirements=[
                    requirement_satisfied(requirement, answers[record.sample_id])
                    for requirement in final
                ],
            )
        )

    return AnswerAdjudicationAudit(
        records=len(details),
        requirements=sum(len(detail.supported_requirements) for detail in details),
        supported_requirements=sum(sum(detail.supported_requirements) for detail in details),
        status_matches=sum(detail.status_matches_reference for detail in details),
        decision_inconsistencies=sum(not detail.decision_consistent for detail in details),
        gold_labels_changed=sum(
            detail.final_matches not in {"primary", "both", "no_answer"} for detail in details
        ),
        details=details,
    )
