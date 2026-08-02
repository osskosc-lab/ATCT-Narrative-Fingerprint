"""Measure whether one deep interpretation erases competing causes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .cause_candidates import CauseCandidate, CauseCandidateAnalysis


@dataclass(frozen=True)
class CauseCompetition:
    primary_cause_type: str
    primary_interpretation: str
    primary_confidence: float
    primary_exclusive: bool
    competing_causes: tuple[str, ...]
    retained_cause_types: tuple[str, ...]
    cause_monopoly: float
    warning: str
    evidence_spans: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_cause_competition(
    analysis: CauseCandidateAnalysis,
) -> CauseCompetition:
    active = [
        item
        for item in analysis.candidates
        if item.status in {"supported", "possible"} and item.confidence > 0.0
    ]
    if not active:
        return CauseCompetition(
            primary_cause_type="",
            primary_interpretation="",
            primary_confidence=0.0,
            primary_exclusive=False,
            competing_causes=(),
            retained_cause_types=(),
            cause_monopoly=0.0,
            warning="",
            evidence_spans=(),
        )
    primary = max(
        active,
        key=lambda item: (
            item.cause_type == "value_conflict",
            item.confidence,
            -item.sentence_indices[0] if item.sentence_indices else 0,
        ),
    )
    total = sum(item.confidence for item in active)
    monopoly = primary.confidence / total if total else 0.0
    competing = tuple(
        item.description for item in active if item is not primary
    )
    exclusive = monopoly >= 0.85 and len(active) == 1
    return CauseCompetition(
        primary_cause_type=primary.cause_type,
        primary_interpretation=primary.description,
        primary_confidence=primary.confidence,
        primary_exclusive=exclusive,
        competing_causes=competing,
        retained_cause_types=tuple(item.cause_type for item in active),
        cause_monopoly=float(monopoly),
        warning=(
            "single_cause_overcompression"
            if monopoly >= 0.70 or len(active) < 3
            else ""
        ),
        evidence_spans=tuple(
            span
            for item in active
            for span in item.evidence_spans
        ),
    )
