"""Audit whether causal responsibility moves from person to state or method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .causal_frames import CausalFrameAnalysis


_CATEGORY_PATTERNS = (
    (
        "personality",
        re.compile(
            r"怠け|才能|努力不足|意志|性格|人格|根性|自分がダメ"
        ),
    ),
    (
        "state",
        re.compile(r"状態|疲労|睡眠|身体|脳|不安|怒り|土台|OS"),
    ),
    ("method", re.compile(r"順番|方法|手順|やり方")),
    ("environment", re.compile(r"環境|制約|資源|時間|貧困|制度")),
)


def cause_category(cause: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(cause):
            return category
    return "other"


@dataclass(frozen=True)
class ResponsibilityShift:
    outcome: str
    rejected_cause: str
    rejected_category: str
    adopted_cause: str
    adopted_category: str
    relief: float
    source_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResponsibilityAnalysis:
    shifts: tuple[ResponsibilityShift, ...]
    relief_score: float
    interpretation_warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_responsibility(
    causal_frames: CausalFrameAnalysis,
) -> ResponsibilityAnalysis:
    shifts: list[ResponsibilityShift] = []
    for substitution in causal_frames.substitutions:
        adopted_category = cause_category(substitution.adopted_cause)
        for rejected in substitution.rejected_causes:
            rejected_category = cause_category(rejected)
            relief = (
                1.0
                if rejected_category == "personality"
                and adopted_category in {"state", "method", "environment"}
                else 0.0
            )
            shifts.append(
                ResponsibilityShift(
                    outcome=substitution.outcome,
                    rejected_cause=rejected,
                    rejected_category=rejected_category,
                    adopted_cause=substitution.adopted_cause,
                    adopted_category=adopted_category,
                    relief=relief,
                    source_indices=substitution.source_indices,
                )
            )
    relief_score = (
        float(sum(item.relief for item in shifts) / len(shifts))
        if shifts
        else 0.0
    )
    return ResponsibilityAnalysis(
        shifts=tuple(shifts),
        relief_score=relief_score,
        interpretation_warning=(
            "Responsibility relief may be an appropriate reappraisal, a "
            "persuasion technique, or pre-offer framing; the score does not "
            "assign moral value."
        ),
    )
