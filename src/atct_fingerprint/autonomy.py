"""Separate subjective ownership from operational autonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_CONDITIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reason_explanation", re.compile(r"選択理由|理由を説明|なぜ選んだ")),
    ("tradeoff_awareness", re.compile(r"失うもの|代償|犠牲|トレードオフ")),
    ("alternative_comparison", re.compile(r"他の選択肢|別の選択肢|比較して|比較する")),
    ("revisability", re.compile(r"再検討|修正でき|見直せ|状況変化|更新でき")),
    (
        "external_self_distinction",
        re.compile(
            r"(?:外部評価|会社(?:や社会)?の基準|社会の基準|他人の評価)"
            r".{0,30}自分の基準|"
            r"自分の基準.{0,30}(?:外部評価|会社|社会|他人)"
        ),
    ),
)
_SUBJECTIVE = re.compile(r"納得感|自分で選んだと思|自分で決めた気が")


@dataclass(frozen=True)
class AutonomyCondition:
    name: str
    satisfied: bool
    sentence_index: int | None
    evidence_span: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutonomyAnalysis:
    conditions: tuple[AutonomyCondition, ...]
    autonomy_score: float
    subjective_ownership_detected: bool
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_autonomy(sentences: Sequence[str]) -> AutonomyAnalysis:
    conditions: list[AutonomyCondition] = []
    for name, pattern in _CONDITIONS:
        record: tuple[int, str] | None = None
        for index, sentence in enumerate(sentences):
            if pattern.search(sentence):
                record = (index, sentence)
                break
        conditions.append(
            AutonomyCondition(
                name=name,
                satisfied=record is not None,
                sentence_index=(record[0] if record else None),
                evidence_span=(record[1] if record else ""),
                confidence=(0.90 if record else 0.0),
            )
        )
    score = sum(item.satisfied for item in conditions) / len(conditions)
    subjective = any(_SUBJECTIVE.search(sentence) for sentence in sentences)
    return AutonomyAnalysis(
        conditions=tuple(conditions),
        autonomy_score=float(score),
        subjective_ownership_detected=subjective,
        warning=(
            "subjective_ownership_without_operational_autonomy"
            if subjective and score < 0.60
            else ""
        ),
    )
