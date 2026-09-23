"""Detect Shadow-Seeking-Transformation-New Shadow recurrence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_CRITERIA: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "initial_model_failure",
        re.compile(
            r"Shadow|シャドウ|(?:^|[^影])影(?:$|[^響])|"
            r"機能しない|不整合|古いOS|合わなくな"
        ),
    ),
    ("exploration", re.compile(r"Seeking|探索|試行|試す|問い直|探し始め")),
    (
        "new_criteria",
        re.compile(r"Transformation|変容|新しい判断基準|自分で定義|書き換"),
    ),
    (
        "provisionality",
        re.compile(r"暫定|完成ではない|修正でき|更新し続け|固定しない|再検討"),
    ),
    (
        "next_shadow",
        re.compile(r"(?:また|次の|新たな|新しい).{0,12}(?:Shadow|シャドウ|影|違和感|不整合)"),
    ),
)


@dataclass(frozen=True)
class CycleCriterion:
    name: str
    detected: bool
    sentence_index: int | None
    evidence_span: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecursiveCycleAnalysis:
    criteria: tuple[CycleCriterion, ...]
    score: float
    detected: bool
    path: tuple[str, ...]
    structure_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_recursive_cycle(sentences: Sequence[str]) -> RecursiveCycleAnalysis:
    criteria: list[CycleCriterion] = []
    for name, pattern in _CRITERIA:
        record: tuple[int, str] | None = None
        for index, sentence in enumerate(sentences):
            if pattern.search(sentence):
                record = (index, sentence)
                break
        criteria.append(
            CycleCriterion(
                name=name,
                detected=record is not None,
                sentence_index=(record[0] if record else None),
                evidence_span=(record[1] if record else ""),
                confidence=(0.90 if record else 0.0),
            )
        )
    count = sum(item.detected for item in criteria)
    has_next_shadow = next(
        item.detected for item in criteria if item.name == "next_shadow"
    )
    detected = count >= 4 and has_next_shadow
    return RecursiveCycleAnalysis(
        criteria=tuple(criteria),
        score=float(count / len(criteria)),
        detected=detected,
        path=(
            ("Shadow", "Seeking", "Transformation", "New Shadow")
            if detected
            else ("Shadow", "Seeking", "Transformation")
        ),
        structure_type=(
            "recursive_transformation_cycle"
            if detected
            else "stepwise_transformation"
        ),
    )
