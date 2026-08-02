"""Shared evidence records for deterministic v0.7 semantic audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Pattern


@dataclass(frozen=True)
class EvidenceSpan:
    sentence_index: int
    text: str
    cue: str
    rule: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evidence_for_match(
    sentence: str,
    sentence_index: int,
    pattern: Pattern[str],
    *,
    rule: str,
    confidence: float,
) -> EvidenceSpan | None:
    """Return a traceable evidence record for the first rule match."""

    match = pattern.search(sentence)
    if not match:
        return None
    return EvidenceSpan(
        sentence_index=sentence_index,
        text=sentence,
        cue=match.group(0),
        rule=rule,
        confidence=float(max(0.0, min(1.0, confidence))),
    )


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("　。、，,：:「」『』()（）")
