"""Track certainty escalation and unsupported subject-scope expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

from .evidence_types import EvidenceAnalysis


_MODALITY = (
    ("certainty", 1.0, re.compile(r"必ず|絶対|唯一|例外なく")),
    (
        "possibility",
        0.15,
        re.compile(
            r"かもしれない|だとしたら|可能性|のではないか|あり得る"
        ),
    ),
    ("viewpoint", 0.30, re.compile(r"一つの見方|と考えられる|といえる")),
    ("tendency", 0.45, re.compile(r"傾向がある|場合がある|ことがある")),
    ("frequent", 0.60, re.compile(r"多い|一般的|当然")),
    (
        "assertion",
        0.75,
        re.compile(r"である|なのだ|だ[。！？]|です[。！？]"),
    ),
)
_SCOPE = (
    ("universal", 1.0, re.compile(r"人生は|人間は|すべての人|誰もが")),
    ("group", 0.75, re.compile(r"多くの人|人々|みんな|頑張っている人")),
    ("reader", 0.50, re.compile(r"あなた|読者")),
    ("personal", 0.25, re.compile(r"私|僕|わたし|自分の経験|昔の僕")),
)


@dataclass(frozen=True)
class ModalityRecord:
    sentence_index: int
    label: str
    certainty: float
    cue: str
    evidence_strength: float
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScopeRecord:
    sentence_index: int
    label: str
    scope: float
    cue: str
    evidence_strength: float
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModalityAnalysis:
    records: tuple[ModalityRecord, ...]
    scope_records: tuple[ScopeRecord, ...]
    opening_certainty: float
    closing_certainty: float
    certainty_escalation: float
    evidence_gain: float
    unsupported_escalation: float
    scope_expansion: float
    unsupported_scope_expansion: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def analyze_modality(
    sentences: Sequence[str],
    evidence: EvidenceAnalysis,
    *,
    body_indices: Sequence[int] | None = None,
) -> ModalityAnalysis:
    records: list[ModalityRecord] = []
    scopes: list[ScopeRecord] = []
    for index, sentence in enumerate(sentences):
        label, certainty, cue = "unmarked", 0.5, ""
        for candidate, level, pattern in _MODALITY:
            match = pattern.search(sentence)
            if match:
                label, certainty, cue = candidate, level, match.group(0)
                break
        strength = (
            evidence.sentence_strengths[index]
            if index < len(evidence.sentence_strengths)
            else 0.0
        )
        records.append(
            ModalityRecord(index, label, certainty, cue, strength, sentence)
        )
        for scope_label, scope_level, pattern in _SCOPE:
            match = pattern.search(sentence)
            if match:
                scopes.append(
                    ScopeRecord(
                        index,
                        scope_label,
                        scope_level,
                        match.group(0),
                        strength,
                        sentence,
                    )
                )
                break

    selected = (
        list(body_indices)
        if body_indices is not None
        else list(range(len(sentences)))
    )
    if not selected:
        selected = list(range(len(sentences)))
    edge = max(1, round(len(selected) * 0.35))
    opening_indices = selected[:edge]
    closing_indices = selected[-edge:]
    opening_certainty = _mean([records[index].certainty for index in opening_indices])
    closing_certainty = _mean([records[index].certainty for index in closing_indices])
    opening_evidence = _mean(
        [records[index].evidence_strength for index in opening_indices]
    )
    closing_evidence = _mean(
        [records[index].evidence_strength for index in closing_indices]
    )
    evidence_gain = closing_evidence - opening_evidence
    escalation = closing_certainty - opening_certainty

    selected_scopes = [item for item in scopes if item.sentence_index in selected]
    if selected_scopes:
        first_scope = selected_scopes[0].scope
        later_scope = max(item.scope for item in selected_scopes[1:] or selected_scopes)
        scope_expansion = max(later_scope - first_scope, 0.0)
    else:
        scope_expansion = 0.0
    return ModalityAnalysis(
        records=tuple(records),
        scope_records=tuple(scopes),
        opening_certainty=opening_certainty,
        closing_certainty=closing_certainty,
        certainty_escalation=escalation,
        evidence_gain=evidence_gain,
        unsupported_escalation=max(escalation - max(evidence_gain, 0.0), 0.0),
        scope_expansion=scope_expansion,
        unsupported_scope_expansion=max(
            scope_expansion - max(evidence_gain, 0.0), 0.0
        ),
    )
