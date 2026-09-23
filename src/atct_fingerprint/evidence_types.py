"""Transparent evidence-type classification for claims and explanations."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Sequence


EVIDENCE_STRENGTH = {
    "experimental_evidence": 1.0,
    "comparative_evidence": 0.80,
    "authority_claim": 0.65,
    "mechanistic_explanation": 0.60,
    "personal_experience": 0.45,
    "observational_claim": 0.35,
    "analogy": 0.25,
    "unsupported_assertion": 0.0,
}

_PATTERNS = (
    (
        "experimental_evidence",
        re.compile(
            r"実験|無作為|ランダム化|対照群|検証結果|測定結果|"
            r"\bp\s*[<=>]|信頼区間|サンプル数"
        ),
    ),
    (
        "comparative_evidence",
        re.compile(
            r"比較|対照|統制|より(?:高|低|多|少)|倍|差が(?:出|あ)"
        ),
    ),
    (
        "authority_claim",
        re.compile(
            r"研究(?:では|によると)|論文|専門家|大学|学会|"
            r"報告(?:では|によると)|引用|出典"
        ),
    ),
    (
        "personal_experience",
        re.compile(
            r"(?:私|僕|わたし|自分)(?:は|が|の経験)|"
            r"経験した|体験した|かつて|当時|ホームレス|"
            r"私自身|僕自身|身をもって"
        ),
    ),
    (
        "analogy",
        re.compile(
            r"たとえば|例えば|比喩|のよう(?:な|に)|みたい(?:な|に)|"
            r"いわば|にたとえる|喩え"
        ),
    ),
    (
        "mechanistic_explanation",
        re.compile(
            r"仕組み|機構|メカニズム|作用|因果|によって|ために|"
            r"その結果|ゆえに|だから|のである"
        ),
    ),
    (
        "observational_claim",
        re.compile(
            r"多くの人|一般に|しばしば|よく見られ|傾向がある|"
            r"場合が多い|珍しくない"
        ),
    ),
)


@dataclass(frozen=True)
class EvidenceRecord:
    sentence_index: int
    evidence_type: str
    cue: str
    strength: float
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAnalysis:
    records: tuple[EvidenceRecord, ...]
    counts: dict[str, int]
    mean_strength: float
    sentence_strengths: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_evidence_types(sentences: Sequence[str]) -> EvidenceAnalysis:
    """Assign one primary evidence type to each sentence.

    The classifier is deliberately deterministic and cue-based. It reports
    evidence form, not whether the underlying claim is true.
    """

    records: list[EvidenceRecord] = []
    for index, sentence in enumerate(sentences):
        evidence_type = "unsupported_assertion"
        cue = ""
        for candidate, pattern in _PATTERNS:
            match = pattern.search(sentence)
            if match:
                evidence_type = candidate
                cue = match.group(0)
                break
        records.append(
            EvidenceRecord(
                sentence_index=index,
                evidence_type=evidence_type,
                cue=cue,
                strength=EVIDENCE_STRENGTH[evidence_type],
                sentence=sentence,
            )
        )
    strengths = tuple(item.strength for item in records)
    counts = dict(Counter(item.evidence_type for item in records))
    return EvidenceAnalysis(
        records=tuple(records),
        counts=counts,
        mean_strength=(
            float(sum(strengths) / len(strengths)) if strengths else 0.0
        ),
        sentence_strengths=strengths,
    )
