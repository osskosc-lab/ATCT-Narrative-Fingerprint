"""Extract rejected and adopted causes while retaining source evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

from .evidence_types import EvidenceAnalysis


_CAUSES = (
    ("怠け", re.compile(r"怠け(?:者)?")),
    ("才能不足", re.compile(r"才能(?:が)?(?:足りない|不足)")),
    ("努力不足", re.compile(r"努力(?:が)?(?:足りない|不足)")),
    ("意志の弱さ", re.compile(r"意志(?:が)?弱い|意志の弱さ")),
    (
        "人格の欠陥",
        re.compile(r"性格(?:の問題|が悪い)|自分がダメ|人格"),
    ),
    (
        "努力する順番の誤り",
        re.compile(
            r"努力(?:する)?(?:の)?順番(?:の誤り|が違う|を間違)|"
            r"順番(?:の誤り|が違う|を間違|ミス)"
        ),
    ),
    (
        "行動できる状態の未整備",
        re.compile(
            r"行動できる状態(?:ではない|が整っていない)|状態の未整備"
        ),
    ),
    (
        "土台の崩壊",
        re.compile(r"土台(?:が崩|の崩壊|が整っていない)"),
    ),
    (
        "OSの不調",
        re.compile(r"(?:人生|脳)?の?OS(?:の不調|が壊|が動かない)"),
    ),
    ("環境制約", re.compile(r"環境(?:の)?制約|環境が原因")),
    (
        "方法の誤り",
        re.compile(r"方法(?:が違う|の誤り)|やり方(?:が違う|の問題)"),
    ),
)

_OUTCOME = re.compile(
    r"([^。！？]{1,45}?(?:変わらない|続かない|行動できない|"
    r"うまくいかない|報われない|苦しい|動けない))"
)
_NEGATION = re.compile(
    r"ではない|じゃない|せいではない|原因ではない"
)
_ADOPTION_CUE = re.compile(
    r"本当の原因|原因は|問題は|理由は|実は|ではなく|じゃなく"
)
_COPULA_END = re.compile(
    r"(?:なの)?(?:だ|です|である|だった|かもしれない|にある)"
    r"[。！？]?$"
)


def _known_causes(text: str) -> list[str]:
    return [name for name, pattern in _CAUSES if pattern.search(text)]


def _generic_after_cue(text: str) -> str:
    match = re.search(
        r"(?:本当の原因|原因|問題|理由)(?:は|が)\s*"
        r"([^。！？]{1,45})",
        text,
    )
    if not match:
        parts = re.split(r"ではなく|じゃなく", text, maxsplit=1)
        value = parts[-1] if len(parts) == 2 else ""
    else:
        value = match.group(1)
    value = _COPULA_END.sub("", value.strip(" 、,「」『』"))
    return value[:45]


@dataclass(frozen=True)
class CausalSubstitution:
    outcome: str
    rejected_causes: tuple[str, ...]
    adopted_cause: str
    evidence_types: tuple[str, ...]
    source_indices: tuple[int, ...]
    contrast_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalFrameAnalysis:
    substitutions: tuple[CausalSubstitution, ...]
    rejected_cause_count: int
    adopted_cause_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_causal_frames(
    sentences: Sequence[str],
    evidence: EvidenceAnalysis,
) -> CausalFrameAnalysis:
    """Find cause replacement chains, including cross-sentence replacements."""

    outcome = ""
    pending_rejected: list[str] = []
    pending_indices: list[int] = []
    substitutions: list[CausalSubstitution] = []

    for index, sentence in enumerate(sentences):
        outcome_match = _OUTCOME.search(sentence)
        if outcome_match and not outcome:
            outcome = outcome_match.group(1).strip(" 、,")

        before_after = re.split(r"ではなく|じゃなく", sentence, maxsplit=1)
        same_sentence_rejected: list[str] = []
        same_sentence_adopted: list[str] = []
        if len(before_after) == 2:
            same_sentence_rejected = _known_causes(before_after[0])
            same_sentence_adopted = _known_causes(before_after[1])

        if _NEGATION.search(sentence) or same_sentence_rejected:
            rejected = _known_causes(sentence)
            if same_sentence_adopted:
                rejected = [
                    item for item in rejected if item not in same_sentence_adopted
                ]
            for item in rejected:
                if item not in pending_rejected:
                    pending_rejected.append(item)
            if rejected and index not in pending_indices:
                pending_indices.append(index)

        adopted = same_sentence_adopted
        if not adopted and _ADOPTION_CUE.search(sentence):
            known = _known_causes(sentence)
            adopted = [item for item in known if item not in pending_rejected]
        adopted_cause = adopted[-1] if adopted else ""
        if not adopted_cause and _ADOPTION_CUE.search(sentence):
            candidate = _generic_after_cue(sentence)
            if candidate and not _NEGATION.search(candidate):
                adopted_cause = candidate

        if not adopted_cause or not pending_rejected:
            continue
        source_indices = tuple(dict.fromkeys((*pending_indices, index)))
        evidence_types = tuple(
            dict.fromkeys(
                evidence.records[source_index].evidence_type
                for source_index in source_indices
                if source_index < len(evidence.records)
            )
        )
        substitutions.append(
            CausalSubstitution(
                outcome=outcome or "明示されていない結果",
                rejected_causes=tuple(pending_rejected),
                adopted_cause=adopted_cause,
                evidence_types=evidence_types,
                source_indices=source_indices,
                contrast_score=(
                    1.0 if adopted_cause not in pending_rejected else 0.0
                ),
            )
        )
        pending_rejected = []
        pending_indices = []

    return CausalFrameAnalysis(
        substitutions=tuple(substitutions),
        rejected_cause_count=sum(
            len(item.rejected_causes) for item in substitutions
        ),
        adopted_cause_count=len(substitutions),
    )
