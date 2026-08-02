"""Audit exclusive title reframing against the scope retained in the body."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

from .evidence_spans import EvidenceSpan


_NONEXCLUSIVE = re.compile(
    r"だけ(?:が原因)?ではない|だけじゃない|だけでなく|だけに限らない"
)
_EXCLUSIVE = re.compile(
    r"(?<!だけ)(?:んじゃない|のではない|ではない|じゃない)"
)
_ALTERNATIVES = re.compile(
    r"労働条件|職場環境|勤務|賃金|収入|給与|健康|体調|"
    r"人間関係|職務適性|家庭|育児|介護|雇用|制度|"
    r"複数|同時に|一方で|場合も|だけではない"
)


@dataclass(frozen=True)
class TitleBodyScopeAudit:
    title_claim: str
    body_claim: str
    title_exclusion: float
    body_exclusion: float
    body_alternative_support: float
    mismatch: float
    warning: str
    reason: str
    suggested_revision: str
    evidence_spans: tuple[EvidenceSpan, ...]
    evaluation_status: str = "pending_annotated_corpus_macro_f1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _claim_kind(text: str) -> tuple[str, float]:
    if not text:
        return "absent", 0.0
    if _NONEXCLUSIVE.search(text):
        return "nonexclusive", 0.0
    if _EXCLUSIVE.search(text):
        return "exclusive", 1.0
    return "unspecified", 0.25


def audit_title_body_scope(
    title: str,
    sentences: Sequence[str],
) -> TitleBodyScopeAudit:
    """Distinguish `Xではない` from `Xだけではない` deterministically."""

    title_kind, title_exclusion = _claim_kind(title)
    body_exclusive: list[tuple[int, str]] = []
    body_nonexclusive: list[tuple[int, str]] = []
    alternative_records: list[tuple[int, str, str]] = []
    for index, sentence in enumerate(sentences):
        kind, _ = _claim_kind(sentence)
        if kind == "exclusive":
            body_exclusive.append((index, sentence))
        elif kind == "nonexclusive":
            body_nonexclusive.append((index, sentence))
        for match in _ALTERNATIVES.finditer(sentence):
            alternative_records.append((index, sentence, match.group(0)))

    if body_nonexclusive or len({item[2] for item in alternative_records}) >= 2:
        body_exclusion = 0.0
        body_claim = "複数原因を保持する非排他的説明"
    elif body_exclusive:
        body_exclusion = 1.0
        body_claim = body_exclusive[0][1]
    else:
        body_exclusion = 0.35
        body_claim = "排他範囲は本文で明示されていない"

    unique_alternatives = tuple(dict.fromkeys(item[2] for item in alternative_records))
    body_alternative_support = min(1.0, len(unique_alternatives) / 3.0)
    if body_nonexclusive:
        body_alternative_support = max(body_alternative_support, 0.75)
    mismatch = max(0.0, title_exclusion - body_exclusion) * max(
        0.35, body_alternative_support
    )

    evidence: list[EvidenceSpan] = []
    if title:
        evidence.append(
            EvidenceSpan(
                sentence_index=-1,
                text=title,
                cue=(
                    _NONEXCLUSIVE.search(title).group(0)
                    if _NONEXCLUSIVE.search(title)
                    else (
                        _EXCLUSIVE.search(title).group(0)
                        if _EXCLUSIVE.search(title)
                        else title
                    )
                ),
                rule=f"title_{title_kind}",
                confidence=1.0 if title_kind in {"exclusive", "nonexclusive"} else 0.5,
            )
        )
    for index, sentence in body_nonexclusive[:2]:
        match = _NONEXCLUSIVE.search(sentence)
        evidence.append(
            EvidenceSpan(
                sentence_index=index,
                text=sentence,
                cue=match.group(0) if match else sentence,
                rule="body_nonexclusive",
                confidence=0.95,
            )
        )
    for index, sentence, cue in alternative_records[:6]:
        evidence.append(
            EvidenceSpan(
                sentence_index=index,
                text=sentence,
                cue=cue,
                rule="body_alternative_cause",
                confidence=0.80,
            )
        )

    suggested = ""
    if mismatch >= 0.50:
        suggested = re.sub(
            r"(?<!だけ)(?:んじゃない|のではない|ではない|じゃない)",
            "だけではない",
            title,
            count=1,
        )
    return TitleBodyScopeAudit(
        title_claim=title,
        body_claim=body_claim,
        title_exclusion=title_exclusion,
        body_exclusion=body_exclusion,
        body_alternative_support=body_alternative_support,
        mismatch=float(min(1.0, mismatch)),
        warning=("title_body_scope_mismatch" if mismatch >= 0.50 else ""),
        reason=(
            "タイトルは表面欲求を排他的に否定するが、本文は競合原因を保持する"
            if mismatch >= 0.50
            else "タイトルと本文の排他範囲に重大な差は検出されない"
        ),
        suggested_revision=suggested,
        evidence_spans=tuple(evidence),
    )
