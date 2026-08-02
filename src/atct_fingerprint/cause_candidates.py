"""Retain supported, possible, rejected, and unexamined cause candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal, Sequence


CauseType = Literal[
    "value_conflict",
    "work_environment",
    "income",
    "health",
    "relationship",
    "job_fit",
    "family_constraint",
    "social_norm",
    "unknown",
]
CauseStatus = Literal["supported", "possible", "rejected", "not_examined"]

_CAUSE_RULES: tuple[tuple[CauseType, str, re.Pattern[str]], ...] = (
    (
        "value_conflict",
        "価値基準の不一致",
        re.compile(r"成功(?:の)?(?:OS|基準|判断基準)|価値基準|価値観|判断基準"),
    ),
    (
        "work_environment",
        "労働条件・職場環境",
        re.compile(
            r"労働条件|職場環境|勤務条件|長時間労働|リモートワーク|"
            r"育休制度|時短勤務|副業解禁|雇用制度|制度"
        ),
    ),
    ("income", "賃金・収入", re.compile(r"賃金|収入|給与|給料|報酬")),
    ("health", "健康", re.compile(r"健康|体調|睡眠|病気|心身|疲労")),
    (
        "relationship",
        "人間関係",
        re.compile(r"人間関係|上司|同僚|ハラスメント|孤立"),
    ),
    (
        "job_fit",
        "職務適性",
        re.compile(r"職務適性|適性|向いていない|仕事内容|仕事が合わない"),
    ),
    (
        "family_constraint",
        "家庭事情",
        re.compile(r"家庭|家族|育児|子育て|介護"),
    ),
    (
        "social_norm",
        "社会規範",
        re.compile(r"社会規範|世間|社会から与え|会社から与え|標準的な成功|普通は"),
    ),
)
_POSSIBLE = re.compile(r"かもしれない|可能性|あり得る|場合がある|一因")
_NONEXCLUSIVE = re.compile(r"だけ(?:が)?原因ではない|だけではない")
_REJECTED = re.compile(r"(?<!だけが)原因ではない|無関係|影響しない")
_SUPPORTED = re.compile(
    r"原因|理由|ため|によって|影響|制約|不一致|重要|関係する|左右する|"
    r"変える|可能にする|無視できない|もある|も含む"
)
_OUTCOME = re.compile(
    r"([^。！？]{1,50}?(?:転職したい|働き方を変えたい|苦しい|"
    r"合わない|続けられない|辞めたい|違和感))"
)


@dataclass(frozen=True)
class CauseCandidate:
    outcome: str
    cause_type: CauseType
    description: str
    status: CauseStatus
    evidence_spans: tuple[str, ...]
    sentence_indices: tuple[int, ...]
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CauseCandidateAnalysis:
    outcome: str
    candidates: tuple[CauseCandidate, ...]
    supported_count: int
    possible_count: int
    rejected_count: int
    not_examined_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_cause_candidates(sentences: Sequence[str]) -> CauseCandidateAnalysis:
    """Keep absent real-world causes visible without pretending they are supported."""

    outcome = ""
    for sentence in sentences:
        match = _OUTCOME.search(sentence)
        if match:
            outcome = match.group(1).strip(" 、，,")
            break
    outcome = outcome or "本文の表面問題"

    candidates: list[CauseCandidate] = []
    for cause_type, description, pattern in _CAUSE_RULES:
        records: list[tuple[int, str]] = []
        statuses: list[CauseStatus] = []
        for index, sentence in enumerate(sentences):
            if not pattern.search(sentence):
                continue
            records.append((index, sentence))
            if _NONEXCLUSIVE.search(sentence):
                statuses.append("possible")
            elif _REJECTED.search(sentence):
                statuses.append("rejected")
            elif _POSSIBLE.search(sentence):
                statuses.append("possible")
            elif _SUPPORTED.search(sentence):
                statuses.append("supported")
            else:
                statuses.append("possible")
        if not records:
            status: CauseStatus = "not_examined"
            confidence = 0.0
        elif "supported" in statuses:
            status = "supported"
            confidence = min(1.0, 0.72 + 0.10 * len(records))
        elif "possible" in statuses:
            status = "possible"
            confidence = min(0.78, 0.48 + 0.08 * len(records))
        else:
            status = "rejected"
            confidence = min(0.90, 0.65 + 0.08 * len(records))
        candidates.append(
            CauseCandidate(
                outcome=outcome,
                cause_type=cause_type,
                description=description,
                status=status,
                evidence_spans=tuple(item[1] for item in records),
                sentence_indices=tuple(item[0] for item in records),
                confidence=confidence,
            )
        )

    known_matches = [
        pattern
        for _, _, pattern in _CAUSE_RULES
        if any(pattern.search(sentence) for sentence in sentences)
    ]
    unknown_records: list[tuple[int, str]] = []
    cause_cue = re.compile(r"原因(?:は|が)|理由(?:は|が)|一因")
    for index, sentence in enumerate(sentences):
        if cause_cue.search(sentence) and not any(
            pattern.search(sentence) for pattern in known_matches
        ):
            unknown_records.append((index, sentence))
    if unknown_records:
        candidates.append(
            CauseCandidate(
                outcome=outcome,
                cause_type="unknown",
                description="未分類の原因",
                status="possible",
                evidence_spans=tuple(item[1] for item in unknown_records),
                sentence_indices=tuple(item[0] for item in unknown_records),
                confidence=0.45,
            )
        )

    return CauseCandidateAnalysis(
        outcome=outcome,
        candidates=tuple(candidates),
        supported_count=sum(item.status == "supported" for item in candidates),
        possible_count=sum(item.status == "possible" for item in candidates),
        rejected_count=sum(item.status == "rejected" for item in candidates),
        not_examined_count=sum(item.status == "not_examined" for item in candidates),
    )
