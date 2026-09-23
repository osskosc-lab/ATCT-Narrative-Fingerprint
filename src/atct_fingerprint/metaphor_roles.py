"""Track changing functional roles of the same metaphor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_OS = re.compile(r"(?:成功)?OS|ＯＳ")
_ROLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("social_standard", re.compile(r"社会|会社|標準|与えられ|外部")),
    ("internalized_rule", re.compile(r"内面化|染み込|当たり前|自分の中|取り込")),
    ("reality_mismatch", re.compile(r"合わない|互換性|ずれ|不整合|機能しない|古い")),
    ("error_detector", re.compile(r"違和感|エラー|警告|検出|気づ")),
    ("rewrite_target", re.compile(r"書き換|更新する対象|変える対象|再構築")),
    ("self_revision_system", re.compile(r"自分で|修正でき|更新でき|選び直|定義し直")),
)
_VALUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stability", re.compile(r"安定")),
    ("freedom", re.compile(r"自由")),
    ("income", re.compile(r"収入|賃金|給与")),
    ("family", re.compile(r"家族|家庭|育児")),
    ("approval", re.compile(r"承認|評価")),
    ("meaning", re.compile(r"意味|意義")),
    ("health", re.compile(r"健康|心身|体調")),
)


@dataclass(frozen=True)
class MetaphorRole:
    label: str
    role: str
    section_index: int
    evidence_span: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetaphorRoleAnalysisV07:
    roles: tuple[MetaphorRole, ...]
    unique_roles: tuple[str, ...]
    transition_score: float
    compressed_values: tuple[str, ...]
    overcompression_score: float
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_metaphor_roles_v07(
    sentences: Sequence[str],
) -> MetaphorRoleAnalysisV07:
    roles: list[MetaphorRole] = []
    for index, sentence in enumerate(sentences):
        if not _OS.search(sentence):
            continue
        matched = False
        for role, pattern in _ROLES:
            if not pattern.search(sentence):
                continue
            matched = True
            roles.append(
                MetaphorRole(
                    label="OS",
                    role=role,
                    section_index=index,
                    evidence_span=sentence,
                    confidence=0.88,
                )
            )
        if not matched:
            roles.append(
                MetaphorRole(
                    label="OS",
                    role="mention",
                    section_index=index,
                    evidence_span=sentence,
                    confidence=0.60,
                )
            )
    unique = tuple(dict.fromkeys(item.role for item in roles if item.role != "mention"))
    transition = max(0.0, min(1.0, (len(unique) - 1) / 5.0))
    joined = "\n".join(sentences)
    values = tuple(name for name, pattern in _VALUES if pattern.search(joined))
    overcompression = 0.0
    if roles and len(values) >= 3:
        overcompression = min(1.0, (len(values) - 1) / 6.0)
    return MetaphorRoleAnalysisV07(
        roles=tuple(roles),
        unique_roles=unique,
        transition_score=float(transition),
        compressed_values=values,
        overcompression_score=float(overcompression),
        warning=("metaphor_overcompression" if overcompression >= 0.50 else ""),
    )
