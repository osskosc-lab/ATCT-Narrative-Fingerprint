"""Separate explanatory metaphors from causal reification claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_METAPHORS = {
    "OS": re.compile(r"OS|ＯＳ"),
    "application": re.compile(r"アプリ"),
    "foundation": re.compile(r"土台"),
    "energy": re.compile(r"エネルギー"),
    "leakage": re.compile(r"漏電"),
    "accelerator_brake": re.compile(r"アクセル|サイドブレーキ"),
}
_MARKER = re.compile(
    r"たとえば|例えば|比喩|のよう(?:な|に)|みたい|いわば"
)
_CAUSAL = re.compile(
    r"整えれば|直せば|変わる|動き始める|解決する|できる|"
    r"原因|によって|必ず|保証|動かない|壊れ"
)
_MAPPING = re.compile(r"とは|である|だ[。！？]|がある|を持つ|に相当")
_CONCRETE = re.compile(
    r"睡眠|注意|実行機能|心拍|行動回数|測定|尺度|不安|怒り|"
    r"作業記憶|認知負荷|休息|栄養|状態"
)


@dataclass(frozen=True)
class MetaphorClaim:
    sentence_index: int
    metaphor: str
    stage: str
    marker: str
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetaphorAudit:
    claims: tuple[MetaphorClaim, ...]
    explanatory_count: int
    mapping_count: int
    causal_guarantee_count: int
    reification_score: float
    specificity_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_metaphors(sentences: Sequence[str]) -> MetaphorAudit:
    claims: list[MetaphorClaim] = []
    concrete_count = 0
    for index, sentence in enumerate(sentences):
        if _CONCRETE.search(sentence):
            concrete_count += 1
        for metaphor, pattern in _METAPHORS.items():
            if not pattern.search(sentence):
                continue
            marker_match = _MARKER.search(sentence)
            if marker_match:
                stage = "explanatory_metaphor"
                marker = marker_match.group(0)
            elif _CAUSAL.search(sentence):
                stage = "causal_guarantee"
                marker = ""
            elif _MAPPING.search(sentence):
                stage = "mapping_claim"
                marker = ""
            else:
                stage = "metaphor_mention"
                marker = ""
            claims.append(
                MetaphorClaim(
                    sentence_index=index,
                    metaphor=metaphor,
                    stage=stage,
                    marker=marker,
                    sentence=sentence,
                )
            )
    explanatory = sum(
        item.stage == "explanatory_metaphor" for item in claims
    )
    mapping = sum(item.stage == "mapping_claim" for item in claims)
    causal = sum(item.stage == "causal_guarantee" for item in claims)
    reification_score = (
        float((0.5 * mapping + causal) / len(claims)) if claims else 0.0
    )
    return MetaphorAudit(
        claims=tuple(claims),
        explanatory_count=explanatory,
        mapping_count=mapping,
        causal_guarantee_count=causal,
        reification_score=reification_score,
        specificity_score=(
            float(concrete_count / len(sentences)) if sentences else 0.0
        ),
    )
