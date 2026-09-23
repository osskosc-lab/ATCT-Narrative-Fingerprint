"""Detect institution-to-agency-to-meaning causal bridges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_ROLE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "institution",
        re.compile(
            r"制度|育休|時短勤務|副業解禁|リモートワーク|雇用|労働条件"
        ),
    ),
    (
        "exploration_possibility",
        re.compile(r"探索|試せる|試行|可能になる|選択肢が増|選べる|余地が生ま"),
    ),
    (
        "individual_agency",
        re.compile(r"自分で選|本人が選|選択する|判断する|決め直|問い直"),
    ),
    (
        "meaning_reconstruction",
        re.compile(
            r"意味(?:を|が)?(?:再構築|変わ|作り直)|成功(?:の)?(?:基準|判断基準)|"
            r"価値基準|判断体系|書き換|再定義"
        ),
    ),
)
_EXPECTED = (
    "institution",
    "exploration_possibility",
    "individual_agency",
    "meaning_reconstruction",
)


@dataclass(frozen=True)
class CausalLayerEvent:
    sentence_index: int
    role: str
    cue: str
    sentence: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalLayerAnalysis:
    events: tuple[CausalLayerEvent, ...]
    path: tuple[str, ...]
    coverage: float
    order_concordance: float
    bridge_score: float
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_causal_layers(sentences: Sequence[str]) -> CausalLayerAnalysis:
    """Score explicit layer coverage and expected causal order separately."""

    events: list[CausalLayerEvent] = []
    indices_by_role: dict[str, list[int]] = {}
    for index, sentence in enumerate(sentences):
        for role, pattern in _ROLE_RULES:
            match = pattern.search(sentence)
            if not match:
                continue
            events.append(
                CausalLayerEvent(
                    sentence_index=index,
                    role=role,
                    cue=match.group(0),
                    sentence=sentence,
                    confidence=0.90,
                )
            )
            indices_by_role.setdefault(role, []).append(index)

    present = tuple(role for role in _EXPECTED if role in indices_by_role)
    coverage = len(present) / len(_EXPECTED)
    cursor = -1
    ordered_length = 0
    for role in _EXPECTED:
        next_indices = [
            value for value in indices_by_role.get(role, []) if value > cursor
        ]
        if not next_indices:
            break
        cursor = min(next_indices)
        ordered_length += 1
    concordance = max(0.0, (ordered_length - 1) / 3.0)
    bridge = coverage * concordance
    observed_path = tuple(
        sorted(present, key=lambda role: min(indices_by_role[role]))
    )
    return CausalLayerAnalysis(
        events=tuple(events),
        path=observed_path,
        coverage=float(coverage),
        order_concordance=float(concordance),
        bridge_score=float(bridge),
        warning=(
            "institution_individual_bridge_unsupported"
            if coverage >= 0.50 and bridge < 0.50
            else ""
        ),
    )
