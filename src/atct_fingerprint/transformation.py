"""Operational audit of claimed transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_VARIABLE = re.compile(
    r"成功(?:の)?(?:判断基準|基準|OS)|価値基準|判断体系|判断主体"
)
_INITIAL = re.compile(
    r"会社(?:や社会)?から与えられ|社会(?:や会社)?が決め|"
    r"外部(?:の)?評価|以前は|古い(?:成功)?OS|他人の基準"
)
_FINAL = re.compile(
    r"自分で(?:定義|選び直|決め直|書き換|更新)|自分の基準|本人が定義"
)
_OBSERVABLE = re.compile(
    r"理由(?:を)?説明でき|失うもの|代償|行動(?:で|を)|"
    r"観測|確認|測定|指標|選択肢と比較|記録"
)
_REVERSIBLE = re.compile(r"再検討|修正でき|更新でき|見直せ|状況変化|可逆")
_NEXT_SHADOW = re.compile(
    r"(?:また|次の|新たな|新しい).{0,12}(?:Shadow|シャドウ|影|違和感|不整合)|"
    r"再び.{0,12}(?:機能不全|合わなく)"
)
_GENERIC_CHANGE = re.compile(r"世界が変わる|人生が動き出す|人生が変わる")


@dataclass(frozen=True)
class TransformationFrame:
    initial_state: str
    changed_state: str
    changed_variable: str
    observable_marker: str | None
    reversibility: str | None
    next_shadow_condition: str | None
    evidence_spans: tuple[str, ...]
    sentence_indices: tuple[int, ...]
    operationality: float
    component_scores: dict[str, float]
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _first(
    sentences: Sequence[str], pattern: re.Pattern[str]
) -> tuple[int, str, str] | None:
    for index, sentence in enumerate(sentences):
        match = pattern.search(sentence)
        if match:
            return index, sentence, match.group(0)
    return None


def analyze_transformation(sentences: Sequence[str]) -> TransformationFrame:
    variable = _first(sentences, _VARIABLE)
    initial = _first(sentences, _INITIAL)
    final = _first(sentences, _FINAL)
    observable = _first(sentences, _OBSERVABLE)
    reversible = _first(sentences, _REVERSIBLE)
    next_shadow = _first(sentences, _NEXT_SHADOW)
    generic = _first(sentences, _GENERIC_CHANGE)

    scores = {
        "variable": 1.0 if variable else 0.0,
        "initial_state": 1.0 if initial else 0.0,
        "changed_state": 1.0 if final else 0.0,
        "observable_marker": 1.0 if observable else 0.0,
        "rechange_condition": (
            1.0 if next_shadow else (0.8 if reversible else 0.0)
        ),
    }
    if generic and not variable:
        scores["changed_state"] = min(scores["changed_state"], 0.20)
    records = [
        item
        for item in (variable, initial, final, observable, reversible, next_shadow)
        if item is not None
    ]
    operationality = sum(scores.values()) / 5.0
    return TransformationFrame(
        initial_state=(initial[2] if initial else ""),
        changed_state=(final[2] if final else ""),
        changed_variable=(variable[2] if variable else ""),
        observable_marker=(observable[2] if observable else None),
        reversibility=(reversible[2] if reversible else None),
        next_shadow_condition=(next_shadow[2] if next_shadow else None),
        evidence_spans=tuple(dict.fromkeys(item[1] for item in records)),
        sentence_indices=tuple(dict.fromkeys(item[0] for item in records)),
        operationality=float(operationality),
        component_scores=scores,
    )
