"""Ending-state diagnostics that separate recognition from executed action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence


_RECOGNITION = re.compile(
    r"気づ|分か|理解|認め|見抜|自覚|問題|正義|悪意|"
    r"自分を|自分の判断|意味していた"
)
_EXECUTION = re.compile(
    r"開いた|書いた|向けた|始めた|直した|出席した|"
    r"確かめた|実行した"
)
_DEFERRED = re.compile(
    r"だろう|かもしれない|まだ|いつか|決めていない|次に|"
    r"もし|予定している|つもり|保留|そのうち"
)
_OPEN = re.compile(r"[?？]|問い|続(?:く|け)|旅を始め|これから|次の")
_NEGATED_EXECUTION = re.compile(
    r"開かない|書かない|向けられない|できない"
)


@dataclass(frozen=True)
class ClosureAnalysis:
    recognition_score: float
    execution_score: float
    deferment_score: float
    openness_score: float
    action_distance: float
    primary_state: str
    states: tuple[str, ...]
    recognition_indices: tuple[int, ...]
    execution_indices: tuple[int, ...]
    deferment_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score(indices: Sequence[int], scope: int) -> float:
    return float(min(1.0, len(indices) / max(scope, 1)))


def analyze_closure(sentences: Sequence[str]) -> ClosureAnalysis:
    if not sentences:
        return ClosureAnalysis(
            0,
            0,
            0,
            0,
            0,
            "disconnected",
            ("disconnected",),
            (),
            (),
            (),
        )
    closing_start = max(0, math.floor(len(sentences) * 0.60))
    recognition = tuple(
        index
        for index, sentence in enumerate(sentences)
        if _RECOGNITION.search(sentence)
    )
    execution = tuple(
        index
        for index in range(closing_start, len(sentences))
        if _EXECUTION.search(sentences[index])
        and not _NEGATED_EXECUTION.search(sentences[index])
    )
    deferred = tuple(
        index
        for index in range(closing_start, len(sentences))
        if _DEFERRED.search(sentences[index])
    )
    open_indices = tuple(
        index
        for index in range(closing_start, len(sentences))
        if _OPEN.search(sentences[index])
    )
    closing_scope = max(len(sentences) - closing_start, 1)
    recognition_score = _score(recognition, max(len(sentences) // 3, 1))
    execution_score = _score(execution, closing_scope)
    deferment_score = _score(deferred, closing_scope)
    openness_score = _score(open_indices, closing_scope)
    action_distance = float(max(recognition_score - execution_score, 0.0))

    if execution:
        states = ("resolved_closure",)
        primary = "resolved_closure"
    elif recognition and deferred:
        states = ("recognition_closure", "deferred_closure")
        primary = "deferred_closure"
    elif recognition:
        states = ("recognition_closure",)
        primary = "recognition_closure"
    elif open_indices:
        states = ("open_spiral",)
        primary = "open_spiral"
    else:
        states = ("disconnected",)
        primary = "disconnected"
    return ClosureAnalysis(
        recognition_score=recognition_score,
        execution_score=execution_score,
        deferment_score=deferment_score,
        openness_score=openness_score,
        action_distance=action_distance,
        primary_state=primary,
        states=states,
        recognition_indices=recognition,
        execution_indices=execution,
        deferment_indices=deferred,
    )
