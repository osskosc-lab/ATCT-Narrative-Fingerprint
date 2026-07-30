"""Detect ordered problem-relief-solution-offer persuasion funnels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

import numpy as np

from .causal_frames import CausalFrameAnalysis
from .document_layers import DocumentLayerAnalysis
from .history import EPSILON


_PAIN = re.compile(
    r"頑張っても|変わらない|続かない|動けない|苦しい|悩み|"
    r"疲れ(?:た|ている)|自分を責め|自己否定"
)
_RELIEF = re.compile(
    r"怠け者ではない|あなたのせいではない|悪くない|"
    r"責めなくていい|努力不足ではない|真面目(?:な|で)"
)
_HOPE = re.compile(
    r"変えられる|やり直せる|ここから変わる|希望|可能性|"
    r"動き始める|続けやすく"
)
_SOLUTION = re.compile(
    r"整え(?:る|直す)|解決策|第一歩|方法|手順|順番|"
    r"段階|ステップ|モデル"
)
_BRAND = re.compile(
    r"([一-龠々ぁ-んァ-ンA-Za-z0-9・]{2,24}"
    r"(?:術|法|メソッド|モデル|システム))"
)
_BRAND_CONTEXT = re.compile(r"独自|名付|という|でいう|「|『|®|™")

_MILESTONES = (
    "pain",
    "relief",
    "causal_substitution",
    "solution",
    "branded_solution",
    "offer",
)
_EMOTIONS = ("pain", "relief", "hope", "offer")


@dataclass(frozen=True)
class PersuasionEvent:
    sentence_index: int
    role: str
    strength: float
    cue: str
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BrandingTransition:
    sentence_index: int | None
    generic_concept: str
    branded_framework: str
    transition_strength: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FunnelAnalysis:
    events: tuple[PersuasionEvent, ...]
    observed_milestones: tuple[str, ...]
    coverage: float
    order_score: float
    persuasion_z: float
    funnel_score: float
    emotional_trajectory_score: float
    branding_transition: BrandingTransition
    offer_start_index: int | None
    seed: int
    shuffle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ordered_pair_ratio(positions: Sequence[int]) -> float:
    if len(positions) < 2:
        return 0.0
    concordant = 0
    pairs = 0
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            pairs += 1
            concordant += positions[left] < positions[right]
    return float(concordant / pairs)


def _first_by_role(events: Sequence[PersuasionEvent]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for event in events:
        positions.setdefault(event.role, event.sentence_index)
    return positions


def _brand_match(sentence: str) -> re.Match[str] | None:
    match = _BRAND.search(sentence)
    if match and _BRAND_CONTEXT.search(sentence):
        return match
    return None


def analyze_funnel(
    sentences: Sequence[str],
    causal_frames: CausalFrameAnalysis,
    layers: DocumentLayerAnalysis,
    *,
    shuffle_count: int = 200,
    seed: int = 42,
) -> FunnelAnalysis:
    if shuffle_count < 2:
        raise ValueError("shuffle_count must be at least two for a Z score")
    events: list[PersuasionEvent] = []
    adopted_indices = {
        item.source_indices[-1]
        for item in causal_frames.substitutions
        if item.source_indices
    }
    for index, sentence in enumerate(sentences):
        patterns = (
            ("pain", _PAIN, 1.0),
            ("relief", _RELIEF, 1.0),
            ("hope", _HOPE, 0.8),
            ("solution", _SOLUTION, 0.8),
        )
        for role, pattern, strength in patterns:
            if role == "solution" and index in adopted_indices:
                continue
            match = pattern.search(sentence)
            if match:
                events.append(
                    PersuasionEvent(
                        index, role, strength, match.group(0), sentence
                    )
                )
        brand = _brand_match(sentence)
        if brand:
            events.append(
                PersuasionEvent(
                    index,
                    "branded_solution",
                    1.0,
                    brand.group(1),
                    sentence,
                )
            )

    for substitution in causal_frames.substitutions:
        adopted_index = substitution.source_indices[-1]
        for rejected_index in substitution.source_indices[:-1]:
            events.append(
                PersuasionEvent(
                    rejected_index,
                    "rejected_cause",
                    0.8,
                    "; ".join(substitution.rejected_causes),
                    sentences[rejected_index],
                )
            )
        events.append(
            PersuasionEvent(
                adopted_index,
                "causal_substitution",
                1.0,
                substitution.adopted_cause,
                sentences[adopted_index],
            )
        )

    offer_indices = [
        item.sentence_index
        for item in layers.records
        if item.layer == "lead_magnet"
    ]
    for index in offer_indices:
        events.append(
            PersuasionEvent(index, "offer", 1.0, "lead_magnet", sentences[index])
        )
    events.sort(key=lambda item: (item.sentence_index, item.role))
    positions = _first_by_role(events)
    observed_roles = tuple(role for role in _MILESTONES if role in positions)
    observed_positions = [positions[role] for role in observed_roles]
    order_score = _ordered_pair_ratio(observed_positions)
    coverage = float(len(observed_roles) / len(_MILESTONES))

    if len(observed_positions) >= 3:
        rng = np.random.default_rng(seed)
        null = np.asarray(
            [
                _ordered_pair_ratio(rng.permutation(observed_positions).tolist())
                for _ in range(shuffle_count)
            ],
            dtype=float,
        )
        persuasion_z = float(
            (order_score - float(np.mean(null)))
            / (float(np.std(null)) + EPSILON)
        )
    else:
        persuasion_z = 0.0
    completion_factor = 1.0 if "offer" in positions else 0.5
    funnel_score = float(coverage * order_score * completion_factor)

    emotional_roles = [role for role in _EMOTIONS if role in positions]
    emotional_positions = [positions[role] for role in emotional_roles]
    emotional_score = _ordered_pair_ratio(emotional_positions)

    branded_events = [
        item for item in events if item.role == "branded_solution"
    ]
    generic_events = [item for item in events if item.role == "solution"]
    if branded_events:
        branded = branded_events[0]
        generic = next(
            (
                item
                for item in reversed(generic_events)
                if item.sentence_index < branded.sentence_index
            ),
            None,
        )
        branding = BrandingTransition(
            sentence_index=branded.sentence_index,
            generic_concept=generic.sentence if generic else "",
            branded_framework=branded.cue,
            transition_strength=1.0 if generic else 0.6,
        )
    else:
        branding = BrandingTransition(None, "", "", 0.0)

    return FunnelAnalysis(
        events=tuple(events),
        observed_milestones=observed_roles,
        coverage=coverage,
        order_score=order_score,
        persuasion_z=persuasion_z,
        funnel_score=funnel_score,
        emotional_trajectory_score=emotional_score,
        branding_transition=branding,
        offer_start_index=min(offer_indices) if offer_indices else None,
        seed=seed,
        shuffle_count=shuffle_count,
    )
