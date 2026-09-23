"""Long-range motif return separated from exact semantic duplication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .history import EPSILON, normalize_rows


@dataclass(frozen=True)
class MotifPair:
    first_index: int
    second_index: int
    first_sentence: str
    second_sentence: str
    similarity: float
    relative_distance: float
    kind: str
    contribution: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotifAnalysis:
    raw_opening_closure: float
    transformed_closure: float
    motif_recurrence: float
    exact_duplicate_rate: float
    motif_pairs: tuple[MotifPair, ...]
    duplicate_pairs: tuple[MotifPair, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["motif_pairs"] = [pair.to_dict() for pair in self.motif_pairs]
        payload["duplicate_pairs"] = [pair.to_dict() for pair in self.duplicate_pairs]
        return payload


def _centroid(matrix: np.ndarray) -> np.ndarray:
    center = np.mean(matrix, axis=0)
    return center / max(np.linalg.norm(center), EPSILON)


def analyze_motifs(
    vectors: np.ndarray,
    sentences: Sequence[str],
    *,
    edge_width: int = 2,
    motif_threshold: float = 0.70,
    duplicate_threshold: float = 0.985,
    min_relative_distance: float = 0.35,
) -> MotifAnalysis:
    """Classify transformed long-range returns and exact copied meaning."""

    matrix = normalize_rows(vectors)
    if len(matrix) != len(sentences):
        raise ValueError("sentence and vector counts must match")
    width = max(1, min(edge_width, len(matrix) // 2))
    opening = _centroid(matrix[:width])
    ending = _centroid(matrix[-width:])
    raw_closure = float(np.clip(np.dot(opening, ending), -1.0, 1.0))
    transformed_closure = (
        max(raw_closure, 0.0) if raw_closure < duplicate_threshold else 0.0
    )

    motif_pairs: list[MotifPair] = []
    duplicate_pairs: list[MotifPair] = []
    possible_pairs = 0
    recurrence_sum = 0.0
    denominator = max(len(matrix) - 1, 1)
    for left in range(len(matrix)):
        for right in range(left + 2, len(matrix)):
            possible_pairs += 1
            similarity = float(np.clip(np.dot(matrix[left], matrix[right]), -1, 1))
            relative_distance = (right - left) / denominator
            if similarity >= duplicate_threshold:
                duplicate_pairs.append(
                    MotifPair(
                        left,
                        right,
                        sentences[left],
                        sentences[right],
                        similarity,
                        relative_distance,
                        "exact_duplicate",
                        0.0,
                    )
                )
            elif (
                similarity >= motif_threshold
                and relative_distance >= min_relative_distance
            ):
                contribution = relative_distance * (
                    (similarity - motif_threshold) / (1.0 - motif_threshold)
                )
                recurrence_sum += contribution
                motif_pairs.append(
                    MotifPair(
                        left,
                        right,
                        sentences[left],
                        sentences[right],
                        similarity,
                        relative_distance,
                        "transformed_return",
                        contribution,
                    )
                )
    return MotifAnalysis(
        raw_opening_closure=raw_closure,
        transformed_closure=transformed_closure,
        motif_recurrence=recurrence_sum / max(len(matrix), 1),
        exact_duplicate_rate=len(duplicate_pairs) / max(possible_pairs, 1),
        motif_pairs=tuple(sorted(motif_pairs, key=lambda pair: -pair.contribution)),
        duplicate_pairs=tuple(duplicate_pairs),
    )
