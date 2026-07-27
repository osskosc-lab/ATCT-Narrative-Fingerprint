"""Order-sensitive ATCT narrative features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from .encoders import SentenceEncoder, TfidfSentenceEncoder

DEFAULT_WINDOWS = (3, 5, 8, 13)


def split_sentences(text: str) -> list[str]:
    """Split Japanese or Latin prose while retaining terminal punctuation."""

    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(
        r"(?<=[。！？])[\t ]*|(?<=[.!?])[\t ]+|\n+",
        cleaned,
    )
    return [part.strip() for part in parts if part.strip()]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _cosine_distance_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = np.asarray(left, dtype=float)
    right_matrix = np.asarray(right, dtype=float)
    if left_matrix.shape != right_matrix.shape:
        raise ValueError("left and right trajectories must have the same shape")

    left_norms = np.linalg.norm(left_matrix, axis=1)
    right_norms = np.linalg.norm(right_matrix, axis=1)
    left_n = left_matrix / np.maximum(left_norms[:, None], 1e-12)
    right_n = right_matrix / np.maximum(right_norms[:, None], 1e-12)
    similarity = np.clip(np.sum(left_n * right_n, axis=1), -1.0, 1.0)

    both_zero = (left_norms <= 1e-12) & (right_norms <= 1e-12)
    similarity[both_zero] = 1.0
    return 1.0 - similarity


def _rolling_trajectory(vectors: np.ndarray, window: int) -> np.ndarray:
    """Build a causal, recency-weighted state trajectory."""

    states: list[np.ndarray] = []
    for index in range(len(vectors)):
        start = max(0, index - window + 1)
        chunk = vectors[start : index + 1]
        weights = np.exp(np.linspace(-1.0, 0.0, len(chunk)))
        state = np.average(chunk, axis=0, weights=weights)
        norm = np.linalg.norm(state)
        states.append(state / max(norm, 1e-12))
    return np.vstack(states)


def _align_trajectory_to_sentence_ids(
    trajectory: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Map perturbed-order states back to the original sentence identities.

    ``permutation[position]`` is the original sentence ID occupying that
    perturbed position. Re-alignment ensures perturbation distances compare the
    state attached to the same current sentence, so the metric isolates changed
    history rather than merely comparing different sentence content at a fixed
    position.
    """

    order = np.asarray(permutation, dtype=int)
    if order.ndim != 1 or len(order) != len(trajectory):
        raise ValueError("permutation must contain one index per trajectory row")
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("permutation must contain every sentence index exactly once")

    aligned = np.empty_like(trajectory)
    aligned[order] = trajectory
    return aligned


def _trajectory_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(_cosine_distance_rows(left, right)))


def _nonadjacent_redundancy(vectors: np.ndarray, threshold: float) -> float:
    matches = 0
    pairs = 0
    normalized = _normalize_rows(vectors)
    for left in range(len(normalized)):
        for right in range(left + 2, len(normalized)):
            pairs += 1
            similarity = float(np.dot(normalized[left], normalized[right]))
            matches += int(similarity >= threshold)
    return matches / pairs if pairs else 0.0


def _score(value: float, scale: float = 1.0) -> float:
    """Map a descriptive non-negative value to a bounded display score."""

    return round(100.0 * (1.0 - np.exp(-max(value, 0.0) / scale)), 1)


@dataclass(frozen=True)
class FingerprintResult:
    """Raw diagnostic features and uncalibrated display scores."""

    sentence_count: int
    order_sensitivity: float
    reverse_sensitivity: float
    long_history_dependence: float
    local_global_consistency: float
    local_global_variation: float
    turning_point_magnitude: float
    semantic_redundancy: float
    order_by_window: Mapping[str, float]
    reverse_by_window: Mapping[str, float]
    display_scores: Mapping[str, float]
    seed: int
    shuffle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_fingerprint(
    sentences: Sequence[str],
    vectors: np.ndarray,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    shuffle_count: int = 32,
    seed: int = 42,
    redundancy_threshold: float = 0.86,
) -> FingerprintResult:
    """Compute a descriptive fingerprint from sentences and their vectors.

    The returned values do not estimate authorship probability. At least four
    sentences are required because order perturbations below that length are
    not a useful narrative diagnostic.
    """

    if len(sentences) < 4:
        raise ValueError("at least four sentences are required")
    if shuffle_count < 1:
        raise ValueError("shuffle_count must be positive")
    clean_windows = tuple(sorted({int(window) for window in windows if window >= 2}))
    if not clean_windows:
        raise ValueError("at least one window >= 2 is required")

    matrix = _normalize_rows(vectors)
    if len(matrix) != len(sentences):
        raise ValueError("sentence and vector counts must match")

    trajectories = {
        window: _rolling_trajectory(matrix, min(window, len(matrix)))
        for window in clean_windows
    }
    rng = np.random.default_rng(seed)
    shuffled_distances = {window: [] for window in clean_windows}
    identity = np.arange(len(matrix))
    for _ in range(shuffle_count):
        permutation = rng.permutation(len(matrix))
        if np.array_equal(permutation, identity):
            permutation = np.roll(permutation, 1)
        shuffled = matrix[permutation]
        for window in clean_windows:
            perturbed = _rolling_trajectory(shuffled, min(window, len(matrix)))
            aligned = _align_trajectory_to_sentence_ids(perturbed, permutation)
            shuffled_distances[window].append(
                _trajectory_distance(trajectories[window], aligned)
            )

    reverse_permutation = identity[::-1]
    reversed_matrix = matrix[reverse_permutation]
    order_by_window = {
        str(window): float(np.mean(shuffled_distances[window]))
        for window in clean_windows
    }
    reverse_by_window = {}
    for window in clean_windows:
        reversed_trajectory = _rolling_trajectory(
            reversed_matrix, min(window, len(matrix))
        )
        aligned_reverse = _align_trajectory_to_sentence_ids(
            reversed_trajectory, reverse_permutation
        )
        reverse_by_window[str(window)] = _trajectory_distance(
            trajectories[window], aligned_reverse
        )

    short_window = clean_windows[0]
    long_window = clean_windows[-1]
    long_history = _trajectory_distance(
        trajectories[short_window], trajectories[long_window]
    )

    global_state = np.mean(matrix, axis=0)
    global_state /= max(np.linalg.norm(global_state), 1e-12)
    local_global = np.clip(matrix @ global_state, -1.0, 1.0)

    pivot_window = min(clean_windows, key=lambda value: abs(value - 5))
    pivot = trajectories[pivot_window]
    second_difference = pivot[2:] - (2.0 * pivot[1:-1]) + pivot[:-2]
    turning_point = float(np.mean(np.linalg.norm(second_difference, axis=1)))

    order = float(np.mean(list(order_by_window.values())))
    reverse = float(np.mean(list(reverse_by_window.values())))
    redundancy = _nonadjacent_redundancy(matrix, redundancy_threshold)
    consistency = float(np.mean((local_global + 1.0) / 2.0))
    variation = float(np.std(local_global))

    scores = {
        "文順序依存性": _score(order, 0.45),
        "逆順感度": _score(reverse, 0.45),
        "長距離履歴依存性": _score(long_history, 0.30),
        "局所・大域整合性": round(100.0 * consistency, 1),
        "局所的揺らぎ": _score(variation, 0.25),
        "転換点の実質性": _score(turning_point, 0.75),
        "意味反復度": round(100.0 * redundancy, 1),
    }

    return FingerprintResult(
        sentence_count=len(sentences),
        order_sensitivity=order,
        reverse_sensitivity=reverse,
        long_history_dependence=long_history,
        local_global_consistency=consistency,
        local_global_variation=variation,
        turning_point_magnitude=turning_point,
        semantic_redundancy=redundancy,
        order_by_window=order_by_window,
        reverse_by_window=reverse_by_window,
        display_scores=scores,
        seed=seed,
        shuffle_count=shuffle_count,
    )


def analyze_text(
    text: str,
    *,
    encoder: SentenceEncoder | None = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    shuffle_count: int = 32,
    seed: int = 42,
) -> FingerprintResult:
    """Split, encode, and analyze a document."""

    sentences = split_sentences(text)
    selected_encoder = encoder or TfidfSentenceEncoder()
    vectors = selected_encoder.encode(sentences)
    return compute_fingerprint(
        sentences,
        vectors,
        windows=windows,
        shuffle_count=shuffle_count,
        seed=seed,
    )
