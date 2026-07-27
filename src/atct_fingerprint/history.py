"""Causal history states that never include the sentence being evaluated."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class PredictiveGainResult:
    method: str
    short_window: int
    long_window: int
    short_test_error: float
    long_test_error: float
    gain: float
    test_samples: int
    sentence_gains: tuple[float | None, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPSILON)


def causal_history_states(
    vectors: np.ndarray,
    window: int,
    *,
    decay: float = 0.35,
) -> np.ndarray:
    """Return recency-weighted states made only from x[t-window:t]."""

    if window < 1:
        raise ValueError("window must be positive")
    if decay < 0:
        raise ValueError("decay must be non-negative")
    matrix = normalize_rows(vectors)
    states = np.zeros_like(matrix)
    for index in range(1, len(matrix)):
        start = max(0, index - window)
        chunk = matrix[start:index]
        ages = np.arange(len(chunk) - 1, -1, -1, dtype=float)
        weights = np.exp(-decay * ages)
        state = np.average(chunk, axis=0, weights=weights)
        states[index] = state / max(np.linalg.norm(state), EPSILON)
    return states


def legacy_inclusive_states(
    vectors: np.ndarray,
    window: int,
    *,
    decay: float = 0.35,
) -> np.ndarray:
    """v0.1 baseline: history state that incorrectly includes x_t."""

    if window < 1:
        raise ValueError("window must be positive")
    matrix = normalize_rows(vectors)
    states = np.zeros_like(matrix)
    for index in range(len(matrix)):
        start = max(0, index - window + 1)
        chunk = matrix[start : index + 1]
        ages = np.arange(len(chunk) - 1, -1, -1, dtype=float)
        weights = np.exp(-decay * ages)
        state = np.average(chunk, axis=0, weights=weights)
        states[index] = state / max(np.linalg.norm(state), EPSILON)
    return states


def history_consistency(vectors: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Compute C_t = cos(x_t, h_t), using 0 when no prior state exists."""

    matrix = normalize_rows(vectors)
    history = normalize_rows(states)
    if matrix.shape != history.shape:
        raise ValueError("vectors and states must have the same shape")
    scores = np.clip(np.sum(matrix * history, axis=1), -1.0, 1.0)
    no_history = np.linalg.norm(states, axis=1) <= EPSILON
    scores[no_history] = 0.0
    return scores


def mean_consistency(vectors: np.ndarray, states: np.ndarray) -> float:
    """Aggregate consistency with the first, history-free sentence fixed at 0."""

    return float(np.mean(history_consistency(vectors, states)))


def consistency_for_permutation(
    vectors: np.ndarray,
    permutation: Sequence[int],
    window: int,
    *,
    decay: float = 0.35,
    inclusive: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate a perturbed order and align values back to sentence identity."""

    matrix = normalize_rows(vectors)
    order = np.asarray(permutation, dtype=int)
    if not np.array_equal(np.sort(order), np.arange(len(matrix))):
        raise ValueError("permutation must contain every sentence index once")
    perturbed = matrix[order]
    state_builder = legacy_inclusive_states if inclusive else causal_history_states
    states = state_builder(perturbed, window, decay=decay)
    consistency = history_consistency(perturbed, states)
    aligned_consistency = np.empty_like(consistency)
    aligned_states = np.empty_like(states)
    aligned_consistency[order] = consistency
    aligned_states[order] = states
    return float(np.mean(consistency)), aligned_consistency, aligned_states


def history_change(states: np.ndarray) -> np.ndarray:
    """First difference magnitude of the history state."""

    matrix = np.asarray(states, dtype=float)
    changes = np.zeros(len(matrix), dtype=float)
    if len(matrix) > 1:
        changes[1:] = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    return changes


def normalized_curvature(states: np.ndarray) -> np.ndarray:
    """Scale-free curvature of consecutive history-state changes."""

    matrix = np.asarray(states, dtype=float)
    curvature = np.zeros(len(matrix), dtype=float)
    if len(matrix) < 3:
        return curvature
    deltas = np.diff(matrix, axis=0)
    numerator = np.linalg.norm(deltas[1:] - deltas[:-1], axis=1)
    denominator = (
        np.linalg.norm(deltas[1:], axis=1)
        + np.linalg.norm(deltas[:-1], axis=1)
        + EPSILON
    )
    curvature[2:] = numerator / denominator
    return curvature


def long_history_gain(
    vectors: np.ndarray,
    *,
    short_window: int = 3,
    long_window: int = 13,
    decay: float = 0.35,
) -> tuple[float, np.ndarray]:
    """Measure predictive consistency gained by using a longer past."""

    short = history_consistency(
        vectors, causal_history_states(vectors, short_window, decay=decay)
    )
    long = history_consistency(
        vectors, causal_history_states(vectors, long_window, decay=decay)
    )
    gain = long - short
    return float(np.mean(gain)), gain


def _ridge_predict(
    train_states: np.ndarray,
    train_targets: np.ndarray,
    test_state: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    design = np.column_stack([train_states, np.ones(len(train_states))])
    test = np.append(test_state, 1.0)
    gram = design @ design.T
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        train_targets,
    )
    prediction = test @ design.T @ coefficients
    return prediction / max(np.linalg.norm(prediction), EPSILON)


def rolling_predictive_gain(
    vectors: np.ndarray,
    *,
    short_window: int = 3,
    long_window: int = 13,
    decay: float = 0.35,
    min_train: int = 4,
    ridge: float = 1e-3,
) -> PredictiveGainResult:
    """Compare short/long histories on held-out future sentences.

    At each time t, two ridge maps are fitted only on earlier state-target
    pairs and evaluated on x_t. Positive gain means the longer history lowered
    rolling-origin cosine prediction error.
    """

    if min_train < 2:
        raise ValueError("min_train must be at least two")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    matrix = normalize_rows(vectors)
    short_states = causal_history_states(matrix, short_window, decay=decay)
    long_states = causal_history_states(matrix, long_window, decay=decay)
    short_errors: list[float] = []
    long_errors: list[float] = []
    sentence_gains: list[float | None] = [None] * len(matrix)
    for index in range(min_train + 1, len(matrix)):
        train_indices = np.arange(1, index)
        if len(train_indices) < min_train:
            continue
        short_prediction = _ridge_predict(
            short_states[train_indices],
            matrix[train_indices],
            short_states[index],
            ridge=ridge,
        )
        long_prediction = _ridge_predict(
            long_states[train_indices],
            matrix[train_indices],
            long_states[index],
            ridge=ridge,
        )
        short_error = 1.0 - float(
            np.clip(np.dot(short_prediction, matrix[index]), -1.0, 1.0)
        )
        long_error = 1.0 - float(
            np.clip(np.dot(long_prediction, matrix[index]), -1.0, 1.0)
        )
        short_errors.append(short_error)
        long_errors.append(long_error)
        sentence_gains[index] = short_error - long_error
    short_mean = float(np.mean(short_errors)) if short_errors else 0.0
    long_mean = float(np.mean(long_errors)) if long_errors else 0.0
    return PredictiveGainResult(
        method="rolling_origin_ridge",
        short_window=short_window,
        long_window=long_window,
        short_test_error=short_mean,
        long_test_error=long_mean,
        gain=short_mean - long_mean,
        test_samples=len(short_errors),
        sentence_gains=tuple(sentence_gains),
    )
