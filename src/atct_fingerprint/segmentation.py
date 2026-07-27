"""Paragraph and fallback segment construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .history import EPSILON, normalize_rows


def paragraph_indices(
    text: str,
    splitter: Callable[[str], list[str]],
) -> list[list[int]]:
    """Map blank-line separated paragraphs to global sentence indices."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    blocks: list[list[int]] = []
    cursor = 0
    for paragraph in paragraphs:
        count = len(splitter(paragraph))
        if count:
            blocks.append(list(range(cursor, cursor + count)))
            cursor += count
    return blocks


def analysis_segments(
    sentence_count: int,
    paragraphs: Sequence[Sequence[int]] | None = None,
    *,
    fallback_size: int = 5,
) -> list[list[int]]:
    """Use real paragraphs when possible, otherwise fixed contiguous scenes."""

    clean = [list(block) for block in (paragraphs or []) if block]
    if len(clean) >= 2 and sum(map(len, clean)) == sentence_count:
        return clean
    return [
        list(range(start, min(start + fallback_size, sentence_count)))
        for start in range(0, sentence_count, fallback_size)
    ]


def segment_diversity(
    vectors: np.ndarray,
    segments: Sequence[Sequence[int]],
) -> float:
    """Mean cosine distance between segment centroids."""

    matrix = normalize_rows(vectors)
    centroids: list[np.ndarray] = []
    for segment in segments:
        indices = np.asarray(segment, dtype=int)
        if len(indices) == 0:
            continue
        centroid = np.mean(matrix[indices], axis=0)
        centroid /= max(np.linalg.norm(centroid), EPSILON)
        centroids.append(centroid)
    if len(centroids) < 2:
        return 0.0
    distances = [
        1.0 - float(np.clip(np.dot(centroids[left], centroids[right]), -1, 1))
        for left in range(len(centroids))
        for right in range(left + 1, len(centroids))
    ]
    return float(np.mean(distances))
