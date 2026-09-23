"""Paragraph and fallback segment construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .document import SectionSpan
from .history import EPSILON, normalize_rows


@dataclass(frozen=True)
class SectionNode:
    section_id: int
    heading: str
    level: int
    parent: str | None
    sentence_indices: tuple[int, ...]
    theme_similarity: float


@dataclass(frozen=True)
class SectionEdge:
    source_id: int
    target_id: int
    source_heading: str
    target_heading: str
    transition_distance: float
    target_history_support: float
    relation: str
    licensed_jump: bool


@dataclass(frozen=True)
class SectionGraph:
    nodes: tuple[SectionNode, ...]
    edges: tuple[SectionEdge, ...]
    opening_ending_similarity: float
    conclusion_convergence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def section_graph(
    vectors: np.ndarray,
    sections: Sequence[SectionSpan],
    *,
    licensed_sentence_indices: Sequence[int] = (),
) -> SectionGraph:
    """Build a sequential graph of section centroids and transitions."""

    matrix = normalize_rows(vectors)
    global_center = np.mean(matrix, axis=0)
    global_center /= max(np.linalg.norm(global_center), EPSILON)
    licensed = set(int(value) for value in licensed_sentence_indices)
    nodes: list[SectionNode] = []
    centroids: list[np.ndarray] = []
    usable_sections: list[SectionSpan] = []
    for section in sections:
        if not section.sentence_indices:
            continue
        indices = np.asarray(section.sentence_indices, dtype=int)
        centroid = np.mean(matrix[indices], axis=0)
        centroid /= max(np.linalg.norm(centroid), EPSILON)
        centroids.append(centroid)
        usable_sections.append(section)
        nodes.append(
            SectionNode(
                section_id=section.section_id,
                heading=section.heading,
                level=section.level,
                parent=section.parent,
                sentence_indices=section.sentence_indices,
                theme_similarity=float(
                    np.clip(np.dot(centroid, global_center), -1.0, 1.0)
                ),
            )
        )
    edges: list[SectionEdge] = []
    for index in range(1, len(usable_sections)):
        source = usable_sections[index - 1]
        target = usable_sections[index]
        source_center = centroids[index - 1]
        target_center = centroids[index]
        distance = 1.0 - float(
            np.clip(np.dot(source_center, target_center), -1.0, 1.0)
        )
        target_first = matrix[target.sentence_indices[0]]
        support = float(
            np.clip(np.dot(target_first, source_center), -1.0, 1.0)
        )
        target_theme = nodes[index].theme_similarity
        source_theme = nodes[index - 1].theme_similarity
        if distance >= 0.65:
            relation = "domain_shift"
        elif target_theme > source_theme + 0.05:
            relation = "convergence"
        elif target_theme < source_theme - 0.05:
            relation = "divergence"
        else:
            relation = "continuation"
        boundary_indices = {
            target.sentence_indices[0],
            max(target.sentence_indices[0] - 1, 0),
        }
        edges.append(
            SectionEdge(
                source_id=source.section_id,
                target_id=target.section_id,
                source_heading=source.heading,
                target_heading=target.heading,
                transition_distance=distance,
                target_history_support=support,
                relation=relation,
                licensed_jump=bool(boundary_indices.intersection(licensed)),
            )
        )
    opening_ending = (
        float(np.clip(np.dot(centroids[0], centroids[-1]), -1.0, 1.0))
        if len(centroids) >= 2
        else 0.0
    )
    conclusion_convergence = nodes[-1].theme_similarity if nodes else 0.0
    return SectionGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        opening_ending_similarity=opening_ending,
        conclusion_convergence=conclusion_convergence,
    )
