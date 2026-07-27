"""ATCT Narrative Fingerprint v0.2 statistical history evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from .diagnostics import confound_audit, editing_risks, theme_cohesion
from .encoders import SentenceEncoder, TfidfSentenceEncoder
from .history import (
    EPSILON,
    causal_history_states,
    consistency_for_permutation,
    history_change,
    history_consistency,
    long_history_gain,
    mean_consistency,
    normalized_curvature,
    normalize_rows,
)
from .motifs import MotifAnalysis, analyze_motifs
from .null_models import (
    CONTROL_NAMES,
    generate_permutations,
    paragraph_swap,
)
from .segmentation import analysis_segments, paragraph_indices, segment_diversity

DEFAULT_WINDOWS = (3, 5, 8, 13)
_CLOSING_PUNCTUATION = "」』】）》〉〕〗〙〛”’\"'"
_TERMINAL_PATTERN = re.compile(
    rf"[。！？!?](?:[{re.escape(_CLOSING_PUNCTUATION)}]*)"
    rf"|\.(?:[{re.escape(_CLOSING_PUNCTUATION)}]*)(?=\s|$)"
)


def split_sentences(text: str) -> list[str]:
    """Split Japanese or Latin prose while retaining punctuation and quotes."""

    cleaned = text.strip()
    if not cleaned:
        return []
    sentences: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        start = 0
        for match in _TERMINAL_PATTERN.finditer(line):
            end = match.end()
            sentence = line[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        remainder = line[start:].strip()
        if remainder:
            sentences.append(remainder)
    return sentences


@dataclass(frozen=True)
class ControlSummary:
    original: float
    null_mean: float
    null_std: float
    effect: float
    z: float | None
    samples: int


@dataclass(frozen=True)
class FingerprintResult:
    version: str
    sentence_count: int
    primary_metric: str
    primary_window: int
    order_z: float
    gate: str
    original_consistency: float
    controls: Mapping[str, ControlSummary]
    reverse_directionality: float
    history_consistency_by_window: Mapping[str, float]
    long_history_gain: float
    turning_point_z: float
    theme_cohesion: float
    segment_diversity: float
    motif_analysis: MotifAnalysis
    structure_type: str
    sentence_map: tuple[Mapping[str, object], ...]
    confound_audit: Mapping[str, object]
    editing_risks: tuple[Mapping[str, object], ...]
    seed: int
    shuffle_count: int
    disclaimer: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _z_score(original: float, values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    z_value = (original - mean) / (std + EPSILON)
    return mean, std, float(z_value)


def _control_summary(
    original: float,
    values: Sequence[float],
    *,
    deterministic: bool = False,
) -> ControlSummary:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array))
    z_value = None if deterministic else float((original - mean) / (std + EPSILON))
    return ControlSummary(
        original=original,
        null_mean=mean,
        null_std=std,
        effect=original - mean,
        z=z_value,
        samples=len(array),
    )


def _sentence_role(
    index: int,
    history_z: float,
    turn_z: float,
    motif_analysis: MotifAnalysis,
) -> str:
    motif_returns = {pair.second_index for pair in motif_analysis.motif_pairs}
    if index == 0:
        return "opening_anchor"
    if index in motif_returns:
        return "motif_return"
    if turn_z >= 2.0:
        return "structural_turn"
    if history_z >= 2.0:
        return "history_supported"
    if history_z <= -2.0:
        return "unexplained_transition"
    return "continuation"


def _structure_type(
    *,
    order_z: float,
    reverse_directionality: float,
    long_gain: float,
    segment_distance: float,
    motifs: MotifAnalysis,
) -> str:
    if motifs.exact_duplicate_rate >= 0.15:
        return "repetitive"
    if motifs.transformed_closure >= 0.55 and motifs.motif_recurrence > 0:
        return "circular"
    if segment_distance >= 0.65 and order_z < 1.0:
        return "mosaic"
    if segment_distance >= 0.55 and order_z >= 2.0:
        return "branching"
    if long_gain >= 0.04 and order_z >= 1.0:
        return "stepwise"
    if order_z >= 2.0 and reverse_directionality > 0:
        return "linear"
    return "weak_or_mixed"


def compute_fingerprint(
    sentences: Sequence[str],
    vectors: np.ndarray,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    primary_window: int = 5,
    shuffle_count: int = 200,
    controls: Sequence[str] = CONTROL_NAMES,
    seed: int = 42,
    decay: float = 0.35,
    paragraphs: Sequence[Sequence[int]] | None = None,
    segments: Sequence[Sequence[int]] | None = None,
) -> FingerprintResult:
    """Measure history-conditioned coherence against order null models."""

    if len(sentences) < 4:
        raise ValueError("at least four sentences are required")
    if shuffle_count < 2:
        raise ValueError("shuffle_count must be at least two for a Z score")
    selected_controls = tuple(dict.fromkeys(controls))
    unknown = set(selected_controls).difference(CONTROL_NAMES)
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}")
    if "random" not in selected_controls:
        raise ValueError("random control is required for Z_order")

    matrix = normalize_rows(vectors)
    if len(matrix) != len(sentences):
        raise ValueError("sentence and vector counts must match")
    clean_windows = tuple(sorted({int(value) for value in windows if value >= 1}))
    if primary_window not in clean_windows:
        clean_windows = tuple(sorted((*clean_windows, primary_window)))

    states = causal_history_states(matrix, primary_window, decay=decay)
    original_sentence_consistency = history_consistency(matrix, states)
    original_consistency = mean_consistency(matrix, states)
    original_curvature = normalized_curvature(states)
    original_turn_mean = float(np.mean(original_curvature))

    rng = np.random.default_rng(seed)
    control_summaries: dict[str, ControlSummary] = {}
    random_sentence_values: list[np.ndarray] = []
    random_curvature_values: list[np.ndarray] = []
    for control in selected_controls:
        permutations = generate_permutations(
            control, len(matrix), shuffle_count, rng
        )
        aggregate: list[float] = []
        for order in permutations:
            score, aligned_consistency, aligned_states = consistency_for_permutation(
                matrix, order, primary_window, decay=decay
            )
            aggregate.append(score)
            if control == "random":
                random_sentence_values.append(aligned_consistency)
                perturbed_curvature = normalized_curvature(
                    causal_history_states(matrix[order], primary_window, decay=decay)
                )
                aligned_curvature = np.empty_like(perturbed_curvature)
                aligned_curvature[order] = perturbed_curvature
                random_curvature_values.append(aligned_curvature)
        control_summaries[control] = _control_summary(
            original_consistency,
            aggregate,
            deterministic=control == "reverse",
        )

    if paragraphs and len(paragraphs) >= 2:
        paragraph_values = []
        for _ in range(shuffle_count):
            order = paragraph_swap(paragraphs, rng)
            score, _, _ = consistency_for_permutation(
                matrix, order, primary_window, decay=decay
            )
            paragraph_values.append(score)
        control_summaries["paragraph"] = _control_summary(
            original_consistency, paragraph_values
        )

    random_summary = control_summaries["random"]
    order_z = float(random_summary.z or 0.0)
    reverse = control_summaries.get("reverse")
    reverse_directionality = reverse.effect if reverse else 0.0

    consistency_by_window = {}
    for window in clean_windows:
        window_states = causal_history_states(matrix, window, decay=decay)
        consistency_by_window[str(window)] = mean_consistency(matrix, window_states)
    long_gain, sentence_long_gain = long_history_gain(
        matrix,
        short_window=min(clean_windows),
        long_window=max(clean_windows),
        decay=decay,
    )

    random_sentence_array = np.asarray(random_sentence_values)
    sentence_null_mean = np.mean(random_sentence_array, axis=0)
    sentence_null_std = np.std(random_sentence_array, axis=0)
    sentence_z = (
        original_sentence_consistency - sentence_null_mean
    ) / (sentence_null_std + EPSILON)

    random_curvature_array = np.asarray(random_curvature_values)
    curvature_null_mean = np.mean(random_curvature_array, axis=0)
    curvature_null_std = np.std(random_curvature_array, axis=0)
    turn_sentence_z = (
        original_curvature - curvature_null_mean
    ) / (curvature_null_std + EPSILON)
    null_turn_means = np.mean(random_curvature_array, axis=1)
    _, _, turning_point_z = _z_score(original_turn_mean, null_turn_means)

    motifs = analyze_motifs(matrix, sentences)
    changes = history_change(states)
    sentence_rows = tuple(
        {
            "index": index,
            "sentence_number": index + 1,
            "text": sentence,
            "history_consistency": (
                None if index == 0 else float(original_sentence_consistency[index])
            ),
            "history_z": None if index == 0 else float(sentence_z[index]),
            "history_change": float(changes[index]),
            "curvature": None if index < 2 else float(original_curvature[index]),
            "turn_z": None if index < 2 else float(turn_sentence_z[index]),
            "long_history_gain": (
                None if index == 0 else float(sentence_long_gain[index])
            ),
            "role": _sentence_role(
                index, float(sentence_z[index]), float(turn_sentence_z[index]), motifs
            ),
        }
        for index, sentence in enumerate(sentences)
    )

    theme_value, theme_values = theme_cohesion(matrix)
    selected_segments = list(segments or analysis_segments(len(sentences), paragraphs))
    segment_distance = segment_diversity(matrix, selected_segments)
    confounds = confound_audit(sentences)
    risks = editing_risks(
        sentences,
        theme_similarities=theme_values,
        sentence_rows=sentence_rows,
        duplicate_rate=motifs.exact_duplicate_rate,
        confounds=confounds,
    )
    structure = _structure_type(
        order_z=order_z,
        reverse_directionality=reverse_directionality,
        long_gain=long_gain,
        segment_distance=segment_distance,
        motifs=motifs,
    )

    return FingerprintResult(
        version="0.2.0",
        sentence_count=len(sentences),
        primary_metric="Z_order",
        primary_window=primary_window,
        order_z=order_z,
        gate="supported" if order_z >= 2.0 else "unsupported",
        original_consistency=original_consistency,
        controls=control_summaries,
        reverse_directionality=reverse_directionality,
        history_consistency_by_window=consistency_by_window,
        long_history_gain=long_gain,
        turning_point_z=turning_point_z,
        theme_cohesion=theme_value,
        segment_diversity=segment_distance,
        motif_analysis=motifs,
        structure_type=structure,
        sentence_map=sentence_rows,
        confound_audit=confounds,
        editing_risks=tuple(risks),
        seed=seed,
        shuffle_count=shuffle_count,
        disclaimer=(
            "Z_order is statistical evidence for order-conditioned coherence. "
            "It is not an authorship probability or a writing-quality score."
        ),
    )


def analyze_text(
    text: str,
    *,
    encoder: SentenceEncoder | None = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    primary_window: int = 5,
    shuffle_count: int = 200,
    controls: Sequence[str] = CONTROL_NAMES,
    seed: int = 42,
) -> FingerprintResult:
    sentences = split_sentences(text)
    selected_encoder = encoder or TfidfSentenceEncoder()
    vectors = selected_encoder.encode(sentences)
    paragraphs = paragraph_indices(text, split_sentences)
    return compute_fingerprint(
        sentences,
        vectors,
        windows=windows,
        primary_window=primary_window,
        shuffle_count=shuffle_count,
        controls=controls,
        seed=seed,
        paragraphs=paragraphs,
        segments=analysis_segments(len(sentences), paragraphs),
    )
