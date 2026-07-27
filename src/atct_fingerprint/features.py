"""ATCT Narrative Fingerprint v0.3 hierarchical intervention evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from .diagnostics import (
    confound_audit,
    editing_risks,
    licensed_jumps,
    theme_cohesion,
)
from .directionality import DirectionalityResult, conditional_directionality
from .document import ParsedDocument, SectionSpan, parse_markdown_document
from .encoders import SentenceEncoder, TfidfSentenceEncoder
from .history import (
    EPSILON,
    PredictiveGainResult,
    causal_history_states,
    consistency_for_permutation,
    history_change,
    history_consistency,
    mean_consistency,
    normalized_curvature,
    normalize_rows,
    rolling_predictive_gain,
)
from .motifs import MotifAnalysis, analyze_motifs
from .null_models import (
    CONTROL_NAMES,
    generate_permutations,
)
from .segmentation import (
    SectionGraph,
    analysis_segments,
    paragraph_indices,
    section_graph as build_section_graph,
    segment_diversity,
)

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
    directionality: DirectionalityResult
    history_consistency_by_window: Mapping[str, float]
    long_history_gain: float
    long_history_prediction: PredictiveGainResult
    turning_point_z: float
    theme_cohesion: float
    segment_diversity: float
    motif_analysis: MotifAnalysis
    structure_type: str
    document_structure: Mapping[str, object]
    section_graph: SectionGraph
    sentence_map: tuple[Mapping[str, object], ...]
    licensed_jumps: tuple[Mapping[str, object], ...]
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
    graph: SectionGraph,
    last_sentence: str,
) -> str:
    if motifs.exact_duplicate_rate >= 0.15:
        return "repetitive"
    if (
        motifs.transformed_closure >= 0.35
        and motifs.motif_recurrence > 0
        and (
            last_sentence.rstrip().endswith(("?", "？"))
            or (
                motifs.transformed_closure < 0.55
                and graph.conclusion_convergence >= 0.55
            )
        )
    ):
        return "open_spiral"
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
    sections: Sequence[SectionSpan] | None = None,
    document_structure: Mapping[str, object] | None = None,
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
        structured_blocks: Sequence[Sequence[int]] | None = None
        if control in {"paragraph_inner", "paragraph_order"}:
            structured_blocks = paragraphs
            if not structured_blocks:
                continue
            if control == "paragraph_order" and len(structured_blocks) < 2:
                continue
            if control == "paragraph_inner" and not any(
                len(block) >= 2 for block in structured_blocks
            ):
                continue
        elif control == "section_order":
            structured_blocks = [
                section.sentence_indices
                for section in (sections or ())
                if section.sentence_indices
            ]
            if len(structured_blocks) < 2:
                continue
        permutations = generate_permutations(
            control,
            len(matrix),
            shuffle_count,
            rng,
            blocks=structured_blocks,
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

    random_summary = control_summaries["random"]
    order_z = float(random_summary.z or 0.0)
    directional = conditional_directionality(
        sentences,
        context_window=primary_window,
    )
    reverse_directionality = directional.asymmetry

    consistency_by_window = {}
    for window in clean_windows:
        window_states = causal_history_states(matrix, window, decay=decay)
        consistency_by_window[str(window)] = mean_consistency(matrix, window_states)
    predictive_gain = rolling_predictive_gain(
        matrix,
        short_window=min(clean_windows),
        long_window=max(clean_windows),
        decay=decay,
    )
    long_gain = predictive_gain.gain
    sentence_long_gain = predictive_gain.sentence_gains

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
    effective_sections = tuple(
        sections
        or (
            SectionSpan(
                section_id=0,
                heading="Document",
                level=0,
                parent=None,
                sentence_indices=tuple(range(len(sentences))),
                start_sentence=0,
                end_sentence=len(sentences) - 1,
            ),
        )
    )
    sentence_sections = ["Document"] * len(sentences)
    for section in effective_sections:
        for sentence_index in section.sentence_indices:
            sentence_sections[sentence_index] = section.heading
    raw_sentence_rows = tuple(
        {
            "index": index,
            "sentence_number": index + 1,
            "text": sentence,
            "section": sentence_sections[index],
            "history_consistency": (
                None if index == 0 else float(original_sentence_consistency[index])
            ),
            "history_z": None if index == 0 else float(sentence_z[index]),
            "history_change": float(changes[index]),
            "curvature": None if index < 2 else float(original_curvature[index]),
            "turn_z": None if index < 2 else float(turn_sentence_z[index]),
            "long_history_gain": (
                None
                if sentence_long_gain[index] is None
                else float(sentence_long_gain[index])
            ),
            "direction_delta": directional.sentence_deltas[index],
            "role": _sentence_role(
                index, float(sentence_z[index]), float(turn_sentence_z[index]), motifs
            ),
        }
        for index, sentence in enumerate(sentences)
    )
    jump_records = licensed_jumps(sentences, raw_sentence_rows)
    licensed_indices = {
        int(record["transition_sentence"]) for record in jump_records
    }
    sentence_rows = tuple(
        {
            **row,
            "licensed_jump": int(row["index"]) in licensed_indices,
            "role": (
                "licensed_jump"
                if int(row["index"]) in licensed_indices
                else row["role"]
            ),
        }
        for row in raw_sentence_rows
    )

    theme_value, theme_values = theme_cohesion(matrix)
    selected_segments = list(segments or analysis_segments(len(sentences), paragraphs))
    segment_distance = segment_diversity(matrix, selected_segments)
    graph = build_section_graph(
        matrix,
        effective_sections,
        licensed_sentence_indices=licensed_indices,
    )
    confounds = confound_audit(sentences)
    risks = editing_risks(
        sentences,
        theme_similarities=theme_values,
        sentence_rows=sentence_rows,
        duplicate_rate=motifs.exact_duplicate_rate,
        confounds=confounds,
        licensed_indices=licensed_indices,
    )
    structure = _structure_type(
        order_z=order_z,
        reverse_directionality=reverse_directionality,
        long_gain=long_gain,
        segment_distance=segment_distance,
        motifs=motifs,
        graph=graph,
        last_sentence=sentences[-1],
    )

    return FingerprintResult(
        version="0.3.0",
        sentence_count=len(sentences),
        primary_metric="Z_order",
        primary_window=primary_window,
        order_z=order_z,
        gate="supported" if order_z >= 2.0 else "unsupported",
        original_consistency=original_consistency,
        controls=control_summaries,
        reverse_directionality=reverse_directionality,
        directionality=directional,
        history_consistency_by_window=consistency_by_window,
        long_history_gain=long_gain,
        long_history_prediction=predictive_gain,
        turning_point_z=turning_point_z,
        theme_cohesion=theme_value,
        segment_diversity=segment_distance,
        motif_analysis=motifs,
        structure_type=structure,
        document_structure=dict(document_structure or {}),
        section_graph=graph,
        sentence_map=sentence_rows,
        licensed_jumps=jump_records,
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
    parse_markdown: bool = True,
) -> FingerprintResult:
    parsed: ParsedDocument | None = None
    if parse_markdown:
        parsed = parse_markdown_document(text, split_sentences)
        sentences = list(parsed.prose_sentences)
        paragraphs = [list(block) for block in parsed.paragraph_indices]
        sections = parsed.sections
        structure_payload = parsed.to_dict()
    else:
        sentences = split_sentences(text)
        paragraphs = paragraph_indices(text, split_sentences)
        sections = (
            SectionSpan(
                section_id=0,
                heading="Document",
                level=0,
                parent=None,
                sentence_indices=tuple(range(len(sentences))),
                start_sentence=0 if sentences else None,
                end_sentence=len(sentences) - 1 if sentences else None,
            ),
        )
        structure_payload = {
            "mode": "plain_text",
            "layer_counts": {
                "prose": len(paragraphs),
                "equation": 0,
                "structure": 0,
            },
        }
    selected_encoder = encoder or TfidfSentenceEncoder()
    vectors = selected_encoder.encode(sentences)
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
        sections=sections,
        document_structure=structure_payload,
    )
