"""CSV evidence tables added by the v0.6 persuasion audit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .features import FingerprintResult


def _write_rows(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_v06_tables(
    result: FingerprintResult,
    destination: Path,
) -> dict[str, str]:
    analysis = result.persuasion_analysis
    paths: dict[str, str] = {}

    causal_path = destination / "causal_substitutions.csv"
    _write_rows(
        causal_path,
        [
            "outcome",
            "rejected_causes",
            "adopted_cause",
            "evidence_types",
            "source_indices",
            "contrast_score",
        ],
        [item.to_dict() for item in analysis.causal_frames.substitutions],
    )
    paths["causal_substitutions"] = str(causal_path)

    responsibility_path = destination / "responsibility_shifts.csv"
    _write_rows(
        responsibility_path,
        [
            "outcome",
            "rejected_cause",
            "rejected_category",
            "adopted_cause",
            "adopted_category",
            "relief",
            "source_indices",
        ],
        [item.to_dict() for item in analysis.responsibility.shifts],
    )
    paths["responsibility_shifts"] = str(responsibility_path)

    evidence_path = destination / "evidence_types.csv"
    _write_rows(
        evidence_path,
        ["sentence_index", "evidence_type", "cue", "strength", "sentence"],
        [item.to_dict() for item in analysis.evidence.records],
    )
    paths["evidence_types"] = str(evidence_path)

    sequence_path = destination / "sequence_audit.csv"
    sequence_rows = [
        {
            **item.to_dict(),
            "necessity_score": analysis.sequence_audit.necessity_score,
            "insufficient_evidence_warning": (
                analysis.sequence_audit.insufficient_evidence_warning
            ),
        }
        for item in analysis.sequence_audit.claims
    ]
    _write_rows(
        sequence_path,
        [
            "sentence_index",
            "stages",
            "branded_name",
            "alignment_score",
            "first_mention_indices",
            "sentence",
            "necessity_score",
            "insufficient_evidence_warning",
        ],
        sequence_rows,
    )
    paths["sequence_audit"] = str(sequence_path)

    metaphor_path = destination / "metaphor_audit.csv"
    _write_rows(
        metaphor_path,
        ["sentence_index", "metaphor", "stage", "marker", "sentence"],
        [item.to_dict() for item in analysis.metaphor_audit.claims],
    )
    paths["metaphor_audit"] = str(metaphor_path)

    modality_path = destination / "modality_history.csv"
    _write_rows(
        modality_path,
        [
            "sentence_index",
            "label",
            "certainty",
            "cue",
            "evidence_strength",
            "sentence",
        ],
        [item.to_dict() for item in analysis.modality.records],
    )
    paths["modality_history"] = str(modality_path)

    events_path = destination / "persuasion_events.csv"
    _write_rows(
        events_path,
        ["sentence_index", "role", "strength", "cue", "sentence"],
        [item.to_dict() for item in analysis.funnel.events],
    )
    paths["persuasion_events"] = str(events_path)

    layers_path = destination / "document_layers.csv"
    _write_rows(
        layers_path,
        ["sentence_index", "layer", "confidence", "cue", "sentence"],
        [item.to_dict() for item in analysis.document_layers.records],
    )
    paths["document_layers"] = str(layers_path)

    funnel_path = destination / "funnel.csv"
    branding = analysis.funnel.branding_transition
    _write_rows(
        funnel_path,
        [
            "observed_milestones",
            "coverage",
            "order_score",
            "persuasion_z",
            "funnel_score",
            "emotional_trajectory_score",
            "offer_start_index",
            "branding_sentence_index",
            "generic_concept",
            "branded_framework",
            "branding_transition_strength",
        ],
        [
            {
                "observed_milestones": analysis.funnel.observed_milestones,
                "coverage": analysis.funnel.coverage,
                "order_score": analysis.funnel.order_score,
                "persuasion_z": analysis.funnel.persuasion_z,
                "funnel_score": analysis.funnel.funnel_score,
                "emotional_trajectory_score": (
                    analysis.funnel.emotional_trajectory_score
                ),
                "offer_start_index": analysis.funnel.offer_start_index,
                "branding_sentence_index": branding.sentence_index,
                "generic_concept": branding.generic_concept,
                "branded_framework": branding.branded_framework,
                "branding_transition_strength": branding.transition_strength,
            }
        ],
    )
    paths["funnel"] = str(funnel_path)
    return paths
