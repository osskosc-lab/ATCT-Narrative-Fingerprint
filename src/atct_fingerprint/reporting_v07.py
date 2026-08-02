"""Evidence tables for v0.7 deep desire and multi-cause audits."""

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


def write_v07_tables(
    result: FingerprintResult,
    destination: Path,
) -> dict[str, str]:
    analysis = result.semantic_structure_analysis
    paths: dict[str, str] = {}

    desire_path = destination / "desire_frames.csv"
    _write_rows(
        desire_path,
        [
            "subject",
            "action",
            "object",
            "depth",
            "polarity",
            "evidence_span",
            "sentence_index",
            "confidence",
            "rule_or_model",
        ],
        [item.to_dict() for item in analysis.desires.frames],
    )
    paths["desire_frames"] = str(desire_path)

    candidate_path = destination / "cause_candidates_v07.csv"
    _write_rows(
        candidate_path,
        [
            "outcome",
            "cause_type",
            "description",
            "status",
            "evidence_spans",
            "sentence_indices",
            "confidence",
            "rule_or_model",
        ],
        [item.to_dict() for item in analysis.cause_candidates.candidates],
    )
    paths["cause_candidates_v07"] = str(candidate_path)

    competition_path = destination / "cause_competition.csv"
    _write_rows(
        competition_path,
        [
            "primary_cause_type",
            "primary_interpretation",
            "primary_confidence",
            "primary_exclusive",
            "competing_causes",
            "retained_cause_types",
            "cause_monopoly",
            "warning",
            "evidence_spans",
        ],
        [analysis.cause_competition.to_dict()],
    )
    paths["cause_competition"] = str(competition_path)

    metaphor_path = destination / "metaphor_roles_v07.csv"
    _write_rows(
        metaphor_path,
        [
            "label",
            "role",
            "section_index",
            "evidence_span",
            "confidence",
            "rule_or_model",
        ],
        [item.to_dict() for item in analysis.metaphor_roles.roles],
    )
    paths["metaphor_roles_v07"] = str(metaphor_path)

    layer_path = destination / "causal_layers.csv"
    _write_rows(
        layer_path,
        [
            "sentence_index",
            "role",
            "cue",
            "sentence",
            "confidence",
            "rule_or_model",
        ],
        [item.to_dict() for item in analysis.causal_layers.events],
    )
    paths["causal_layers"] = str(layer_path)

    transformation_path = destination / "transformation_v07.csv"
    _write_rows(
        transformation_path,
        [
            "initial_state",
            "changed_state",
            "changed_variable",
            "observable_marker",
            "reversibility",
            "next_shadow_condition",
            "evidence_spans",
            "sentence_indices",
            "operationality",
            "component_scores",
            "rule_or_model",
        ],
        [analysis.transformation.to_dict()],
    )
    paths["transformation_v07"] = str(transformation_path)

    autonomy_path = destination / "autonomy_v07.csv"
    autonomy_rows = [
        {
            **item.to_dict(),
            "autonomy_score": analysis.autonomy.autonomy_score,
            "subjective_ownership_detected": (
                analysis.autonomy.subjective_ownership_detected
            ),
            "warning": analysis.autonomy.warning,
        }
        for item in analysis.autonomy.conditions
    ]
    _write_rows(
        autonomy_path,
        [
            "name",
            "satisfied",
            "sentence_index",
            "evidence_span",
            "confidence",
            "rule_or_model",
            "autonomy_score",
            "subjective_ownership_detected",
            "warning",
        ],
        autonomy_rows,
    )
    paths["autonomy_v07"] = str(autonomy_path)

    scope_path = destination / "title_body_scope.csv"
    _write_rows(
        scope_path,
        [
            "title_claim",
            "body_claim",
            "title_exclusion",
            "body_exclusion",
            "body_alternative_support",
            "mismatch",
            "warning",
            "reason",
            "suggested_revision",
            "evidence_spans",
            "evaluation_status",
        ],
        [analysis.title_body_scope.to_dict()],
    )
    paths["title_body_scope"] = str(scope_path)

    cycle_path = destination / "recursive_cycles.csv"
    cycle_rows = [
        {
            **item.to_dict(),
            "score": analysis.recursive_cycle.score,
            "cycle_detected": analysis.recursive_cycle.detected,
            "structure_type": analysis.recursive_cycle.structure_type,
        }
        for item in analysis.recursive_cycle.criteria
    ]
    _write_rows(
        cycle_path,
        [
            "name",
            "detected",
            "sentence_index",
            "evidence_span",
            "confidence",
            "rule_or_model",
            "score",
            "cycle_detected",
            "structure_type",
        ],
        cycle_rows,
    )
    paths["recursive_cycles"] = str(cycle_path)

    components_path = destination / "v07_components.csv"
    _write_rows(
        components_path,
        [
            "deep_redefinition_score",
            "multi_cause_score",
            "transformation_completeness",
            "metaphor_transition_score",
            "recursive_cycle_score",
            "v07_component_index",
            "interpretation_boundary",
        ],
        [
            {
                "deep_redefinition_score": analysis.deep_redefinition_score,
                "multi_cause_score": analysis.multi_cause_score,
                "transformation_completeness": (
                    analysis.transformation_completeness
                ),
                "metaphor_transition_score": (
                    analysis.metaphor_roles.transition_score
                ),
                "recursive_cycle_score": analysis.recursive_cycle.score,
                "v07_component_index": analysis.v07_component_index,
                "interpretation_boundary": analysis.interpretation_boundary,
            }
        ],
    )
    paths["v07_components"] = str(components_path)
    return paths
