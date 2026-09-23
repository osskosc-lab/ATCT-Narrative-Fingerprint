"""Orchestrate transparent v0.7 deep-reframing audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

from .autonomy import AutonomyAnalysis, analyze_autonomy
from .causal_layers import CausalLayerAnalysis, analyze_causal_layers
from .cause_candidates import CauseCandidateAnalysis, analyze_cause_candidates
from .cause_competition import CauseCompetition, analyze_cause_competition
from .desire_frames import DesireAnalysis, analyze_desires
from .metaphor_roles import MetaphorRoleAnalysisV07, analyze_metaphor_roles_v07
from .recursive_cycles import RecursiveCycleAnalysis, analyze_recursive_cycle
from .scope_audit import TitleBodyScopeAudit, audit_title_body_scope
from .transformation import TransformationFrame, analyze_transformation


_DISCOURSE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("call_to_action", re.compile(r"登録|申し込|読んで|受け取|フォロー|試してください")),
    ("recursive_closure", re.compile(r"新たなShadow|次のShadow|また新たな|循環|再び")),
    ("individual_agency", re.compile(r"自分で選|自分で定義|本人が選|判断する")),
    ("structural_constraint", re.compile(r"制度|労働条件|雇用|育休|時短|リモート")),
    ("transformation_definition", re.compile(r"Transformation|変容とは|変わるとは|書き換")),
    ("case_example", re.compile(r"たとえば|例えば|事例|私の場合|ある人")),
    ("exploration", re.compile(r"Seeking|探索|試行|問い直|探し")),
    ("model_failure", re.compile(r"機能しない|不整合|合わなく|違和感|古いOS")),
    ("historical_model", re.compile(r"以前|かつて|これまで|会社から与え|社会から与え")),
    ("deep_reframing", re.compile(r"本当に必要なのは|本当は|重要なのは|判断基準|成功OS")),
    ("alternative_causes", re.compile(r"労働条件|賃金|健康|人間関係|職務適性|家庭")),
    ("surface_problem", re.compile(r"転職したい|辞めたい|苦しい|違和感")),
)


@dataclass(frozen=True)
class SemanticDiscourseRole:
    sentence_index: int
    role: str
    cue: str
    sentence: str
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticStructureAnalysis:
    document_types: tuple[str, ...]
    desires: DesireAnalysis
    title_body_scope: TitleBodyScopeAudit
    cause_candidates: CauseCandidateAnalysis
    cause_competition: CauseCompetition
    causal_layers: CausalLayerAnalysis
    metaphor_roles: MetaphorRoleAnalysisV07
    transformation: TransformationFrame
    autonomy: AutonomyAnalysis
    recursive_cycle: RecursiveCycleAnalysis
    discourse_roles: tuple[SemanticDiscourseRole, ...]
    deep_redefinition_score: float
    multi_cause_score: float
    transformation_completeness: float
    v07_component_index: float
    interpretation_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _discourse_roles(sentences: Sequence[str]) -> tuple[SemanticDiscourseRole, ...]:
    records: list[SemanticDiscourseRole] = []
    for index, sentence in enumerate(sentences):
        for role, pattern in _DISCOURSE_RULES:
            match = pattern.search(sentence)
            if not match:
                continue
            records.append(
                SemanticDiscourseRole(
                    sentence_index=index,
                    role=role,
                    cue=match.group(0),
                    sentence=sentence,
                    confidence=0.85,
                )
            )
            break
    return tuple(records)


def analyze_semantic_structure(
    sentences: Sequence[str],
    *,
    title: str = "",
) -> SemanticStructureAnalysis:
    """Run v0.7 rules without collapsing candidates into one true cause."""

    desires = analyze_desires(sentences)
    scope = audit_title_body_scope(title, sentences)
    candidates = analyze_cause_candidates(sentences)
    competition = analyze_cause_competition(candidates)
    layers = analyze_causal_layers(sentences)
    metaphor = analyze_metaphor_roles_v07(sentences)
    transformation = analyze_transformation(sentences)
    autonomy = analyze_autonomy(sentences)
    cycle = analyze_recursive_cycle(sentences)
    discourse = _discourse_roles(sentences)

    deep_score = (
        desires.depth_score
        * desires.evidence_support
        * (1.0 - scope.mismatch)
    )
    multi_score = layers.bridge_score * (1.0 - competition.cause_monopoly)
    transformation_complete = (
        transformation.operationality * autonomy.autonomy_score
    )
    component_index = (
        0.25 * deep_score
        + 0.25 * multi_score
        + 0.25 * transformation_complete
        + 0.15 * metaphor.transition_score
        + 0.10 * cycle.score
    )
    document_types: list[str] = []
    if desires.deep_support >= 0.50:
        document_types.append("deep_reframing_essay")
    if cycle.detected:
        document_types.append("recursive_transformation_cycle")
    if layers.bridge_score >= 0.50:
        document_types.append("institution_individual_bridge")
    if not document_types:
        document_types.append("semantic_structure_undetermined")

    return SemanticStructureAnalysis(
        document_types=tuple(document_types),
        desires=desires,
        title_body_scope=scope,
        cause_candidates=candidates,
        cause_competition=competition,
        causal_layers=layers,
        metaphor_roles=metaphor,
        transformation=transformation,
        autonomy=autonomy,
        recursive_cycle=cycle,
        discourse_roles=discourse,
        deep_redefinition_score=float(deep_score),
        multi_cause_score=float(multi_score),
        transformation_completeness=float(transformation_complete),
        v07_component_index=float(component_index),
        interpretation_boundary=(
            "The v0.7 component index is a rule-coverage summary, not a "
            "quality score or proof of a true psychological cause. Deep "
            "interpretations remain non-exclusive and every finding retains "
            "its source span. Macro-F1 is pending a frozen annotated corpus."
        ),
    )
