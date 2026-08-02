"""v0.6 reader-recognition transformation and persuasion analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .causal_frames import CausalFrameAnalysis, analyze_causal_frames
from .document_layers import DocumentLayerAnalysis, analyze_document_layers
from .evidence_types import EvidenceAnalysis, analyze_evidence_types
from .funnel import FunnelAnalysis, analyze_funnel
from .metaphor_audit import MetaphorAudit, analyze_metaphors
from .modality import ModalityAnalysis, analyze_modality
from .responsibility import ResponsibilityAnalysis, analyze_responsibility
from .sequence_audit import SequenceAudit, analyze_sequence_claims


@dataclass(frozen=True)
class ReaderTransformation:
    sentence_index: int
    reader_state: str
    cause_role: str
    emotion_role: str
    solution_role: str
    offer_role: str
    modality: str
    evidence_type: str
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PersuasionAnalysis:
    content_z: float
    persuasion_z: float
    quadrant: str
    content_method: str
    content_components: Mapping[str, float]
    reader_transformations: tuple[ReaderTransformation, ...]
    causal_frames: CausalFrameAnalysis
    responsibility: ResponsibilityAnalysis
    sequence_audit: SequenceAudit
    evidence: EvidenceAnalysis
    metaphor_audit: MetaphorAudit
    modality: ModalityAnalysis
    document_layers: DocumentLayerAnalysis
    funnel: FunnelAnalysis

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _quadrant(content_z: float, persuasion_z: float) -> str:
    content_high = content_z >= 2.0
    persuasion_high = persuasion_z >= 2.0
    if content_high and persuasion_high:
        return "argument_and_sales_design"
    if content_high:
        return "essay_or_education"
    if persuasion_high:
        return "copywriting_led"
    return "weak_or_fragmented"


def analyze_persuasion(
    sentences: Sequence[str],
    *,
    content_z: float,
    content_components: Mapping[str, float],
    shuffle_count: int = 200,
    seed: int = 42,
) -> PersuasionAnalysis:
    evidence = analyze_evidence_types(sentences)
    layers = analyze_document_layers(sentences)
    causal = analyze_causal_frames(sentences, evidence)
    responsibility = analyze_responsibility(causal)
    sequence = analyze_sequence_claims(sentences)
    metaphor = analyze_metaphors(sentences)
    modality = analyze_modality(
        sentences,
        evidence,
        body_indices=layers.content_indices,
    )
    funnel = analyze_funnel(
        sentences,
        causal,
        layers,
        shuffle_count=shuffle_count,
        seed=seed,
    )

    event_roles: dict[int, list[str]] = {}
    for event in funnel.events:
        event_roles.setdefault(event.sentence_index, []).append(event.role)
    rejected_indices: set[int] = set()
    adopted_indices: set[int] = set()
    for substitution in causal.substitutions:
        rejected_indices.update(substitution.source_indices[:-1])
        adopted_indices.add(substitution.source_indices[-1])

    transformations: list[ReaderTransformation] = []
    for index, sentence in enumerate(sentences):
        roles = event_roles.get(index, [])
        reader_state = next(
            (role for role in ("pain", "relief", "hope") if role in roles),
            "",
        )
        emotion_role = reader_state
        if index in rejected_indices:
            cause_role = "rejected_cause"
        elif index in adopted_indices:
            cause_role = "adopted_cause"
        else:
            cause_role = ""
        solution_role = next(
            (
                role
                for role in ("solution", "branded_solution")
                if role in roles
            ),
            "",
        )
        offer_role = "offer" if "offer" in roles else ""
        transformations.append(
            ReaderTransformation(
                sentence_index=index,
                reader_state=reader_state,
                cause_role=cause_role,
                emotion_role=emotion_role,
                solution_role=solution_role,
                offer_role=offer_role,
                modality=modality.records[index].label,
                evidence_type=evidence.records[index].evidence_type,
                sentence=sentence,
            )
        )

    return PersuasionAnalysis(
        content_z=content_z,
        persuasion_z=funnel.persuasion_z,
        quadrant=_quadrant(content_z, funnel.persuasion_z),
        content_method="max_preregistered_content_channel",
        content_components=dict(content_components),
        reader_transformations=tuple(transformations),
        causal_frames=causal,
        responsibility=responsibility,
        sequence_audit=sequence,
        evidence=evidence,
        metaphor_audit=metaphor,
        modality=modality,
        document_layers=layers,
        funnel=funnel,
    )
