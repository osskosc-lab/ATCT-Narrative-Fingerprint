"""Functional motif roles, including different words with the same role."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


_FUNCTIONS = {
    "self_confirmation_device": (
        "ノート",
        "鏡",
        "記録",
        "ページ",
        "録音データ",
        "過去の動画",
        "動画",
    ),
    "agency_instrument": ("手", "指先", "ペン"),
    "ethical_judgment": ("正義", "悪意", "責任"),
}


@dataclass(frozen=True)
class MotifFunctionOccurrence:
    sentence_index: int
    motif: str
    function: str
    role: str
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotifFunctionTransition:
    motif: str
    function: str
    first_index: int
    last_index: int
    initial_role: str
    final_role: str
    stage_count: int
    changed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FunctionalMotifCluster:
    function: str
    motifs: tuple[str, ...]
    sentence_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotifFunctionAnalysis:
    occurrences: tuple[MotifFunctionOccurrence, ...]
    transitions: tuple[MotifFunctionTransition, ...]
    functional_clusters: tuple[FunctionalMotifCluster, ...]
    motif_chain: tuple[str, ...]
    role_transforming_cycle: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrences": [item.to_dict() for item in self.occurrences],
            "transitions": [item.to_dict() for item in self.transitions],
            "functional_clusters": [
                item.to_dict() for item in self.functional_clusters
            ],
            "motif_chain": list(self.motif_chain),
            "role_transforming_cycle": self.role_transforming_cycle,
        }


def _role(motif: str, sentence: str) -> str:
    other_judgment = any(
        cue in sentence
        for cue in ("誰か", "他人", "他者", "咎め", "裁", "告発")
    )
    self_reference = any(
        cue in sentence for cue in ("自分", "自己", "私の名前", "鏡の中")
    )
    blocked = any(
        cue in sentence
        for cue in (
            "開かれていない",
            "開かない",
            "避け",
            "逸ら",
            "向けられない",
            "動きが鈍",
        )
    )
    future = any(
        cue in sentence for cue in ("いつか", "次に", "だろう", "予定")
    )
    if motif in _FUNCTIONS["self_confirmation_device"]:
        if other_judgment:
            return "other_judgment_record"
        if blocked:
            return "avoided_self_evidence"
        if self_reference and future:
            return "self_judgment_gateway"
        if self_reference:
            return "self_observation_device"
        return "confirmation_record"
    if motif in _FUNCTIONS["agency_instrument"]:
        if self_reference and blocked:
            return "blocked_self_direction"
        if other_judgment:
            return "other_judgment_instrument"
        return "agency_instrument"
    if motif in _FUNCTIONS["ethical_judgment"]:
        return "ethical_consequence" if "悪意" in sentence else "ethical_standard"
    return "motif"


def analyze_motif_functions(
    sentences: Sequence[str],
) -> MotifFunctionAnalysis:
    occurrences: list[MotifFunctionOccurrence] = []
    for index, sentence in enumerate(sentences):
        for function, motifs in _FUNCTIONS.items():
            for motif in motifs:
                if motif in sentence:
                    occurrences.append(
                        MotifFunctionOccurrence(
                            sentence_index=index,
                            motif=motif,
                            function=function,
                            role=_role(motif, sentence),
                            sentence=sentence,
                        )
                    )

    transitions: list[MotifFunctionTransition] = []
    for motif in sorted({item.motif for item in occurrences}):
        items = [item for item in occurrences if item.motif == motif]
        roles = tuple(dict.fromkeys(item.role for item in items))
        if len(items) < 2:
            continue
        transitions.append(
            MotifFunctionTransition(
                motif=motif,
                function=items[0].function,
                first_index=items[0].sentence_index,
                last_index=items[-1].sentence_index,
                initial_role=items[0].role,
                final_role=items[-1].role,
                stage_count=len(roles),
                changed=len(roles) >= 2,
            )
        )

    clusters: list[FunctionalMotifCluster] = []
    for function in _FUNCTIONS:
        items = [item for item in occurrences if item.function == function]
        motifs = tuple(dict.fromkeys(item.motif for item in items))
        if len(motifs) < 2:
            continue
        clusters.append(
            FunctionalMotifCluster(
                function=function,
                motifs=motifs,
                sentence_indices=tuple(
                    dict.fromkeys(item.sentence_index for item in items)
                ),
            )
        )
    chain = tuple(dict.fromkeys(item.motif for item in occurrences))
    ending_threshold = max(1, int(len(sentences) * 0.60))
    same_motif_cycle = any(
        item.changed
        and item.first_index <= max(1, int(len(sentences) * 0.40))
        and item.last_index >= ending_threshold
        for item in transitions
    )
    functional_cycle = any(
        cluster.sentence_indices
        and cluster.sentence_indices[0] <= max(1, int(len(sentences) * 0.40))
        and cluster.sentence_indices[-1] >= ending_threshold
        and len(
            {
                item.role
                for item in occurrences
                if item.function == cluster.function
            }
        )
        >= 2
        for cluster in clusters
    )
    return MotifFunctionAnalysis(
        occurrences=tuple(occurrences),
        transitions=tuple(transitions),
        functional_clusters=tuple(clusters),
        motif_chain=chain,
        role_transforming_cycle=same_motif_cycle or functional_cycle,
    )
