"""Subject-predicate-target history for ATCT Narrative Fingerprint v0.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence

import numpy as np

from .history import EPSILON
from .predicate_families import predicate_match


_CLAUSE_BOUNDARY = re.compile(
    r"(?:のに|一方(?:で|、)?|しかし|だが|けれども|ところが|"
    r"反対に|まま[、,]|[／/])"
)
_SELF_TARGET = re.compile(
    r"自分|自身|自己|(?:私|僕|俺)(?:自身)?(?:を|に|へ|の)"
)
_OTHER = re.compile(
    r"誰か|他人|他者|相手|人(?:の|を|に|へ)|彼|彼女|部下|読者"
)
_OTHER_SUBJECT = re.compile(
    r"(?:誰か|他人|他者|相手|彼|彼女|部下)(?:が|は|には)"
)
_PASSIVE = re.compile(r"られ(?:る|た|ない|なかった)|され(?:る|た|ない)")
_NEGATIVE = re.compile(
    r"ない|なかった|なくなった|ず|拒|避け|逸ら|止ま"
)
_REFUSAL = re.compile(
    r"拒|できない|られない|向けられない|慣れなかった"
)
_AVOIDANCE = re.compile(
    r"避け|逸ら|出席しなく|開かれていない|開かない|見ない|"
    r"しなくなった|動きが鈍|止ま(?:る|った)|先延ば"
)
_DEFERRED = re.compile(
    r"まだ|いつか|決めていない|次に|もし|保留|そのうち|時期未定"
)
_PLANNED = re.compile(r"予定|つもり|だろう|でしょう|ことにする")
_POSSIBLE = re.compile(r"かもしれ|可能|できる|得る")
_PAST = re.compile(
    r"当時|以前|去年|かつて|これまで|た[。！？]?$|ていた"
)
_FUTURE = re.compile(r"いつか|次に|予定|つもり|だろう|でしょう|未来")

_MOTIFS = (
    "ノート",
    "鏡",
    "記録",
    "ページ",
    "録音データ",
    "過去の動画",
    "動画",
    "手",
    "指先",
    "ペン",
    "正義",
    "悪意",
)


@dataclass(frozen=True)
class RelationFrame:
    frame_index: int
    sentence_index: int
    clause_index: int
    sentence: str
    clause: str
    subject: str
    predicate: str
    predicate_family: str
    target: str
    polarity: str
    tense: str
    modality: str
    intensity: float
    motifs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TargetAsymmetry:
    predicate_family: str
    other_target_mean: float
    self_target_mean: float
    asymmetry: float
    other_frame_indices: tuple[int, ...]
    self_frame_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RelationFlip:
    predicate_family: str
    subject: str
    first_frame_index: int
    second_frame_index: int
    first_sentence_index: int
    second_sentence_index: int
    first_target: str
    second_target: str
    intensity_contrast: float
    temporal_distance: float
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RelationalAnalysis:
    frames: tuple[RelationFrame, ...]
    target_asymmetries: tuple[TargetAsymmetry, ...]
    relation_flips: tuple[RelationFlip, ...]
    relational_coherence: float
    relational_order_z: float
    null_mean: float
    null_std: float
    quadrant: str
    seed: int
    shuffle_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": [item.to_dict() for item in self.frames],
            "target_asymmetries": [
                item.to_dict() for item in self.target_asymmetries
            ],
            "relation_flips": [item.to_dict() for item in self.relation_flips],
            "relational_coherence": self.relational_coherence,
            "relational_order_z": self.relational_order_z,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "quadrant": self.quadrant,
            "seed": self.seed,
            "shuffle_count": self.shuffle_count,
        }


def _clauses(sentence: str) -> list[str]:
    return [
        value.strip(" 、,")
        for value in _CLAUSE_BOUNDARY.split(sentence)
        if value.strip(" 、,")
    ]


def _target(clause: str) -> str:
    if _SELF_TARGET.search(clause):
        return "self"
    if _OTHER.search(clause):
        return "other"
    if _PASSIVE.search(clause):
        return "self"
    for motif in _MOTIFS:
        if motif in clause:
            return f"object:{motif}"
    return "unspecified"


def _modality(clause: str) -> tuple[str, float]:
    if _REFUSAL.search(clause):
        return "refusal", -0.5
    if _AVOIDANCE.search(clause):
        return "avoidance", 0.0
    if _DEFERRED.search(clause):
        return "deferred", 0.2
    if _PLANNED.search(clause):
        return "planned", 0.6
    if _POSSIBLE.search(clause):
        return "possible", 0.4
    return "executed", 1.0


def _tense(clause: str) -> str:
    if _FUTURE.search(clause):
        return "future"
    if _PAST.search(clause):
        return "past"
    return "present"


def extract_relation_frames(sentences: Sequence[str]) -> tuple[RelationFrame, ...]:
    """Extract deterministic lightweight relation frames from Japanese prose."""

    frames: list[RelationFrame] = []
    for sentence_index, sentence in enumerate(sentences):
        for clause_index, clause in enumerate(_clauses(sentence)):
            match = predicate_match(clause)
            if match is None:
                continue
            target = _target(clause)
            subject = "other" if _OTHER_SUBJECT.search(clause) else "self"
            modality, intensity = _modality(clause)
            motifs = tuple(motif for motif in _MOTIFS if motif in clause)
            frames.append(
                RelationFrame(
                    frame_index=len(frames),
                    sentence_index=sentence_index,
                    clause_index=clause_index,
                    sentence=sentence,
                    clause=clause,
                    subject=subject,
                    predicate=match.predicate,
                    predicate_family=match.family,
                    target=target,
                    polarity="negative" if _NEGATIVE.search(clause) else "positive",
                    tense=_tense(clause),
                    modality=modality,
                    intensity=intensity,
                    motifs=motifs,
                )
            )
    return tuple(frames)


def _target_asymmetries(
    frames: Sequence[RelationFrame],
) -> tuple[TargetAsymmetry, ...]:
    results: list[TargetAsymmetry] = []
    families = sorted({frame.predicate_family for frame in frames})
    for family in families:
        other = [
            frame
            for frame in frames
            if frame.predicate_family == family and frame.target == "other"
        ]
        self_target = [
            frame
            for frame in frames
            if frame.predicate_family == family and frame.target == "self"
        ]
        if not other or not self_target:
            continue
        other_mean = float(np.mean([frame.intensity for frame in other]))
        self_mean = float(np.mean([frame.intensity for frame in self_target]))
        results.append(
            TargetAsymmetry(
                predicate_family=family,
                other_target_mean=other_mean,
                self_target_mean=self_mean,
                asymmetry=other_mean - self_mean,
                other_frame_indices=tuple(frame.frame_index for frame in other),
                self_frame_indices=tuple(frame.frame_index for frame in self_target),
            )
        )
    results.sort(key=lambda item: abs(item.asymmetry), reverse=True)
    return tuple(results)


def _relation_flips(
    frames: Sequence[RelationFrame],
) -> tuple[RelationFlip, ...]:
    flips: list[RelationFlip] = []
    denominator = max(len(frames) - 1, 1)
    for left_index, left in enumerate(frames):
        if left.target not in {"self", "other"}:
            continue
        for right in frames[left_index + 1 :]:
            if (
                right.target not in {"self", "other"}
                or left.target == right.target
                or left.subject != right.subject
                or left.predicate_family != right.predicate_family
            ):
                continue
            distance = (right.frame_index - left.frame_index) / denominator
            contrast = abs(left.intensity - right.intensity)
            flips.append(
                RelationFlip(
                    predicate_family=left.predicate_family,
                    subject=left.subject,
                    first_frame_index=left.frame_index,
                    second_frame_index=right.frame_index,
                    first_sentence_index=left.sentence_index,
                    second_sentence_index=right.sentence_index,
                    first_target=left.target,
                    second_target=right.target,
                    intensity_contrast=contrast,
                    temporal_distance=distance,
                    score=distance * (0.5 + 0.5 * contrast),
                )
            )
    flips.sort(key=lambda item: item.score, reverse=True)
    return tuple(flips)


def _order_statistic(
    frames: Sequence[RelationFrame],
    positions: np.ndarray,
) -> float:
    values: list[float] = []
    denominator = max(len(frames) - 1, 1)
    families = {frame.predicate_family for frame in frames}
    for family in families:
        other = [
            frame
            for frame in frames
            if frame.predicate_family == family and frame.target == "other"
        ]
        self_target = [
            frame
            for frame in frames
            if frame.predicate_family == family and frame.target == "self"
        ]
        for external in other:
            for internal in self_target:
                temporal = (
                    positions[internal.frame_index] - positions[external.frame_index]
                ) / denominator
                contrast = external.intensity - internal.intensity
                values.append(float(temporal * contrast))
    return float(np.mean(values)) if values else 0.0


def classify_quadrant(lexical_z: float, relational_z: float) -> str:
    lexical_high = lexical_z >= 2.0
    relational_high = relational_z >= 2.0
    if lexical_high and relational_high:
        return "lexical_and_relational_continuity"
    if lexical_high:
        return "lexical_repetition_dominant"
    if relational_high:
        return "deep_relational_structure"
    return "weak_or_mosaic"


def analyze_relations(
    sentences: Sequence[str],
    *,
    lexical_z: float = 0.0,
    shuffle_count: int = 200,
    seed: int = 42,
) -> RelationalAnalysis:
    """Measure target asymmetry and ordered relation transformation."""

    frames = extract_relation_frames(sentences)
    asymmetries = _target_asymmetries(frames)
    flips = _relation_flips(frames)
    if len(frames) < 2:
        null_mean = 0.0
        null_std = 0.0
        relation_z = 0.0
    else:
        positions = np.arange(len(frames), dtype=float)
        observed = _order_statistic(frames, positions)
        rng = np.random.default_rng(seed)
        null = np.asarray(
            [
                _order_statistic(frames, rng.permutation(len(frames)).astype(float))
                for _ in range(shuffle_count)
            ],
            dtype=float,
        )
        null_mean = float(np.mean(null))
        null_std = float(np.std(null))
        relation_z = float((observed - null_mean) / (null_std + EPSILON))
    asymmetry_strength = (
        float(np.mean([abs(item.asymmetry) for item in asymmetries]))
        if asymmetries
        else 0.0
    )
    flip_strength = (
        float(np.mean([item.score for item in flips[:10]])) if flips else 0.0
    )
    coherence = float(np.clip(0.70 * asymmetry_strength + 0.30 * flip_strength, 0, 1))
    return RelationalAnalysis(
        frames=frames,
        target_asymmetries=asymmetries,
        relation_flips=flips,
        relational_coherence=coherence,
        relational_order_z=relation_z,
        null_mean=null_mean,
        null_std=null_std,
        quadrant=classify_quadrant(lexical_z, relation_z),
        seed=seed,
        shuffle_count=shuffle_count,
    )
