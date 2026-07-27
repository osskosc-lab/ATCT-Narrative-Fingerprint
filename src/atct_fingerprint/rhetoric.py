"""Evidence-oriented rhetorical diagnostics for ATCT v0.4.

These diagnostics retrieve candidate passages for human inspection.  They do
not establish authorial intent, logical truth, or writing quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

import numpy as np

from .history import normalize_rows


_QUOTED_PHRASE = re.compile(r"[「『“\"]([^」』”\"]{1,24})[」』”\"]")
_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_JAPANESE_RUN = re.compile(r"[一-龥ぁ-んァ-ヶー]{2,}")
_KANJI_RUN = re.compile(r"[一-龥]{2,8}")
_QUESTION = re.compile(r"[?？]|(?:だろうか|でしょうか|なのか|のか)[。！？]?$")
_ANSWER_CUE = re.compile(
    r"ではない|のではない|だから|つまり|答え|証明ではない|"
    r"\b(?:no|not|because|therefore|answer)\b",
    re.I,
)
_QUALIFIER = re.compile(
    r"もし|仮定|とする|と定義|なら|思考実験|限定された意味|"
    r"\b(?:if|assuming|suppose|defined as|within this)\b",
    re.I,
)
_EXPLICIT_SCOPE = re.compile(
    r"この(?:思考実験|仮定|定義|文脈|範囲)|"
    r"(?:思考実験|仮定|定義|文脈|範囲)(?:の中|では|において)|"
    r"と定義するなら|限定された意味|"
    r"\b(?:in this thought experiment|under this assumption|within this scope)\b",
    re.I,
)
_CATEGORICAL = re.compile(
    r"(?:だ|である|ではない|しない|求めない|できない|なる|でない)"
    r"[。！？]?$|"
    r"\b(?:is|are|does not|cannot|never)\b",
    re.I,
)
_ANCHOR = re.compile(r"[一-龥]{1,8}|[ァ-ヶー]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}")
_GENERIC_ANCHORS = {
    "思考",
    "実験",
    "仮定",
    "定義",
    "意味",
    "存在",
    "場合",
    "こと",
    "もの",
    "this",
    "that",
    "with",
    "from",
}
_STOP_MOTIFS = {
    "という",
    "こと",
    "もの",
    "ため",
    "それ",
    "これ",
    "して",
    "ます",
    "です",
    "ある",
    "いる",
    "ない",
    "から",
    "ので",
    "よう",
    "なる",
    "する",
    "今日",
    "自分",
    "私",
    "人",
}
_EDGE_PARTICLE = re.compile(
    r"^(?:は|が|を|に|で|と|へ|も|の)|(?:は|が|を|に|で|と|へ|も|の)$"
)


@dataclass(frozen=True)
class MotifRoleTransition:
    motif: str
    first_index: int
    second_index: int
    first_sentence: str
    second_sentence: str
    context_similarity: float
    role_shift: float
    relative_distance: float
    kind: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MotifRoleAnalysis:
    candidate_count: int
    transformed_return_count: int
    transitions: tuple[MotifRoleTransition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "transformed_return_count": self.transformed_return_count,
            "transitions": [item.to_dict() for item in self.transitions],
        }


@dataclass(frozen=True)
class ConceptBranch:
    sentence_index: int
    paired_sentence_index: int | None
    marker: str
    left_branch: str
    right_branch: str
    lexical_similarity: float
    lexical_divergence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionAnswerPair:
    question_index: int
    answer_index: int
    question: str
    answer: str
    semantic_similarity: float
    lexical_overlap: float
    answer_cue: bool
    score: float
    kind: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionAnswerClosure:
    opening_question_count: int
    closing_answer_count: int
    best_score: float
    closed: bool
    pairs: tuple[QuestionAnswerPair, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "opening_question_count": self.opening_question_count,
            "closing_answer_count": self.closing_answer_count,
            "best_score": self.best_score,
            "closed": self.closed,
            "pairs": [item.to_dict() for item in self.pairs],
        }


@dataclass(frozen=True)
class ClaimScopeFinding:
    qualifier_index: int
    assertion_index: int
    qualifier_sentence: str
    assertion_sentence: str
    shared_anchors: tuple[str, ...]
    semantic_similarity: float
    warning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimScopeAudit:
    qualifier_count: int
    findings: tuple[ClaimScopeFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "qualifier_count": self.qualifier_count,
            "findings": [item.to_dict() for item in self.findings],
        }


def _normalized_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


def _char_ngrams(value: str, sizes: tuple[int, ...] = (2, 3)) -> set[str]:
    cleaned = _normalized_text(value)
    return {
        cleaned[index : index + size]
        for size in sizes
        for index in range(max(len(cleaned) - size + 1, 0))
    }


def _lexical_overlap(left: str, right: str) -> float:
    left_terms = _char_ngrams(left)
    right_terms = _char_ngrams(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _motif_candidates(
    sentences: Sequence[str],
    *,
    max_candidates: int = 16,
) -> list[tuple[str, tuple[int, ...]]]:
    explicit: set[str] = set()
    occurrences: dict[str, set[int]] = {}
    for index, sentence in enumerate(sentences):
        for phrase in _QUOTED_PHRASE.findall(sentence):
            cleaned = phrase.strip()
            if cleaned:
                explicit.add(cleaned)
        for term in _LATIN_TERM.findall(sentence):
            occurrences.setdefault(term.casefold(), set()).add(index)
        for term in _KANJI_RUN.findall(sentence):
            if term not in _STOP_MOTIFS:
                occurrences.setdefault(term, set()).add(index)
        for run in _JAPANESE_RUN.findall(sentence):
            for size in range(2, min(6, len(run)) + 1):
                for start in range(len(run) - size + 1):
                    term = run[start : start + size]
                    if (
                        term not in _STOP_MOTIFS
                        and re.search(r"[一-龥ァ-ヶ]", term)
                        and not _EDGE_PARTICLE.search(term)
                    ):
                        occurrences.setdefault(term, set()).add(index)
    for phrase in explicit:
        for index, sentence in enumerate(sentences):
            if phrase in sentence:
                occurrences.setdefault(phrase, set()).add(index)

    repeated = [
        (term, tuple(sorted(indices)))
        for term, indices in occurrences.items()
        if len(indices) >= 2
    ]
    repeated.sort(
        key=lambda item: (
            item[0] in explicit,
            (item[1][-1] - item[1][0]) * len(item[0]),
            len(item[0]),
        ),
        reverse=True,
    )
    selected: list[tuple[str, tuple[int, ...]]] = []
    for term, indices in repeated:
        if any(
            term in chosen
            and indices == chosen_indices
            for chosen, chosen_indices in selected
        ):
            continue
        selected.append((term, indices))
        if len(selected) >= max_candidates:
            break
    return selected


def analyze_motif_roles(
    sentences: Sequence[str],
    vectors: np.ndarray,
    *,
    min_role_shift: float = 0.20,
    min_return_distance: float = 0.40,
) -> MotifRoleAnalysis:
    """Track repeated lexical motifs whose surrounding sentence meaning changes."""

    matrix = normalize_rows(vectors)
    candidates = _motif_candidates(sentences)
    transitions: list[MotifRoleTransition] = []
    denominator = max(len(sentences) - 1, 1)
    for motif, indices in candidates:
        pairs = list(zip(indices, indices[1:]))
        if len(indices) > 2:
            pairs.append((indices[0], indices[-1]))
        for first, second in dict.fromkeys(pairs):
            similarity = float(np.clip(np.dot(matrix[first], matrix[second]), -1, 1))
            shift = 1.0 - similarity
            distance = (second - first) / denominator
            identical = _normalized_text(sentences[first]) == _normalized_text(
                sentences[second]
            )
            if identical or shift < 0.15:
                kind = "stable_role"
            elif shift >= min_role_shift and distance >= min_return_distance:
                kind = "return_with_role_shift"
            else:
                kind = "context_shift"
            transitions.append(
                MotifRoleTransition(
                    motif=motif,
                    first_index=first,
                    second_index=second,
                    first_sentence=sentences[first],
                    second_sentence=sentences[second],
                    context_similarity=similarity,
                    role_shift=shift,
                    relative_distance=distance,
                    kind=kind,
                )
            )
    transitions.sort(
        key=lambda item: (
            item.kind == "return_with_role_shift",
            item.relative_distance * item.role_shift,
        ),
        reverse=True,
    )
    return MotifRoleAnalysis(
        candidate_count=len(candidates),
        transformed_return_count=sum(
            item.kind == "return_with_role_shift" for item in transitions
        ),
        transitions=tuple(transitions),
    )


_BRANCH_PATTERNS = (
    re.compile(r"(.{1,100}?)(ではなくて|ではなく|じゃなく|というより)(.{1,100})"),
    re.compile(r"(.{1,100}?)(それとも|あるいは)(.{1,100})"),
    re.compile(r"(.{1,100}?)(\bnot\b.+?\bbut\b|\brather than\b)(.{1,100})", re.I),
)


def detect_concept_branches(sentences: Sequence[str]) -> tuple[ConceptBranch, ...]:
    """Retrieve explicit contrast or alternative branches."""

    branches: list[ConceptBranch] = []
    for index, sentence in enumerate(sentences):
        for pattern in _BRANCH_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            left = match.group(1).strip(" 、。！？?!")
            right = match.group(3).strip(" 、。！？?!")
            similarity = _lexical_overlap(left, right)
            branches.append(
                ConceptBranch(
                    sentence_index=index,
                    paired_sentence_index=None,
                    marker=match.group(2),
                    left_branch=left,
                    right_branch=right,
                    lexical_similarity=similarity,
                    lexical_divergence=1.0 - similarity,
                )
            )
            break
        if (
            index + 1 < len(sentences)
            and _QUESTION.search(sentence)
            and re.match(r"^\s*(それとも|あるいは)", sentences[index + 1])
        ):
            next_sentence = sentences[index + 1]
            marker_match = re.match(r"^\s*(それとも|あるいは)[、,\s]*", next_sentence)
            assert marker_match is not None
            left = sentence.strip(" 、。！？?!")
            right = next_sentence[marker_match.end() :].strip(" 、。！？?!")
            similarity = _lexical_overlap(left, right)
            branches.append(
                ConceptBranch(
                    sentence_index=index,
                    paired_sentence_index=index + 1,
                    marker=marker_match.group(1),
                    left_branch=left,
                    right_branch=right,
                    lexical_similarity=similarity,
                    lexical_divergence=1.0 - similarity,
                )
            )
    return tuple(branches)


def analyze_qa_closure(
    sentences: Sequence[str],
    vectors: np.ndarray,
    *,
    threshold: float = 0.25,
) -> QuestionAnswerClosure:
    """Match opening questions to semantically related closing statements."""

    matrix = normalize_rows(vectors)
    opening_end = max(2, math.ceil(len(sentences) * 0.40))
    closing_start = min(len(sentences) - 1, math.floor(len(sentences) * 0.60))
    questions = [
        index
        for index in range(opening_end)
        if _QUESTION.search(sentences[index])
    ]
    answers = list(range(closing_start, len(sentences)))
    pairs: list[QuestionAnswerPair] = []
    for question_index in questions:
        for answer_index in answers:
            if answer_index <= question_index:
                continue
            semantic = float(
                np.clip(np.dot(matrix[question_index], matrix[answer_index]), -1, 1)
            )
            lexical = _lexical_overlap(
                sentences[question_index], sentences[answer_index]
            )
            cue = bool(_ANSWER_CUE.search(sentences[answer_index]))
            score = 0.65 * max(semantic, 0.0) + 0.20 * lexical + 0.15 * cue
            pairs.append(
                QuestionAnswerPair(
                    question_index=question_index,
                    answer_index=answer_index,
                    question=sentences[question_index],
                    answer=sentences[answer_index],
                    semantic_similarity=semantic,
                    lexical_overlap=lexical,
                    answer_cue=cue,
                    score=float(score),
                    kind="candidate_answer",
                )
            )
    pairs.sort(key=lambda item: item.score, reverse=True)
    best_score = pairs[0].score if pairs else 0.0
    ranked = tuple(
        QuestionAnswerPair(
            **{
                **item.to_dict(),
                "kind": (
                    "question_answer_closure"
                    if rank == 0 and best_score >= threshold
                    else "candidate_answer"
                ),
            }
        )
        for rank, item in enumerate(pairs[:10])
    )
    return QuestionAnswerClosure(
        opening_question_count=len(questions),
        closing_answer_count=len(answers),
        best_score=best_score,
        closed=best_score >= threshold,
        pairs=ranked,
    )


def _anchors(sentence: str) -> set[str]:
    return {
        token.casefold()
        for token in _ANCHOR.findall(sentence)
        if token.casefold() not in _GENERIC_ANCHORS
    }


def audit_claim_scope(
    sentences: Sequence[str],
    vectors: np.ndarray,
    *,
    max_distance: int = 12,
) -> ClaimScopeAudit:
    """Flag a scoped premise followed by a related unscoped assertion."""

    matrix = normalize_rows(vectors)
    qualifier_indices = [
        index for index, sentence in enumerate(sentences) if _QUALIFIER.search(sentence)
    ]
    findings: list[ClaimScopeFinding] = []
    for qualifier_index in qualifier_indices:
        qualifier_anchors = _anchors(sentences[qualifier_index])
        for assertion_index in range(
            qualifier_index + 1,
            min(len(sentences), qualifier_index + max_distance + 1),
        ):
            assertion = sentences[assertion_index]
            if not _CATEGORICAL.search(assertion) or _EXPLICIT_SCOPE.search(assertion):
                continue
            shared = tuple(sorted(qualifier_anchors & _anchors(assertion)))
            similarity = float(
                np.clip(
                    np.dot(matrix[qualifier_index], matrix[assertion_index]),
                    -1,
                    1,
                )
            )
            if not shared and similarity < 0.20:
                continue
            findings.append(
                ClaimScopeFinding(
                    qualifier_index=qualifier_index,
                    assertion_index=assertion_index,
                    qualifier_sentence=sentences[qualifier_index],
                    assertion_sentence=assertion,
                    shared_anchors=shared,
                    semantic_similarity=similarity,
                    warning="qualifier_dropped",
                )
            )
            break
    return ClaimScopeAudit(
        qualifier_count=len(qualifier_indices),
        findings=tuple(findings),
    )
