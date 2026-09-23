"""Audit evidence for asserted stage order and order alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_ARROW = re.compile(r"\s*(?:→|⇒|->)\s*")
_BRAND = re.compile(
    r"([一-龠々ぁ-んァ-ンA-Za-z0-9・]{2,24}"
    r"(?:術|法|メソッド|モデル|システム))"
)
_ORDER_CUE = re.compile(r"順番|段階|ステップ|まず|次に|最後に")
_STAGE_CONTEXT = re.compile(
    r"整え|休め|回復|見直|定め|段階|ステップ|"
    r"まず|次に|その後|最後"
)
_CHECKS = {
    "measurable_stages": re.compile(r"測定|指標|尺度|数値|スコア"),
    "completion_criteria": re.compile(
        r"完了条件|完了したら|基準|満たしたら"
    ),
    "order_comparison": re.compile(r"順番を入れ替|順序を変|比較|対照"),
    "concurrency_excluded": re.compile(
        r"同時進行を排除|並行を排除|同時には"
    ),
    "reverse_effects": re.compile(
        r"逆方向|相互作用|双方向|フィードバック"
    ),
    "exceptions": re.compile(
        r"例外|個人差|場合がある|ただし|必ずしも"
    ),
}


def _pair_alignment(values: Sequence[int]) -> float:
    if len(values) < 2:
        return 0.5
    concordant = 0
    pairs = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            pairs += 1
            if values[left] < values[right]:
                concordant += 1
    return float(concordant / pairs)


@dataclass(frozen=True)
class SequenceClaim:
    sentence_index: int
    stages: tuple[str, ...]
    branded_name: str
    alignment_score: float
    first_mention_indices: tuple[int | None, ...]
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceAudit:
    claims: tuple[SequenceClaim, ...]
    checks: dict[str, bool]
    evidence_count: int
    order_claim_count: int
    necessity_score: float
    mean_alignment: float
    insufficient_evidence_warning: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _extract_stages(sentence: str) -> tuple[str, ...]:
    if _ARROW.search(sentence):
        parts = _ARROW.split(sentence)
    elif _ORDER_CUE.search(sentence) and sentence.count("・") >= 2:
        parts = sentence.split("・")
    else:
        return ()
    cleaned_items: list[str] = []
    for part in parts:
        value = re.sub(r"[。！？].*$", "", part)
        value = re.sub(
            r"^(?:.*?(?:順番は|段階は|ステップは|いう|として))",
            "",
            value,
        )
        value = re.sub(
            r"[」』）)]?(?:の順番|の段階|のステップ).*$",
            "",
            value,
        )
        value = value.strip(" 、,:：「」『』()（）")
        cleaned_items.append(value)
    cleaned = tuple(cleaned_items)
    return tuple(item for item in cleaned if item)[:12]


def _first_mentions(
    sentences: Sequence[str],
    stages: Sequence[str],
    claim_index: int,
) -> tuple[int | None, ...]:
    values: list[int | None] = []
    for stage in stages:
        found = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if index != claim_index
                and stage in sentence
                and (len(stage) > 1 or _STAGE_CONTEXT.search(sentence))
            ),
            None,
        )
        values.append(found)
    return tuple(values)


def analyze_sequence_claims(sentences: Sequence[str]) -> SequenceAudit:
    joined = "\n".join(sentences)
    claims: list[SequenceClaim] = []
    latest_brand = ""
    for index, sentence in enumerate(sentences):
        brand_match = _BRAND.search(sentence)
        if brand_match:
            latest_brand = brand_match.group(1)
        stages = _extract_stages(sentence)
        if len(stages) < 3:
            continue
        mentions = _first_mentions(sentences, stages, index)
        comparable = [value for value in mentions if value is not None]
        claims.append(
            SequenceClaim(
                sentence_index=index,
                stages=stages,
                branded_name=latest_brand,
                alignment_score=_pair_alignment(comparable),
                first_mention_indices=mentions,
                sentence=sentence,
            )
        )
    checks = {
        name: bool(pattern.search(joined)) for name, pattern in _CHECKS.items()
    }
    evidence_count = sum(checks.values())
    order_claim_count = sum(bool(_ORDER_CUE.search(item)) for item in sentences)
    necessity_score = float(evidence_count / len(_CHECKS))
    mean_alignment = (
        float(sum(item.alignment_score for item in claims) / len(claims))
        if claims
        else 0.0
    )
    return SequenceAudit(
        claims=tuple(claims),
        checks=checks,
        evidence_count=evidence_count,
        order_claim_count=order_claim_count,
        necessity_score=necessity_score,
        mean_alignment=mean_alignment,
        insufficient_evidence_warning=bool(claims and necessity_score < 0.5),
    )
