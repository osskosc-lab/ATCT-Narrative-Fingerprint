"""Separate editorial content from lead magnets and promotional tails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Sequence


_LEAD_MAGNET = re.compile(
    r"無料(?:レポート|資料|PDF|診断|講座)|"
    r"(?:登録|ダウンロード|受け取)"
    r"(?:はこちら|ってください|できます)"
)
_CROSS_PROMOTION = re.compile(
    r"関連記事|おすすめ記事|あわせて読みたい|注目記事"
)
_FOLLOW = re.compile(
    r"フォロー(?:してください|はこちら|をお願い)|フォローする"
)
_SOCIAL_PROOF = re.compile(r"お客様の声|受講者|実績|満足度|利用者")
_BACKLINK = re.compile(r"https?://|www\.")
_HASHTAG = re.compile(r"^\s*#[^\s#]")
_SUMMARY = re.compile(r"まとめ|要するに|結論(?:として|は)|第一歩")
_FRAMEWORK = re.compile(r"段階|ステップ|モデル|メソッド|成功術|体系")
_TESTIMONY = re.compile(
    r"(?:私|僕|わたし)(?:は|が)|私自身|僕自身|"
    r"経験した|体験した|かつて"
)

PROMOTIONAL_LAYERS = {
    "lead_magnet",
    "cross_promotion",
    "follow_request",
    "backlinks",
    "hashtags",
}


@dataclass(frozen=True)
class DocumentLayerRecord:
    sentence_index: int
    layer: str
    confidence: float
    cue: str
    sentence: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentLayerAnalysis:
    records: tuple[DocumentLayerRecord, ...]
    promotion_start_index: int | None
    editorial_end_index: int | None
    content_indices: tuple[int, ...]
    promotional_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _match(pattern: re.Pattern[str], sentence: str) -> str:
    found = pattern.search(sentence)
    return found.group(0) if found else ""


def _promotional_kind(sentence: str) -> tuple[str, str] | None:
    patterns = (
        ("lead_magnet", _LEAD_MAGNET),
        ("cross_promotion", _CROSS_PROMOTION),
        ("follow_request", _FOLLOW),
        ("backlinks", _BACKLINK),
        ("hashtags", _HASHTAG),
    )
    for layer, pattern in patterns:
        cue = _match(pattern, sentence)
        if cue:
            return layer, cue
    return None


def analyze_document_layers(
    sentences: Sequence[str],
) -> DocumentLayerAnalysis:
    late_boundary = max(1, int(len(sentences) * 0.60))
    promo_candidates: list[int] = []
    for index, sentence in enumerate(sentences):
        promotional = _promotional_kind(sentence)
        if promotional is None:
            continue
        layer, _ = promotional
        if layer != "backlinks" or index >= late_boundary:
            promo_candidates.append(index)
    promotion_start = min(promo_candidates) if promo_candidates else None
    records: list[DocumentLayerRecord] = []
    for index, sentence in enumerate(sentences):
        promotional = _promotional_kind(sentence)
        if promotion_start is not None and index >= promotion_start:
            if promotional is not None:
                layer, cue = promotional
            elif _SOCIAL_PROOF.search(sentence):
                layer, cue = "social_proof", _match(_SOCIAL_PROOF, sentence)
            else:
                layer, cue = "cross_promotion", ""
            confidence = 1.0 if cue else 0.65
        elif _SOCIAL_PROOF.search(sentence):
            layer, cue, confidence = (
                "social_proof",
                _match(_SOCIAL_PROOF, sentence),
                0.85,
            )
        elif _SUMMARY.search(sentence):
            layer, cue, confidence = "summary", _match(_SUMMARY, sentence), 0.75
        elif _FRAMEWORK.search(sentence):
            layer, cue, confidence = (
                "framework_explanation",
                _match(_FRAMEWORK, sentence),
                0.75,
            )
        elif _TESTIMONY.search(sentence):
            layer, cue, confidence = (
                "personal_testimony",
                _match(_TESTIMONY, sentence),
                0.70,
            )
        else:
            layer, cue, confidence = "editorial_body", "", 0.60
        records.append(
            DocumentLayerRecord(index, layer, confidence, cue, sentence)
        )
    content_indices = tuple(
        index
        for index in range(len(sentences))
        if promotion_start is None or index < promotion_start
    )
    promotional_indices = tuple(
        index
        for index in range(len(sentences))
        if promotion_start is not None and index >= promotion_start
    )
    return DocumentLayerAnalysis(
        records=tuple(records),
        promotion_start_index=promotion_start,
        editorial_end_index=(
            content_indices[-1] if content_indices else None
        ),
        content_indices=content_indices,
        promotional_indices=promotional_indices,
    )
