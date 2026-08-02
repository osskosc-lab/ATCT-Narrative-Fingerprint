"""Extract surface, intermediate, and deep desire frames with evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal, Sequence


DepthLevel = Literal["surface", "intermediate", "deep"]
Polarity = Literal["affirmed", "negated", "qualified"]

_DESIRE = re.compile(
    r"(?P<object>[^。！？]{0,32}?)(?P<action>転職|退職|辞職|"
    r"働き方を変え|環境を変え|仕事を変え|辞め|やめ|変え|"
    r"選び直|書き換え|定義し直|定義)(?:し)?たい"
)
_DEEP = re.compile(
    r"本当(?:に|は)|本当に必要なのは|重要なのは|深(?:い|層)|"
    r"成功(?:の)?(?:判断基準|基準|OS)|価値基準|判断基準|"
    r"自分で(?:定義|選び直|決め|書き換|更新)"
)
_INTERMEDIATE = re.compile(
    r"働き方|働く環境|職場|勤務|仕事の環境|現在の仕事|"
    r"生活と仕事|労働条件"
)
_NONEXCLUSIVE = re.compile(r"だけではない|だけじゃない|だけでなく")
_NEGATED = re.compile(r"んじゃない|のではない|ではない|じゃない")
_QUALIFIED = re.compile(r"かもしれない|場合がある|一つの|だけではない")
_DEEP_OBJECT = re.compile(
    r"((?:成功|価値|判断)[^。！？]{0,28}?(?:基準|OS|体系)[^。！？]{0,24})"
)


@dataclass(frozen=True)
class DesireFrame:
    subject: str
    action: str
    object: str
    depth: DepthLevel
    polarity: Polarity
    evidence_span: str
    sentence_index: int
    confidence: float
    rule_or_model: str = "rule"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DesireAnalysis:
    frames: tuple[DesireFrame, ...]
    surface_desire: str
    intermediate_desire: str
    deep_desire: str
    surface_support: float
    intermediate_support: float
    deep_support: float
    depth_score: float
    evidence_support: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _polarity(sentence: str) -> Polarity:
    if _NONEXCLUSIVE.search(sentence) or _QUALIFIED.search(sentence):
        return "qualified"
    if _NEGATED.search(sentence):
        return "negated"
    return "affirmed"


def _depth(sentence: str) -> DepthLevel:
    if _DEEP.search(sentence):
        return "deep"
    if _INTERMEDIATE.search(sentence):
        return "intermediate"
    return "surface"


def _label(frame: DesireFrame) -> str:
    if frame.action == "転職":
        return "転職したい"
    if frame.depth == "deep" and re.search(r"成功|判断基準|価値基準|OS", frame.object):
        return "成功の判断基準を自分で選び直したい"
    if frame.depth == "intermediate":
        return "現在の働き方を変えたい"
    return f"{frame.object}{frame.action}たい".strip()


def analyze_desires(sentences: Sequence[str]) -> DesireAnalysis:
    """Build a conservative desire hierarchy from explicit Japanese cues."""

    frames: list[DesireFrame] = []
    for index, sentence in enumerate(sentences):
        matches = list(_DESIRE.finditer(sentence))
        if not matches and _DEEP.search(sentence) and re.search(
            r"必要|望|求め|選び直|書き換|定義", sentence
        ):
            deep_object = _DEEP_OBJECT.search(sentence)
            frames.append(
                DesireFrame(
                    subject="書き手または本文の主体",
                    action="再定義",
                    object=(
                        deep_object.group(1)
                        if deep_object
                        else sentence.strip("。！？")
                    ),
                    depth="deep",
                    polarity=_polarity(sentence),
                    evidence_span=sentence,
                    sentence_index=index,
                    confidence=0.90,
                )
            )
            continue
        for match in matches:
            depth = _depth(sentence)
            object_text = match.group("object").strip(" 、，,「」『』")
            if depth == "deep":
                deep_object = _DEEP_OBJECT.search(sentence)
                if deep_object:
                    object_text = deep_object.group(1)
            frames.append(
                DesireFrame(
                    subject="書き手または本文の主体",
                    action=match.group("action"),
                    object=object_text,
                    depth=depth,
                    polarity=_polarity(sentence),
                    evidence_span=sentence,
                    sentence_index=index,
                    confidence=(0.92 if depth == "deep" else 0.86),
                )
            )

    def support(depth: DepthLevel) -> float:
        values = []
        for frame in frames:
            if frame.depth != depth:
                continue
            multiplier = {
                "affirmed": 1.0,
                "qualified": 0.70,
                "negated": 0.10,
            }[frame.polarity]
            values.append(frame.confidence * multiplier)
        return min(1.0, sum(values))

    surface_support = support("surface")
    intermediate_support = support("intermediate")
    deep_support = support("deep")
    depth_score = max(0.0, min(1.0, deep_support - 0.35 * surface_support))
    deep_count = sum(frame.depth == "deep" for frame in frames)
    evidence_support = min(1.0, 0.45 * deep_count + 0.25 * intermediate_support)

    def best(depth: DepthLevel) -> str:
        candidates = [frame for frame in frames if frame.depth == depth]
        if not candidates:
            return ""
        chosen = max(
            candidates,
            key=lambda item: (item.confidence, item.sentence_index),
        )
        return _label(chosen)

    return DesireAnalysis(
        frames=tuple(frames),
        surface_desire=best("surface"),
        intermediate_desire=best("intermediate"),
        deep_desire=best("deep"),
        surface_support=surface_support,
        intermediate_support=intermediate_support,
        deep_support=deep_support,
        depth_score=depth_score,
        evidence_support=evidence_support,
    )
