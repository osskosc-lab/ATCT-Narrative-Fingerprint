"""Deterministic predicate families for the lightweight v0.5 relation channel."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PredicateMatch:
    predicate: str
    family: str


_FAMILY_PATTERNS = (
    (
        "judge_or_evaluate",
        re.compile(
            r"咎め|裁(?:く|い|か)|検討|評価|批判|点検|確かめ|"
            r"判断|審判|責任|記録|見(?:る|た|ない|られ)"
        ),
    ),
    (
        "self_check",
        re.compile(r"開(?:く|い|かれ|け)|書(?:く|い)|読(?:む|ん)|見返"),
    ),
    (
        "direct_or_apply",
        re.compile(r"向け|指(?:す|し)|当て|及ぼ|与え|受け取"),
    ),
    (
        "care_or_forgive",
        re.compile(r"優し|許(?:す|し)|守(?:る|っ)|労わ|いたわ"),
    ),
    (
        "act_or_change",
        re.compile(
            r"変(?:わ|え)|直(?:す|し)|始め|進(?:む|ん)|行(?:く|っ)"
        ),
    ),
)


def predicate_match(text: str) -> PredicateMatch | None:
    """Return the first explicit predicate family found in one clause."""

    for family, pattern in _FAMILY_PATTERNS:
        match = pattern.search(text)
        if match:
            return PredicateMatch(predicate=match.group(0), family=family)
    return None
