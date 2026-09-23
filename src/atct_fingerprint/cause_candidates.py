"""Retain supported, possible, rejected, and unexamined cause candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal, Sequence


CauseType = Literal[
    "value_conflict",
    "work_environment",
    "income",
    "health",
    "relationship",
    "job_fit",
    "family_constraint",
    "social_norm",
    "unknown",
]
CauseStatus = Literal["supported", "possible", "rejected", "not_examined"]

_CAUSE_RULES: tuple[tuple[CauseType, str, re.Pattern[str]], ...] = (
    (
        "value_conflict",
        "価値基準の不一致",
        re.compile(r"成功(?:の)?(?:OS|基準|判断基準)|価値基準|価値観|判断基準"),
    ),
    (
        "work_environment",
        "労働条件・職場環境",
        re.compile(
            r"労働条件|職場環境|勤務条件|長時間労働|リモートワーク|"
            r"育休制度|時短勤務|副業解禁|雇用制度|制度"
        ),
    ),
    ("income", "賃金・収入", re.compile(r"賃金|収入|給与|給料|報酬")),
    (
        "health",
        "健康",
        re.compile(r"健康|体調|睡眠|病気|身心|疲労|身体が(?:落ち|先に)|体調不良"),
    ),
    (
        "relationship",
        "人間関係",
        re.compile(r"人間関係|上司|同僚|ハラスメント|孤立"),
    ),
    (
        "job_fit",
        "職務適性",
        re.compile(r"職務適性|適性|向いていない|仕事内容|仕事が合わない"),
    ),
    (
        "family_constraint",
        "家庭事情",
        re.compile(r"家庭|家族|育児|子育て|介護"),
    ),
    (
        "social_norm",
        "社会規範",
        re.compile(r"社会規範|世間|社会から与え|会社から与え|標準的な成功|普通は"),
    ),
)
_POSSIBLE = re.compile(r"かもしれない|可能性|あり得る|場合がある|一因")
_NONEXCLUSIVE = re.compile(r"だけ(?:が)?原因ではない|だけではない")
_REJECTED = re.compile(r"(?<!だけが)原因ではない|無関係|影響しない")
_SUPPORTED = re.compile(
    r"原因|理由|ため|によって|影響|制約|不一致|重要|関係する|左右する|"
    r"変える|可能にする|無視できない|もある|も含む"
)
_LISTISH = re.compile(r"(?:も[、,]).{0,24}も")
_ENUMERATION = re.compile(r"三つに分け|理由は(?:、)?だいたい三|一つ目は")
_OUTCOME = re.compile(
    r"([。！？]{1,50}?(?:転職したい|働き方を変えたい|苦しい|"
    r"合わない|続けられない|辞めたい|違和感))"
)
