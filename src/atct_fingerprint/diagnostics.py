"""Surface confounds and editing-oriented risk flags."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from .history import EPSILON, normalize_rows


def confound_audit(sentences: Sequence[str]) -> dict[str, object]:
    lengths = np.asarray([len(re.sub(r"\s+", "", text)) for text in sentences])
    short_rate = float(np.mean(lengths <= 12))
    length_cv = float(np.std(lengths) / max(float(np.mean(lengths)), EPSILON))
    joined = "".join(sentences)
    bigrams = [joined[index : index + 2] for index in range(max(len(joined) - 1, 0))]
    lexical_diversity = len(set(bigrams)) / max(len(bigrams), 1)
    technical = re.findall(
        r"[A-Za-zΑ-ωα-ω][A-Za-z0-9_Α-ωα-ω]*|[A-Z]{2,}|\d+(?:\.\d+)?",
        joined,
    )
    technical_density = len(technical) / max(len(sentences), 1)
    warnings: list[str] = []
    if len(sentences) < 8:
        warnings.append("low_sentence_count")
    if short_rate > 0.50:
        warnings.append("short_sentence_concentration")
    if length_cv > 0.90:
        warnings.append("high_sentence_length_variance")
    return {
        "sentence_count": len(sentences),
        "document_characters": int(np.sum(lengths)),
        "median_sentence_length": float(np.median(lengths)),
        "short_sentence_rate": short_rate,
        "sentence_length_cv": length_cv,
        "lexical_diversity": lexical_diversity,
        "technical_term_density": technical_density,
        "warnings": warnings,
    }


def theme_cohesion(vectors: np.ndarray) -> tuple[float, np.ndarray]:
    matrix = normalize_rows(vectors)
    center = np.mean(matrix, axis=0)
    center /= max(np.linalg.norm(center), EPSILON)
    similarities = np.clip(matrix @ center, -1.0, 1.0)
    return float(np.mean(similarities)), similarities


def editing_risks(
    sentences: Sequence[str],
    *,
    theme_similarities: np.ndarray,
    sentence_rows: Sequence[Mapping[str, object]],
    duplicate_rate: float,
    confounds: Mapping[str, object],
) -> list[dict[str, object]]:
    risks: list[dict[str, object]] = []
    if duplicate_rate > 0.10:
        risks.append({"risk": "semantic_duplication", "severity": duplicate_rate})
    low_theme = [
        index for index, value in enumerate(theme_similarities) if value < 0.20
    ]
    if low_theme:
        risks.append({"risk": "abrupt_theme_deviation", "sentences": low_theme})
    leaps = [
        int(row["index"])
        for row in sentence_rows
        if row["history_consistency"] is not None
        and row["turn_z"] is not None
        and float(row["history_consistency"]) < 0.05
        and float(row["turn_z"]) > 2.0
    ]
    if leaps:
        risks.append({"risk": "unexplained_leap", "sentences": leaps})
    if float(confounds["short_sentence_rate"]) > 0.50:
        risks.append(
            {
                "risk": "short_sentence_overuse",
                "severity": confounds["short_sentence_rate"],
            }
        )
    if float(confounds["technical_term_density"]) > 2.0:
        risks.append(
            {
                "risk": "technical_term_concentration",
                "severity": confounds["technical_term_density"],
            }
        )
    opening = " ".join(sentences[:2])
    if re.search(r"^(結論|要するに|In conclusion|Therefore)", opening, re.I):
        risks.append({"risk": "conclusion_first", "sentences": [0]})
    if len(theme_similarities) >= 4 and float(np.mean(theme_similarities[:2])) < 0.2:
        risks.append({"risk": "opening_body_mismatch", "sentences": [0, 1]})
    return risks
