"""Asymmetric forward/backward conditional-likelihood diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DirectionalityResult:
    method: str
    forward_mean_log_likelihood: float
    backward_mean_log_likelihood: float
    asymmetry: float
    sentence_deltas: tuple[float | None, ...]
    disclaimer: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _character_ngrams(text: str, n: int = 2) -> list[str]:
    normalized = re.sub(r"\s+", "", text).lower()
    if not normalized:
        return []
    padded = f"^{normalized}$"
    if len(padded) < n:
        return [padded]
    return [padded[index : index + n] for index in range(len(padded) - n + 1)]


def _conditional_log_likelihood(
    target: str,
    context: str,
    vocabulary: set[str],
    *,
    ngram: int,
    alpha: float,
) -> float:
    target_tokens = _character_ngrams(target, ngram)
    if not target_tokens:
        return 0.0
    context_counts = Counter(_character_ngrams(context, ngram))
    vocabulary_size = max(len(vocabulary), 1)
    denominator = sum(context_counts.values()) + alpha * vocabulary_size
    values = [
        math.log((context_counts[token] + alpha) / denominator)
        for token in target_tokens
    ]
    return float(np.mean(values))


def conditional_directionality(
    sentences: Sequence[str],
    *,
    context_window: int = 5,
    ngram: int = 2,
    alpha: float = 0.5,
) -> DirectionalityResult:
    """Compare each sentence under preceding and following lexical contexts.

    The score is a lightweight conditional-likelihood baseline suitable for CI.
    It is directional because the conditioning sets differ. It is not a
    substitute for a frozen autoregressive language model in confirmatory work.
    """

    if context_window < 1:
        raise ValueError("context_window must be positive")
    if ngram < 1:
        raise ValueError("ngram must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    vocabulary = {
        token
        for sentence in sentences
        for token in _character_ngrams(sentence, ngram)
    }
    forward: list[float] = []
    backward: list[float] = []
    deltas: list[float | None] = [None] * len(sentences)
    for index, sentence in enumerate(sentences):
        left = sentences[max(0, index - context_window) : index]
        right = sentences[index + 1 : index + 1 + context_window]
        if not left or not right:
            continue
        forward_value = _conditional_log_likelihood(
            sentence,
            " ".join(left),
            vocabulary,
            ngram=ngram,
            alpha=alpha,
        )
        backward_value = _conditional_log_likelihood(
            sentence,
            " ".join(right),
            vocabulary,
            ngram=ngram,
            alpha=alpha,
        )
        forward.append(forward_value)
        backward.append(backward_value)
        deltas[index] = forward_value - backward_value
    forward_mean = float(np.mean(forward)) if forward else 0.0
    backward_mean = float(np.mean(backward)) if backward else 0.0
    return DirectionalityResult(
        method="conditional_character_ngram",
        forward_mean_log_likelihood=forward_mean,
        backward_mean_log_likelihood=backward_mean,
        asymmetry=forward_mean - backward_mean,
        sentence_deltas=tuple(deltas),
        disclaimer=(
            "Directional lexical baseline; confirmatory semantic claims require "
            "a frozen autoregressive model and corpus."
        ),
    )
