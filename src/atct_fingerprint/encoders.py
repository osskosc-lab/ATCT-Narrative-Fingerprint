"""Sentence encoders used by the fingerprint extractor.

The lightweight TF-IDF encoder is deterministic and intended for tests and
offline smoke runs. The semantic encoder is an optional dependency and is the
recommended backend for research runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SentenceEncoder(Protocol):
    """Minimal encoder contract."""

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        """Return one finite vector per sentence."""


@dataclass
class TfidfSentenceEncoder:
    """Encode sentences with document-local character TF-IDF.

    This is a lexical baseline, not a semantic embedding model. Fitting within
    one document is label-free and makes the extractor usable without network
    access or model downloads.
    """

    ngram_range: tuple[int, int] = (2, 5)

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            raise ValueError("at least one sentence is required")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            lowercase=True,
            norm="l2",
        )
        return vectorizer.fit_transform(sentences).toarray().astype(float)


@dataclass
class SentenceTransformerEncoder:
    """Encode sentences with a multilingual sentence-transformer.

    The import and model download are lazy so core installation remains small.
    E5 passage prefixes are applied by default.
    """

    model_name: str = "intfloat/multilingual-e5-small"
    prefix: str = "passage: "
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "semantic encoding requires: pip install "
                "'atct-narrative-fingerprint[semantic]'"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            raise ValueError("at least one sentence is required")
        inputs = [f"{self.prefix}{sentence}" for sentence in sentences]
        vectors = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)
