"""Leakage-aware evaluation of the ATCT incremental hypothesis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .encoders import SentenceEncoder, SentenceTransformerEncoder, TfidfSentenceEncoder
from .features import analyze_text, split_sentences

FEATURE_NAMES = (
    "order_sensitivity",
    "reverse_sensitivity",
    "long_history_dependence",
    "local_global_consistency",
    "local_global_variation",
    "turning_point_magnitude",
    "semantic_redundancy",
)


@dataclass(frozen=True)
class DatasetRow:
    text: str
    label: int
    split: str
    source: str
    genre: str


def load_dataset(path: str | Path) -> list[DatasetRow]:
    """Load the explicit train/test protocol CSV."""

    rows: list[DatasetRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label", "split", "source", "genre"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for line_number, record in enumerate(reader, start=2):
            split = record["split"].strip().lower()
            if split not in {"train", "test"}:
                raise ValueError(f"line {line_number}: split must be train or test")
            try:
                label = int(record["label"])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: label must be 0 or 1") from exc
            if label not in {0, 1}:
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            rows.append(
                DatasetRow(
                    text=record["text"].strip(),
                    label=label,
                    split=split,
                    source=record["source"].strip(),
                    genre=record["genre"].strip(),
                )
            )
    validate_protocol(rows)
    return rows


def validate_protocol(rows: Sequence[DatasetRow]) -> dict[str, object]:
    """Reject class-invalid splits and unseen-AI source leakage."""

    if not rows:
        raise ValueError("dataset is empty")
    train = [row for row in rows if row.split == "train"]
    test = [row for row in rows if row.split == "test"]
    if not train or not test:
        raise ValueError("both train and test rows are required")
    if {row.label for row in train} != {0, 1}:
        raise ValueError("train split must contain both labels")
    if {row.label for row in test} != {0, 1}:
        raise ValueError("test split must contain both labels")

    train_ai_sources = {row.source for row in train if row.label == 1}
    test_ai_sources = {row.source for row in test if row.label == 1}
    leakage = train_ai_sources.intersection(test_ai_sources)
    if leakage:
        raise ValueError(
            "AI source leakage between train and test: "
            + ", ".join(sorted(leakage))
        )

    train_genres = {row.genre for row in train}
    test_genres = {row.genre for row in test}
    unmatched_test_genres = sorted(test_genres.difference(train_genres))
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_ai_sources": sorted(train_ai_sources),
        "test_ai_sources": sorted(test_ai_sources),
        "unmatched_test_genres": unmatched_test_genres,
    }


def _feature_row(text: str, encoder: SentenceEncoder, seed: int) -> list[float]:
    result = analyze_text(text, encoder=encoder, seed=seed)
    return [float(getattr(result, name)) for name in FEATURE_NAMES]


def _shuffle_text(text: str, rng: np.random.Generator) -> str:
    sentences = split_sentences(text)
    permutation = rng.permutation(len(sentences))
    return "\n".join(sentences[index] for index in permutation)


def _atct_matrix(
    texts: Iterable[str],
    encoder: SentenceEncoder,
    *,
    seed: int,
) -> np.ndarray:
    return np.asarray(
        [_feature_row(text, encoder, seed + index) for index, text in enumerate(texts)],
        dtype=float,
    )


def _fit_auc(
    train_features,
    train_labels: np.ndarray,
    test_features,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> float:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train_features, train_labels)
    probability = model.predict_proba(test_features)[:, 1]
    return float(roc_auc_score(test_labels, probability))


def evaluate_rows(
    rows: Sequence[DatasetRow],
    *,
    encoder: SentenceEncoder | None = None,
    seed: int = 42,
    improvement_threshold: float = 0.05,
    shuffle_epsilon: float = 0.01,
) -> dict[str, object]:
    """Compare lexical baseline, ATCT addition, and shuffled control."""

    audit = validate_protocol(rows)
    selected_encoder = encoder or TfidfSentenceEncoder()
    train = [row for row in rows if row.split == "train"]
    test = [row for row in rows if row.split == "test"]
    train_texts = [row.text for row in train]
    test_texts = [row.text for row in test]
    y_train = np.asarray([row.label for row in train])
    y_test = np.asarray([row.label for row in test])

    lexical = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=20_000,
        sublinear_tf=True,
    )
    lexical_train = lexical.fit_transform(train_texts)
    lexical_test = lexical.transform(test_texts)
    baseline_auc = _fit_auc(
        lexical_train, y_train, lexical_test, y_test, seed=seed
    )

    atct_train = _atct_matrix(train_texts, selected_encoder, seed=seed)
    atct_test = _atct_matrix(test_texts, selected_encoder, seed=seed + 10_000)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(atct_train)
    scaled_test = scaler.transform(atct_test)
    combined_auc = _fit_auc(
        hstack([lexical_train, csr_matrix(scaled_train)]),
        y_train,
        hstack([lexical_test, csr_matrix(scaled_test)]),
        y_test,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    shuffled_train = [_shuffle_text(text, rng) for text in train_texts]
    shuffled_test = [_shuffle_text(text, rng) for text in test_texts]
    shuffled_atct_train = _atct_matrix(
        shuffled_train, selected_encoder, seed=seed + 20_000
    )
    shuffled_atct_test = _atct_matrix(
        shuffled_test, selected_encoder, seed=seed + 30_000
    )
    shuffled_scaler = StandardScaler()
    shuffled_scaled_train = shuffled_scaler.fit_transform(shuffled_atct_train)
    shuffled_scaled_test = shuffled_scaler.transform(shuffled_atct_test)
    shuffled_auc = _fit_auc(
        hstack([lexical_train, csr_matrix(shuffled_scaled_train)]),
        y_train,
        hstack([lexical_test, csr_matrix(shuffled_scaled_test)]),
        y_test,
        seed=seed,
    )

    improvement = combined_auc - baseline_auc
    mechanism_drop = combined_auc - shuffled_auc
    if improvement < improvement_threshold:
        gate = "unsupported"
    elif mechanism_drop <= shuffle_epsilon:
        gate = "mechanism_falsified"
    else:
        gate = "supported"

    lengths_train = [len(re.sub(r"\s+", "", text)) for text in train_texts]
    lengths_test = [len(re.sub(r"\s+", "", text)) for text in test_texts]
    return {
        "primary_metric": "AUROC",
        "baseline_auroc": baseline_auc,
        "combined_auroc": combined_auc,
        "shuffled_control_auroc": shuffled_auc,
        "improvement": improvement,
        "mechanism_drop": mechanism_drop,
        "improvement_threshold": improvement_threshold,
        "shuffle_epsilon": shuffle_epsilon,
        "gate": gate,
        "protocol_audit": audit,
        "length_audit": {
            "train_median_characters": float(np.median(lengths_train)),
            "test_median_characters": float(np.median(lengths_test)),
        },
        "feature_names": list(FEATURE_NAMES),
        "disclaimer": (
            "This result tests incremental structural signal; it does not prove "
            "AI authorship for any individual text."
        ),
    }


def build_encoder(name: str) -> SentenceEncoder:
    if name == "tfidf":
        return TfidfSentenceEncoder()
    if name == "e5":
        return SentenceTransformerEncoder()
    raise ValueError(f"unknown encoder: {name}")
