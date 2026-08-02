import unittest

import numpy as np

from atct_fingerprint.evaluation import (
    DatasetRow,
    _shuffle_text,
    validate_protocol,
)
from atct_fingerprint.features import split_sentences


def row(label, split, source, genre="essay", text=None):
    return DatasetRow(
        text=text or f"{source}一文目です。二文目です。三文目です。四文目です。",
        label=label,
        split=split,
        source=source,
        genre=genre,
    )


class ProtocolTests(unittest.TestCase):
    def test_unseen_ai_source_protocol_passes(self):
        rows = [
            row(0, "train", "human-a"),
            row(1, "train", "known-model"),
            row(0, "test", "human-b"),
            row(1, "test", "unseen-model"),
        ]
        audit = validate_protocol(rows)
        self.assertEqual(audit["test_ai_sources"], ["unseen-model"])
        self.assertEqual(audit["normalized_duplicate_documents"], 0)

    def test_ai_source_leakage_is_rejected(self):
        rows = [
            row(0, "train", "human-a"),
            row(1, "train", "same-model", text="訓練AI。二文。三文。四文。"),
            row(0, "test", "human-b"),
            row(1, "test", "same-model", text="試験AI。二文。三文。四文。"),
        ]
        with self.assertRaisesRegex(ValueError, "source leakage"):
            validate_protocol(rows)

    def test_document_leakage_is_rejected_after_whitespace_normalization(self):
        duplicate_train = "同じ文章です。 二文目です。三文目です。四文目です。"
        duplicate_test = "同じ文章です。\n二文目です。三文目です。四文目です。"
        rows = [
            row(0, "train", "human-a", text=duplicate_train),
            row(1, "train", "known-model"),
            row(0, "test", "human-b", text=duplicate_test),
            row(1, "test", "unseen-model"),
        ]
        with self.assertRaisesRegex(ValueError, "document leakage"):
            validate_protocol(rows)

    def test_both_labels_are_required_in_test(self):
        rows = [
            row(0, "train", "human-a"),
            row(1, "train", "known-model"),
            row(0, "test", "human-b"),
        ]
        with self.assertRaisesRegex(ValueError, "both labels"):
            validate_protocol(rows)

    def test_short_document_is_rejected_during_protocol_validation(self):
        rows = [
            row(0, "train", "human-a", text="一文。二文。三文。"),
            row(1, "train", "known-model"),
            row(0, "test", "human-b"),
            row(1, "test", "unseen-model"),
        ]
        with self.assertRaisesRegex(ValueError, "at least four"):
            validate_protocol(rows)

    def test_shuffle_control_never_returns_identity(self):
        text = "一文目です。二文目です。三文目です。四文目です。"
        shuffled = _shuffle_text(text, np.random.default_rng(0))
        self.assertNotEqual(split_sentences(shuffled), split_sentences(text))


if __name__ == "__main__":
    unittest.main()
