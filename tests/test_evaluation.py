import unittest

from atct_fingerprint.evaluation import DatasetRow, validate_protocol


def row(label, split, source, genre="essay"):
    return DatasetRow(
        text="一文目です。二文目です。三文目です。四文目です。",
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

    def test_ai_source_leakage_is_rejected(self):
        rows = [
            row(0, "train", "human-a"),
            row(1, "train", "same-model"),
            row(0, "test", "human-b"),
            row(1, "test", "same-model"),
        ]
        with self.assertRaisesRegex(ValueError, "source leakage"):
            validate_protocol(rows)

    def test_both_labels_are_required_in_test(self):
        rows = [
            row(0, "train", "human-a"),
            row(1, "train", "known-model"),
            row(0, "test", "human-b"),
        ]
        with self.assertRaisesRegex(ValueError, "both labels"):
            validate_protocol(rows)


if __name__ == "__main__":
    unittest.main()
