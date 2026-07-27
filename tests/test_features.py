import unittest

import numpy as np

from atct_fingerprint.features import compute_fingerprint, split_sentences


class FeatureTests(unittest.TestCase):
    def test_japanese_sentence_split(self):
        text = "最初の文です。次へ進む！\n最後の文です？"
        self.assertEqual(
            split_sentences(text),
            ["最初の文です。", "次へ進む！", "最後の文です？"],
        )

    def test_constant_trajectory_has_no_order_signal(self):
        sentences = [f"sentence {index}" for index in range(6)]
        vectors = np.ones((6, 4))
        result = compute_fingerprint(sentences, vectors, shuffle_count=8)
        self.assertAlmostEqual(result.order_sensitivity, 0.0)
        self.assertAlmostEqual(result.reverse_sensitivity, 0.0)

    def test_fingerprint_is_seed_deterministic(self):
        sentences = [f"sentence {index}" for index in range(8)]
        vectors = np.eye(8)
        first = compute_fingerprint(sentences, vectors, shuffle_count=12, seed=7)
        second = compute_fingerprint(sentences, vectors, shuffle_count=12, seed=7)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreater(first.reverse_sensitivity, 0.0)

    def test_nonadjacent_duplicate_counts_as_redundancy(self):
        sentences = [f"sentence {index}" for index in range(5)]
        vectors = np.eye(5)
        vectors[3] = vectors[0]
        result = compute_fingerprint(
            sentences,
            vectors,
            shuffle_count=4,
            redundancy_threshold=0.99,
        )
        self.assertGreater(result.semantic_redundancy, 0.0)

    def test_too_short_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least four"):
            compute_fingerprint(["a", "b", "c"], np.eye(3))


if __name__ == "__main__":
    unittest.main()
