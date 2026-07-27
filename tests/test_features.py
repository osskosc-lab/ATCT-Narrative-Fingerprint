import unittest

import numpy as np

from atct_fingerprint.features import (
    _align_trajectory_to_sentence_ids,
    _cosine_distance_rows,
    compute_fingerprint,
    split_sentences,
)


class FeatureTests(unittest.TestCase):
    def test_japanese_sentence_split(self):
        text = "最初の文です。次へ進む！\n最後の文です？"
        self.assertEqual(
            split_sentences(text),
            ["最初の文です。", "次へ進む！", "最後の文です？"],
        )

    def test_closing_quote_stays_with_sentence(self):
        text = "彼女は叫んだ。\n「欲しい！」\n次へ進む。"
        self.assertEqual(
            split_sentences(text),
            ["彼女は叫んだ。", "「欲しい！」", "次へ進む。"],
        )

    def test_english_sentence_split_preserves_decimal(self):
        text = "Version 3.14 is stable. Next sentence! Final sentence?"
        self.assertEqual(
            split_sentences(text),
            ["Version 3.14 is stable.", "Next sentence!", "Final sentence?"],
        )

    def test_ascii_question_mark_can_split_without_space(self):
        text = "本当に?次へ進む!"
        self.assertEqual(split_sentences(text), ["本当に?", "次へ進む!"])

    def test_constant_trajectory_has_no_order_signal(self):
        sentences = [f"sentence {index}" for index in range(6)]
        vectors = np.ones((6, 4))
        result = compute_fingerprint(sentences, vectors, shuffle_count=8)
        self.assertAlmostEqual(result.order_sensitivity, 0.0)
        self.assertAlmostEqual(result.reverse_sensitivity, 0.0)

    def test_zero_vectors_compare_as_identical(self):
        zeros = np.zeros((3, 4))
        distances = _cosine_distance_rows(zeros, zeros)
        np.testing.assert_allclose(distances, np.zeros(3))

    def test_perturbed_states_are_aligned_by_sentence_identity(self):
        trajectory = np.asarray(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
            ]
        )
        permutation = np.asarray([2, 0, 3, 1])
        aligned = _align_trajectory_to_sentence_ids(trajectory, permutation)
        np.testing.assert_allclose(
            aligned,
            np.asarray(
                [
                    [20.0, 0.0],
                    [40.0, 0.0],
                    [10.0, 0.0],
                    [30.0, 0.0],
                ]
            ),
        )

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
