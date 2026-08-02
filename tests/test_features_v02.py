import unittest

import numpy as np

from atct_fingerprint.features import analyze_text, compute_fingerprint, split_sentences
from atct_fingerprint.history import (
    causal_history_states,
    history_consistency,
    legacy_inclusive_states,
)


def chain_vectors(count=14):
    angles = np.linspace(0.0, 1.5, count)
    return np.column_stack(
        [np.cos(angles), np.sin(angles), np.linspace(0.0, 0.2, count)]
    )


class FeatureV02Tests(unittest.TestCase):
    def test_japanese_and_english_sentence_split(self):
        text = "値は3.14です。「進む！」 Version 3.14 is stable. Next?"
        self.assertEqual(
            split_sentences(text),
            ["値は3.14です。", "「進む！」", "Version 3.14 is stable.", "Next?"],
        )

    def test_inline_quoted_question_does_not_split_the_host_sentence(self):
        text = "このノートは、「価値とは何か。」という問いから始まった。"
        self.assertEqual(split_sentences(text), [text])

    def test_adjacent_complete_quotes_remain_separate_sentences(self):
        self.assertEqual(
            split_sentences("「行く。」「待って。」"),
            ["「行く。」", "「待って。」"],
        )

    def test_numbered_heading_is_not_split_after_its_number(self):
        text = "1. 魂の奥底を揺さぶる合図"
        self.assertEqual(split_sentences(text), [text])

    def test_current_sentence_is_excluded_from_history(self):
        vectors = np.eye(5)
        causal = causal_history_states(vectors, 3)
        legacy = legacy_inclusive_states(vectors, 3)
        self.assertAlmostEqual(float(np.dot(causal[1], vectors[1])), 0.0)
        self.assertGreater(float(np.dot(legacy[1], vectors[1])), 0.0)

    def test_random_orthogonal_text_has_zero_order_evidence(self):
        vectors = np.eye(10)
        sentences = [f"sentence {index}" for index in range(10)]
        result = compute_fingerprint(
            sentences,
            vectors,
            controls=("random",),
            shuffle_count=32,
            seed=3,
        )
        self.assertAlmostEqual(result.order_z, 0.0)

    def test_coherent_chain_has_positive_order_evidence(self):
        vectors = chain_vectors()
        sentences = [f"step {index}" for index in range(len(vectors))]
        result = compute_fingerprint(
            sentences,
            vectors,
            controls=("local", "block", "random", "reverse"),
            shuffle_count=96,
            seed=11,
        )
        self.assertGreaterEqual(result.order_z, 2.0)
        self.assertGreater(
            result.controls["random"].effect,
            result.controls["local"].effect,
        )
        self.assertGreater(result.reverse_directionality, 0.0)

    def test_fully_shuffled_chain_loses_order_evidence(self):
        vectors = chain_vectors()
        sentences = [f"step {index}" for index in range(len(vectors))]
        original = compute_fingerprint(
            sentences,
            vectors,
            controls=("random",),
            shuffle_count=96,
            seed=11,
        )
        order = np.random.default_rng(91).permutation(len(vectors))
        shuffled = compute_fingerprint(
            [sentences[index] for index in order],
            vectors[order],
            controls=("random",),
            shuffle_count=96,
            seed=11,
        )
        self.assertGreater(original.order_z, shuffled.order_z)

    def test_paragraph_shuffle_changes_global_not_within_paragraph_order(self):
        vectors = chain_vectors(15)
        sentences = [f"step {index}" for index in range(len(vectors))]
        paragraphs = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))]
        result = compute_fingerprint(
            sentences,
            vectors,
            controls=("local", "paragraph_order", "random"),
            paragraphs=paragraphs,
            shuffle_count=64,
            seed=8,
        )
        self.assertIn("paragraph_order", result.controls)
        self.assertGreater(result.controls["paragraph_order"].effect, 0.0)

    def test_seed_is_exactly_reproducible(self):
        vectors = chain_vectors()
        sentences = [f"step {index}" for index in range(len(vectors))]
        first = compute_fingerprint(
            sentences, vectors, shuffle_count=24, seed=17
        )
        second = compute_fingerprint(
            sentences, vectors, shuffle_count=24, seed=17
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_thirty_seeds_preserve_primary_z_sign(self):
        vectors = chain_vectors()
        sentences = [f"step {index}" for index in range(len(vectors))]
        values = [
            compute_fingerprint(
                sentences,
                vectors,
                controls=("random",),
                shuffle_count=24,
                seed=seed,
            ).order_z
            for seed in range(30)
        ]
        self.assertTrue(all(value > 0 for value in values))

    def test_surface_length_does_not_change_semantic_order_z(self):
        vectors = chain_vectors()
        short = [f"s{index}" for index in range(len(vectors))]
        long = [
            f"This is a deliberately much longer surface sentence {index}."
            for index in range(len(vectors))
        ]
        short_result = compute_fingerprint(
            short,
            vectors,
            controls=("random",),
            shuffle_count=48,
            seed=5,
        )
        long_result = compute_fingerprint(
            long,
            vectors,
            controls=("random",),
            shuffle_count=48,
            seed=5,
        )
        self.assertAlmostEqual(short_result.order_z, long_result.order_z)
        self.assertNotEqual(
            short_result.confound_audit["short_sentence_rate"],
            long_result.confound_audit["short_sentence_rate"],
        )

    def test_new_history_gap_exceeds_inclusive_baseline_gap(self):
        vectors = chain_vectors()
        original_new = float(
            np.mean(
                history_consistency(
                    vectors, causal_history_states(vectors, window=5)
                )
            )
        )
        original_old = float(
            np.mean(
                history_consistency(
                    vectors, legacy_inclusive_states(vectors, window=5)
                )
            )
        )
        rng = np.random.default_rng(7)
        new_null = []
        old_null = []
        for _ in range(80):
            order = rng.permutation(len(vectors))
            shuffled = vectors[order]
            new_null.append(
                np.mean(
                    history_consistency(
                        shuffled, causal_history_states(shuffled, window=5)
                    )
                )
            )
            old_null.append(
                np.mean(
                    history_consistency(
                        shuffled, legacy_inclusive_states(shuffled, window=5)
                    )
                )
            )
        self.assertGreater(
            original_new - float(np.mean(new_null)),
            original_old - float(np.mean(old_null)),
        )

    def test_no_0_to_100_aggregate_score_is_exposed(self):
        result = compute_fingerprint(
            [f"step {index}" for index in range(10)],
            chain_vectors(10),
            controls=("random",),
            shuffle_count=16,
        )
        self.assertNotIn("display_scores", result.to_dict())

    def test_markdown_analysis_emits_hierarchical_controls(self):
        text = """# 導入
問題を定義します。履歴を確認します。

## 反証
順序を交換します。結果の差を測ります。

## 結論
原文へ戻ります。次の実験を問いますか？
"""
        result = analyze_text(text, shuffle_count=8, seed=4)
        self.assertEqual(result.sentence_count, 6)
        self.assertIn("paragraph_inner", result.controls)
        self.assertIn("paragraph_order", result.controls)
        self.assertIn("section_order", result.controls)
        self.assertEqual(result.document_structure["layer_counts"]["heading"], 3)
        self.assertEqual(len(result.section_graph.nodes), 3)


if __name__ == "__main__":
    unittest.main()
