import unittest

import numpy as np

from atct_fingerprint.motifs import analyze_motifs


class MotifTests(unittest.TestCase):
    def test_closing_motif_drops_when_ending_is_removed(self):
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.75, 0.25, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        sentences = [f"sentence {index}" for index in range(len(vectors))]
        closed = analyze_motifs(vectors, sentences, edge_width=2)
        removed_vectors = vectors[:-2]
        removed = analyze_motifs(
            removed_vectors,
            sentences[:-2],
            edge_width=1,
        )
        self.assertGreater(closed.transformed_closure, removed.transformed_closure)

    def test_exact_copy_is_duplication_not_transformed_return(self):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        sentences = [f"sentence {index}" for index in range(len(vectors))]
        result = analyze_motifs(vectors, sentences, edge_width=2)
        self.assertEqual(result.transformed_closure, 0.0)
        self.assertGreater(result.exact_duplicate_rate, 0.0)
        self.assertTrue(all(pair.kind != "exact_duplicate" for pair in result.motif_pairs))


if __name__ == "__main__":
    unittest.main()
