import unittest

import numpy as np

from atct_fingerprint.history import rolling_predictive_gain


class HistoryPredictionTests(unittest.TestCase):
    def test_rolling_gain_does_not_use_later_targets(self):
        angles = np.linspace(0.0, 1.3, 12)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        original = rolling_predictive_gain(
            vectors,
            short_window=2,
            long_window=5,
            min_train=4,
        )
        altered = vectors.copy()
        altered[10:] = np.asarray([[0.0, 1.0], [0.0, -1.0]])
        changed = rolling_predictive_gain(
            altered,
            short_window=2,
            long_window=5,
            min_train=4,
        )
        self.assertEqual(original.method, "rolling_origin_ridge")
        self.assertEqual(original.test_samples, 7)
        self.assertAlmostEqual(
            original.sentence_gains[6],
            changed.sentence_gains[6],
        )


if __name__ == "__main__":
    unittest.main()
