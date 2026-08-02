import unittest

from atct_fingerprint.directionality import conditional_directionality


class DirectionalityTests(unittest.TestCase):
    def test_reversal_flips_conditional_asymmetry(self):
        sentences = [
            "alpha begins.",
            "alpha opens beta.",
            "beta establishes gamma.",
            "gamma supports delta.",
            "delta yields epsilon.",
            "epsilon closes omega.",
        ]
        forward = conditional_directionality(sentences, context_window=2)
        reverse = conditional_directionality(sentences[::-1], context_window=2)
        self.assertAlmostEqual(forward.asymmetry, -reverse.asymmetry)
        self.assertEqual(forward.method, "conditional_character_ngram")
        self.assertTrue(any(value is not None for value in forward.sentence_deltas))


if __name__ == "__main__":
    unittest.main()
