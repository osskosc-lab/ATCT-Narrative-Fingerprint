import unittest

import numpy as np

from atct_fingerprint.diagnostics import editing_risks, licensed_jumps


class DiagnosticTests(unittest.TestCase):
    def test_explicit_boundary_marker_licenses_a_jump(self):
        sentences = [
            "最初の議論です。",
            "ここからは比喩であり、同一視しない。",
            "別領域の議論へ移ります。",
            "最後の文です。",
        ]
        rows = [
            {"index": 0, "history_consistency": None, "turn_z": None},
            {"index": 1, "history_consistency": 0.2, "turn_z": 0.5},
            {"index": 2, "history_consistency": 0.0, "turn_z": 3.0},
            {"index": 3, "history_consistency": 0.2, "turn_z": 0.5},
        ]
        licenses = licensed_jumps(sentences, rows)
        self.assertEqual(licenses[0]["transition_sentence"], 2)
        risks = editing_risks(
            sentences,
            theme_similarities=np.ones(4),
            sentence_rows=rows,
            duplicate_rate=0.0,
            confounds={
                "short_sentence_rate": 0.0,
                "technical_term_density": 0.0,
            },
            licensed_indices=[2],
        )
        self.assertFalse(any(item["risk"] == "unexplained_leap" for item in risks))


if __name__ == "__main__":
    unittest.main()
