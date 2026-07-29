import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from atct_fingerprint.features import compute_fingerprint
from atct_fingerprint.reporting import write_report_bundle


class ReportingTests(unittest.TestCase):
    def test_complete_bundle_is_written(self):
        angles = np.linspace(0.0, 1.2, 10)
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        result = compute_fingerprint(
            [f"文{index}です。" for index in range(10)],
            vectors,
            shuffle_count=12,
        )
        with tempfile.TemporaryDirectory(dir=".") as directory:
            paths = write_report_bundle(result, directory)
            self.assertEqual(len(paths), 18)
            for value in paths.values():
                self.assertGreater(Path(value).stat().st_size, 0)
            payload = json.loads(Path(paths["fingerprint"]).read_text("utf-8"))
            self.assertEqual(payload["version"], "0.5.0")
            self.assertIn("motif_role_analysis", payload)
            self.assertIn("qa_closure", payload)
            self.assertIn("relational_analysis", payload)
            self.assertIn("closure_analysis", payload)
            self.assertIn("relations", paths)
            self.assertIn("discourse_units", paths)
            self.assertTrue(Path(paths["pdf"]).read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
