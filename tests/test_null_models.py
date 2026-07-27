import unittest

import numpy as np

from atct_fingerprint.null_models import (
    block_swap,
    local_swap,
    paragraph_swap,
)


class NullModelTests(unittest.TestCase):
    def test_local_control_swaps_one_adjacent_pair(self):
        order = local_swap(10, np.random.default_rng(2))
        changed = np.flatnonzero(order != np.arange(10))
        self.assertEqual(len(changed), 2)
        self.assertEqual(changed[1] - changed[0], 1)

    def test_block_control_preserves_order_inside_blocks(self):
        order = block_swap(12, np.random.default_rng(3), block_min=3, block_max=3)
        positions = {value: index for index, value in enumerate(order)}
        for block_start in (0, 3, 6, 9):
            block = [positions[value] for value in range(block_start, block_start + 3)]
            self.assertEqual(block, sorted(block))

    def test_paragraph_control_preserves_within_paragraph_order(self):
        paragraphs = [[0, 1, 2], [3, 4], [5, 6, 7]]
        order = paragraph_swap(paragraphs, np.random.default_rng(4))
        positions = {value: index for index, value in enumerate(order)}
        for paragraph in paragraphs:
            observed = [positions[value] for value in paragraph]
            self.assertEqual(observed, sorted(observed))
        self.assertFalse(np.array_equal(order, np.arange(8)))


if __name__ == "__main__":
    unittest.main()
