from __future__ import annotations

import unittest

import numpy as np

from analysis.core.metrics import (
    _class_contact_summary,
    connected_component_sizes,
    switching_weight,
)


class MetricKernelTests(unittest.TestCase):
    def test_connected_components_include_singletons(self) -> None:
        sizes = connected_component_sizes(
            6, np.asarray([[0, 1], [1, 2], [4, 5]], dtype=np.int32)
        )
        self.assertEqual(sorted(sizes.tolist()), [1, 2, 3])

    def test_switching_midpoint(self) -> None:
        distance = np.asarray([0.75])
        sigma = np.asarray([0.5])
        result = switching_weight(distance, sigma, beta=50.0, gamma=1.5)
        self.assertAlmostEqual(float(result[0]), 0.5)

    def test_any_dna_events_are_unique_unions(self) -> None:
        chains, events, hits = _class_contact_summary(
            np.asarray([0, 0, 1, 486], dtype=np.int64),
            np.asarray([0, 1, 1, 2], dtype=np.int64),
            np.asarray([0, 0, 1], dtype=np.int16),
            chain_count=2,
            residue_count=486,
        )
        self.assertEqual(chains, 2)
        self.assertEqual(events, 2)
        self.assertEqual(int(hits.sum()), 3)


if __name__ == "__main__":
    unittest.main()
