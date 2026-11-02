from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.common.preparation.geometry import (  # noqa: E402
    PackingFailure,
    _has_periodic_clash,
    _place_one,
    _self_image_safe,
    build_kapoor_2b_duplex_template,
    load_canonical_residue_com_template,
    long_dna_periodic_qa,
    reverse_complement,
    uniform_rotation_matrix,
)


CANONICAL_PDB = ROOT / "inputs" / "protein" / "WT" / "MeCP2.pdb"


class CanonicalProteinGeometryTests(unittest.TestCase):
    def test_residue_com_matches_frozen_calvados_algorithm(self) -> None:
        from MDAnalysis import Universe

        observed = load_canonical_residue_com_template(CANONICAL_PDB)
        universe = Universe(str(CANONICAL_PDB))
        universe.atoms.translate(-universe.atoms.center_of_mass())
        expected = np.asarray(
            [residue.atoms.center_of_mass() for residue in universe.residues], dtype=float
        ) / 10.0
        self.assertEqual(observed.mapping, "canonical_all_residue_mass_COM")
        self.assertEqual(observed.positions_nm.shape, (486, 3))
        np.testing.assert_allclose(observed.positions_nm, expected, rtol=0.0, atol=1.0e-12)

        ca = universe.select_atoms("name CA").positions / 10.0
        rms = float(np.sqrt(np.mean(np.sum((observed.positions_nm - ca) ** 2, axis=1))))
        self.assertGreater(rms, 0.10)  # guards against a silent CA "repair"

    def test_independent_proper_rigid_rotations_preserve_geometry(self) -> None:
        template = load_canonical_residue_com_template(CANONICAL_PDB).positions_nm
        rng = np.random.default_rng(541342803)
        first = uniform_rotation_matrix(rng)
        second = uniform_rotation_matrix(rng)
        self.assertFalse(np.array_equal(first, second))
        for rotation in (first, second):
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertTrue(math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-12))
        pairs = np.array([[0, 1], [90, 161], [224, 305], [0, 485]])
        original = np.linalg.norm(template[pairs[:, 0]] - template[pairs[:, 1]], axis=1)
        transformed = template @ first.T + np.array([7.0, 8.0, 9.0])
        after = np.linalg.norm(
            transformed[pairs[:, 0]] - transformed[pairs[:, 1]], axis=1
        )
        np.testing.assert_allclose(after, original, rtol=0.0, atol=1e-12)


class KapoorDuplexGeometryTests(unittest.TestCase):
    def test_reverse_complement_and_antiparallel_particle_order(self) -> None:
        sequence = "ACGTTGCAACGT"
        template = build_kapoor_2b_duplex_template(sequence)
        self.assertEqual(template.strand_sequences, (sequence, reverse_complement(sequence)))
        self.assertEqual(template.positions_nm.shape, (2, len(sequence), 2, 3))
        coords = template.positions_nm
        axis_a = coords[0, -1, 0] - coords[0, 0, 0]
        axis_b = coords[1, -1, 0] - coords[1, 0, 0]
        self.assertLess(float(np.dot(axis_a, axis_b)), 0.0)

        bb_a = np.linalg.norm(np.diff(coords[0, :, 0], axis=0), axis=1)
        bb_b = np.linalg.norm(np.diff(coords[1, :, 0], axis=0), axis=1)
        bb_base = np.linalg.norm(coords[:, :, 0] - coords[:, :, 1], axis=2)
        np.testing.assert_allclose(bb_a, 0.550, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(bb_b, 0.550, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(bb_base, 0.575, rtol=0.0, atol=1e-12)

        # Native A(i) partner is B(N-1-i), never same storage direction.
        native = np.linalg.norm(coords[0, :, 1] - coords[1, ::-1, 1], axis=1)
        self.assertTrue(np.all(native > 0.45))
        self.assertTrue(np.all(native < 0.65))


class PeriodicPackingQATests(unittest.TestCase):
    def test_explicit_self_image_and_minimum_image_clash(self) -> None:
        diameters = np.array([0.55, 0.55])
        self.assertTrue(
            _self_image_safe(
                np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                diameters,
                np.array([50.0, 50.0, 500.0]),
            )
        )
        self.assertFalse(
            _self_image_safe(
                np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
                diameters,
                np.array([50.0, 50.0, 500.0]),
            )
        )
        box = np.array([50.0, 50.0, 500.0])
        self.assertTrue(
            _has_periodic_clash(
                np.array([[49.9, 2.0, 2.0]]),
                np.array([0.55]),
                np.array([[0.1, 2.0, 2.0]]),
                np.array([0.55]),
                box,
            )
        )
        self.assertFalse(
            _has_periodic_clash(
                np.array([[25.0, 2.0, 2.0]]),
                np.array([0.55]),
                np.array([[0.1, 2.0, 2.0]]),
                np.array([0.55]),
                box,
            )
        )

    def test_long_dna_records_unwrapped_self_image_and_inter_duplex_gate(self) -> None:
        box = np.array([50.0, 50.0, 500.0])
        raw = [
            np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
            np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]]),
            np.array([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]]),
        ]
        wrapped = [np.mod(value, box) for value in raw]
        report = long_dna_periodic_qa(wrapped, raw, box, packing_gate_nm=0.55)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["packing_gate_nm"], 0.55)
        self.assertEqual(report["total_violation_count"], 0)
        self.assertGreaterEqual(report["initial_self_image_min_distance_nm"], 0.55)
        self.assertGreaterEqual(report["initial_inter_duplex_min_distance_nm"], 0.55)
        self.assertEqual(len(report["self_image"]), 3)
        self.assertTrue(all("pbc_spans_nm" in item for item in report["self_image"]))

    def test_no_layout_is_a_hard_failure_without_fallback(self) -> None:
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        with self.assertRaisesRegex(PackingFailure, "hard STOP"):
            _place_one(
                molecule_id="impossible",
                molecule_kind="test",
                template_nm=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                diameters_nm=np.array([1.0, 1.0]),
                rotation_rng=rng_a,
                translation_rng=rng_b,
                existing_positions=np.empty((0, 3)),
                existing_diameters=np.empty((0,)),
                box_nm=np.array([1.0, 1.0, 1.0]),
                max_attempts=8,
            )


if __name__ == "__main__":
    unittest.main()

