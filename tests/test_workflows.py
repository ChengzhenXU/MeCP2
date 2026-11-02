from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.core import ginter
from analysis.mutation import main as mutation_main
from simulation.common.workflow import ROOT, build_protein


class DistributionTests(unittest.TestCase):
    def test_periodic_histograms_recover_known_interchain_pairs(self):
        xyz = np.array([[0.1, 0, 0], [2.0, 0, 0], [9.9, 0, 0], [2.5, 0, 0]])
        patch_ids = np.array([0, 1])
        lookup = np.array([[0, 1], [1, 2]])
        total, regional = ginter.ginter_frame_histograms(
            xyz, np.full(3, 10.0), np.arange(0, 1.1, 0.1), 2, 2, patch_ids, lookup)
        self.assertEqual(total.sum(), 2)
        self.assertEqual(regional[0].sum(), 1)
        self.assertEqual(regional[2].sum(), 1)
        np.testing.assert_array_equal(regional.sum(axis=0), total)

    def test_patch_opportunities_cover_all_residue_pairs(self):
        possible = sum(ginter.possible_patch_pairs_per_chain_pair(pair)
                       for pair in ginter.PATCH_PAIRS)
        self.assertEqual(possible, 486**2)


class ModelInputTests(unittest.TestCase):
    def test_mutant_sequences_are_single_substitutions(self):
        def read(construct):
            return "".join(line for line in (ROOT / "inputs/protein" / construct /
                           "MeCP2.fasta").read_text().splitlines() if not line.startswith(">"))
        wt = read("WT")
        self.assertEqual(len(wt), 486)
        for mutant in ("R133C", "T158M", "P225R", "R306C"):
            sequence = read(mutant)
            differences = [i + 1 for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
            self.assertEqual(len(sequence), 486)
            self.assertEqual(differences, [int(mutant[1:-1])])
            self.assertEqual(sequence[differences[0] - 1], mutant[-1])

    def test_wt_system_has_reference_restraints_and_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_protein("WT", 0.1, 101, Path(directory), [50, 50, 150], copies=1)
            self.assertEqual(model.system.getNumParticles(), 486)
            self.assertEqual(model.components[0].cs.getNumBonds(), 288)
            self.assertEqual(model.ah.getNumExclusions(), 485 + 288)
            self.assertEqual(model.yu.getNumExclusions(), model.ah.getNumExclusions())


class AnalysisPipelineTests(unittest.TestCase):
    def test_selected_frames_and_contact_closure(self):
        import MDAnalysis as mda
        from analysis.run import main as analyze

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = build_protein("WT", 0.1, 101, root, [50, 50, 150], copies=2)
            universe = mda.Universe(str(root / "top.pdb"))
            xyz = np.tile(model.components[0].xinit, (2, 1))
            xyz[:486] += [10, 10, 10]
            xyz[486:] += [10.7, 10, 10]
            with mda.Writer(str(root / "trajectory.dcd"), n_atoms=len(universe.atoms)) as writer:
                for frame in range(6):
                    universe.atoms.positions = (xyz + frame * 0.01) * 10
                    universe.dimensions = [500, 500, 500, 90, 90, 90]
                    writer.write(universe.atoms)
            analyze(["--topology", str(root / "top.pdb"), "--trajectory", str(root / "trajectory.dcd"),
                     "--chains", "2", "--start-frame", "1", "--stop-frame", "6", "--stride", "2",
                     "--output", str(root / "analysis")])
            metadata = json.loads((root / "analysis/analysis.json").read_text())
            self.assertEqual(metadata["selected_frames"], 3)
            with np.load(root / "analysis/clusters.npz") as z:
                self.assertEqual(z["largest_cluster_by_frame"].shape, (3, 3))
            with np.load(root / "analysis/contacts.npz") as z:
                np.testing.assert_allclose(z["qij"], z["qij"].T, atol=1e-15)
                self.assertAlmostEqual(float(z["wab"].sum()), float(z["qij"].sum()))
            self.assertEqual(len(list((root / "analysis").glob("*.png"))), 0)


class MutationSelectionTests(unittest.TestCase):
    def test_signed_losses_and_cross_mutant_recurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profiles.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["background", "construct", "salt_molar",
                                                            "replicate", "residue", "contact"])
                writer.writeheader()
                for construct in ("WT", "R133C", "T158M", "P225R", "R306C"):
                    for replicate in (1, 2):
                        for residue in range(91, 163):
                            value = 10.0 + replicate * 0.1
                            if construct != "WT":
                                value += -1.0 - residue / 1000 if residue < 97 else 5.0
                            writer.writerow(dict(background="protein", construct=construct,
                                                 salt_molar=0.1, replicate=replicate,
                                                 residue=residue, contact=value))
            mutation_main(["--profiles", str(source), "--output", str(root / "result")])
            with (root / "result/top6.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertEqual({int(row["residue"]) for row in rows}, set(range(91, 97)))
            self.assertTrue(all(float(row["delta"]) < 0 for row in rows))
            self.assertTrue(all(int(row["occurrence_count"]) == 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
