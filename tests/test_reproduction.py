"""Recovery and statistical definitions using temporary analytical fixtures."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.summarize import main as summarize


class StatisticsTests(unittest.TestCase):
    def test_one_direction_scalar_and_nested_block_sd_remain_distinct(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            block_values = []
            for replicate in (1, 2):
                directory = root / str(replicate)
                directory.mkdir()
                values = np.arange(1, 7) + (replicate - 1) * 10
                block_values.extend(values)
                blocks = np.zeros((6, 486, 486))
                blocks[:, 0, 90] = blocks[:, 90, 0] = values
                np.savez_compressed(directory / "contacts.npz", qij=blocks.mean(0), qij_blocks=blocks)
                np.savez_compressed(directory / "clusters.npz", cutoffs_nm=[.5, .55, .6],
                                    largest_cluster_by_frame=np.ones((3000, 3)),
                                    cluster_event_counts=np.ones((3, 151)),
                                    cluster_event_probability=np.full((3, 151), 1 / 150))
                (directory / "ginter.csv").write_text("pair,r_nm,g_inter\nALL,0.025,1.0\n")
                (directory / "analysis.json").write_text(json.dumps(dict(
                    trajectory_sha256=str(replicate), construct="WT", dna_duplexes=0,
                    chains=150, selected_frames=3000, sampled_interval_ns=1,
                    frames_per_block=500, block_ns=500)))
                records.append(dict(directory=str(replicate), construct="WT", salt_molar=.1,
                                    background="protein", replicate=replicate))
            with (root / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            summarize(["--manifest", str(root / "manifest.csv"), "--output", str(root / "summary"),
                       "--paper-statistics"])
            trajectories = pd.read_csv(root / "summary/trajectory_metrics.csv")
            self.assertEqual(trajectories.idr_fd_contact.tolist(), [3.5, 13.5])
            self.assertEqual(trajectories.idr_fd_complete_pair_contact.tolist(), [7, 27])
            profiles = pd.read_csv(root / "summary/native_idr_fd_profiles.csv")
            self.assertEqual(profiles[profiles.residue.eq(91)].contact.tolist(), [7, 27])
            paper = pd.read_csv(root / "summary/paper_block_observables.csv")
            row = paper[paper.metric.eq("idr_fd_one_direction")].iloc[0]
            self.assertEqual(row.independent_n, 2)
            self.assertEqual(row.n_temporal_blocks, 12)
            self.assertAlmostEqual(row["std"], np.std(block_values, ddof=1))
            condition = pd.read_csv(root / "summary/condition_observables.csv")
            row = condition[condition.metric.eq("idr_fd_one_direction")].iloc[0]
            self.assertAlmostEqual(row["std"], np.std([3.5, 13.5], ddof=1))
            for replicate in (1, 2):
                metadata_path = root / str(replicate) / "analysis.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["simulation_seed"] = 123
                metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "same simulation seed"):
                summarize(["--manifest", str(root / "manifest.csv"), "--output", str(root / "duplicate_seed")])


class RecoveryTests(unittest.TestCase):
    def test_fresh_context_handoff_preserves_geometry_without_copying_clock(self):
        import openmm as mm
        from openmm import app, unit
        from simulation.common.workflow import install_host_state, make_simulation

        system = mm.System()
        system.addParticle(100)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(50, 0, 0), mm.Vec3(0, 50, 0), mm.Vec3(0, 0, 50))
        topology = app.Topology()
        residue = topology.addResidue("ALA", topology.addChain())
        topology.addAtom("CA", app.element.carbon, residue)
        source = make_simulation(system, topology, "Reference", 101)
        source.context.setPositions([[1, 2, 3]] * unit.nanometer)
        source.context.setVelocities([[.1, .2, .3]] * unit.nanometer / unit.picosecond)
        source.step(5)
        state = source.context.getState(getPositions=True, getVelocities=True)
        self.assertGreater(source.currentStep, 0)
        for seed in (101, 104):
            destination = make_simulation(system, topology, "Reference", seed)
            install_host_state(destination.context, state)
            observed = destination.context.getState(getPositions=True, getVelocities=True)
            self.assertEqual(destination.currentStep, 0)
            self.assertEqual(observed.getTime().value_in_unit(unit.picosecond), 0)
            np.testing.assert_array_equal(observed.getPositions(asNumpy=True), state.getPositions(asNumpy=True))
            np.testing.assert_array_equal(observed.getVelocities(asNumpy=True), state.getVelocities(asNumpy=True))
            np.testing.assert_array_equal(observed.getPeriodicBoxVectors(asNumpy=True), state.getPeriodicBoxVectors(asNumpy=True))
            self.assertEqual(destination.integrator.getRandomNumberSeed(), seed)
            del destination
            gc.collect()

    def test_checkpoint_reload_and_segment_time_continuity(self):
        import openmm as mm
        from openmm import app, unit
        from analysis.core.segments import SegmentReader
        from simulation.common.checkpoints import restore_snapshot
        from simulation.common.production import run_production

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = mm.System()
            system.addParticle(100)
            force = mm.CustomExternalForce("0*x")
            force.addParticle(0, [])
            system.addForce(force)
            topology = app.Topology()
            residue = topology.addResidue("ALA", topology.addChain())
            topology.addAtom("CA", app.element.carbon, residue)
            with (root / "topology.pdb").open("w") as stream:
                app.PDBFile.writeFile(topology, [[1, 1, 1]] * unit.nanometer, stream)
            (root / "settings.json").write_text("{}\n")
            integrator = mm.LangevinMiddleIntegrator(310, .01, .01)
            integrator.setRandomNumberSeed(123)
            simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("Reference"))
            simulation.context.setPositions([[1, 1, 1]])
            run_production(simulation, root, 6, 2)
            expected = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
            del simulation, integrator
            gc.collect()
            simulation, metadata = restore_snapshot(root)
            self.assertEqual(metadata["platform"], "Reference")
            self.assertEqual(simulation.context.getPlatform().getName(), metadata["platform"])
            self.assertEqual(simulation.currentStep, 6)
            np.testing.assert_array_equal(simulation.context.getState(getPositions=True).getPositions(asNumpy=True), expected)
            run_production(simulation, root, 10, 2, metadata["progress"])
            del simulation
            gc.collect()
            with SegmentReader(root / "trajectories.json") as reader:
                self.assertEqual(len(reader), 5)
                np.testing.assert_allclose([reader[i].time for i in range(5)], [.02, .04, .06, .08, .10])
            self.assertEqual(len(list((root / "checkpoints").glob("generation-*"))), 2)
            manifest = json.loads((root / "trajectories.json").read_text())
            manifest["segments"][1]["first_elapsed_ns"] += .001
            (root / "invalid_segments.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "gap or overlap"):
                SegmentReader(root / "invalid_segments.json")
            pointer = json.loads((root / "checkpoints/current.json").read_text())
            checkpoint = root / "checkpoints" / pointer["generation"] / "checkpoint.chk"
            with checkpoint.open("ab") as stream:
                stream.write(b"invalid")
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_snapshot(root, "Reference")


class PreparationGateTests(unittest.TestCase):
    def test_original_bond_and_minimization_limits(self):
        from simulation.common.preparation.health import validate_geometry

        positions = np.zeros((77412, 3))
        forces = np.zeros_like(positions)
        box = np.eye(3) * 50
        validate_geometry(positions, forces, box, expected_forces=13, actual_forces=13, final_box=True)
        positions[0, 0] = 1.1
        with self.assertRaisesRegex(RuntimeError, "protein_chain bond"):
            validate_geometry(positions, forces, box, expected_forces=13, actual_forces=13)
        positions[0, 0] = 0
        forces[:, 0] = 11
        with self.assertRaisesRegex(RuntimeError, "RMS force"):
            validate_geometry(positions, forces, box, expected_forces=13, actual_forces=13,
                              final_minimization=True)

    def test_saved_nvt_temperature_samples_control_production_acceptance(self):
        import argparse
        import openmm as mm
        from openmm import app
        from simulation.common.workflow import prepare

        system = mm.System()
        system.addParticle(100)
        force = mm.CustomExternalForce("0*x")
        force.addParticle(0, [])
        system.addForce(force)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(50, 0, 0), mm.Vec3(0, 50, 0), mm.Vec3(0, 0, 50))
        topology = app.Topology()
        residue = topology.addResidue("ALA", topology.addChain())
        topology.addAtom("CA", app.element.carbon, residue)
        integrator = mm.LangevinMiddleIntegrator(310, .01, .01)
        simulation = app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName("Reference"))
        simulation.context.setPositions([[1, 1, 1]])
        args = argparse.Namespace(seed=101, protein_protocol="legacy")
        for temperature, accepted in ((300, False), (310, True)):
            metadata = dict(progress=dict(stage="nvt_equilibration", completed=100,
                                          temperatures=[temperature] * 100))
            if accepted:
                result = prepare(system, topology, None, None, (), (), args, True,
                                 restored=(simulation, metadata))
                self.assertEqual(result.currentStep, 0)
            else:
                with self.assertRaisesRegex(RuntimeError, "305-315"):
                    prepare(system, topology, None, None, (), (), args, True,
                            restored=(simulation, metadata))


if __name__ == "__main__":
    unittest.main()
