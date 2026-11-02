"""Entry points for CALVADOS3 and CALVADOS3-Kapoor simulations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation" / "vendor"))
CONSTRUCTS = ("WT", "R133C", "T158M", "P225R", "R306C")


def parser(story: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a MeCP2 coarse-grained simulation.")
    p.add_argument("--salt", type=float, choices=(0.01, 0.1, 0.3), default=0.1,
                   help="Ionic strength in mol/L.")
    p.add_argument("--construct", choices=CONSTRUCTS, default="WT")
    p.add_argument("--seed", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path, help="Resume the latest checkpoint in an existing run directory.")
    p.add_argument("--protein-protocol", choices=("legacy", "adaptive"), default="legacy",
                   help="Use adaptive compression for R306C n2.")
    p.add_argument("--platform", help="Optional installed OpenMM platform name; otherwise select automatically.")
    p.add_argument("--steps", type=int,
                   help="Production steps, excluding preparation (0.01 ps per step).")
    p.add_argument("--report-interval", type=int,
                   default=100_000 if story == "protein_dna" else 50_000)
    p.add_argument("--log-interval", type=int,
                   default=10_000 if story == "protein_dna" else 1_000_000)
    p.add_argument("--build-only", action="store_true",
                   help="Construct the initial system without creating a simulation context.")
    return p


def build_protein(construct, salt, seed, directory, box, copies=150):
    import numpy as np
    from calvados.cfg import Config, Components
    from calvados.sim import Sim

    inputs = ROOT / "inputs" / "protein" / construct
    from Bio import SeqIO
    from MDAnalysis import Universe
    with (inputs / "MeCP2.fasta").open() as handle:
        expected = str(SeqIO.read(handle, "fasta").seq)
    observed = Universe(str(inputs / "MeCP2.pdb")).residues.sequence(format="string")
    if len(expected) != 486 or observed != expected:
        raise ValueError("Initial PDB and FASTA must describe the same 486-residue construct.")
    random.seed(seed)
    np.random.seed(seed)
    config = Config(box=list(box), temp=310.0, ionic=salt, pH=7.0,
                    sysname="protein", topol="grid", restart=None, verbose=False).config
    components = Components(molecule_type="protein", restraint=True,
                            charge_termini="both", use_com=True, colabfold=1,
                            k_harmonic=700.0, restraint_type="harmonic",
                            fresidues=str(inputs / "residues_CALVADOS3.csv"),
                            ffasta=str(inputs / "MeCP2.fasta"),
                            fdomains=str(inputs / "domains.yaml"), pdb_folder=str(inputs))
    components.add(name="MeCP2", nmol=copies)
    model = Sim(str(directory), config, components.components)
    model.build_system()
    return model


def build_initial(construct, salt, seed, directory, with_dna):
    import numpy as np
    from openmm import app, unit
    from .kapoor2b_openmm import append_kapoor2b_to_system
    from .preparation.geometry import (build_kapoor_2b_duplex_template,
                                      load_canonical_residue_com_template,
                                      normalize_dna_sequence, pack_project_geometry)
    from .topology import build_topology

    box = np.array([50.0, 50.0, 500.0 if with_dna else 150.0])
    model = build_protein(construct, salt, seed, directory, box)
    component = model.components[0]
    sequence = str(component.seq)
    if len(sequence) != 486:
        raise ValueError("MeCP2 must contain exactly 486 residues.")
    if not with_dna:
        rng = np.random.RandomState(seed)
        positions = np.concatenate([
            component.xinit + rng.rand(3) * box +
            (rng.rand(*component.xinit.shape) - 0.5) * 0.1 for _ in range(150)
        ])
        topology = model.top.to_openmm()
        groups = tuple(tuple(range(i * 486, (i + 1) * 486)) for i in range(150))
        edges = tuple((i * 486 + j, i * 486 + j + 1)
                      for i in range(150) for j in range(485))
    else:
        dna_sequence = normalize_dna_sequence((ROOT / "inputs/dna/sequence.txt").read_text())
        template = load_canonical_residue_com_template(
            ROOT / "inputs/protein" / construct / "MeCP2.pdb")
        packing = pack_project_geometry(
            template, component.sigmas, build_kapoor_2b_duplex_template(dna_sequence),
            protein_rotation_seed=seed, packing_translation_seed=seed + 1,
            dna_orientation_seed=seed + 2, dna_placement_seed=seed + 3)
        positions = packing.positions_nm
        append_kapoor2b_to_system(
            model.system, protein_indices=list(range(72900)),
            protein_charges_e=np.tile(component.qs, 150),
            protein_sigmas_nm=np.tile(component.sigmas, 150),
            protein_lambdas=np.tile(component.lambdas, 150),
            protein_ah_force=model.ah, protein_dh_force=model.yu,
            dna_sequence=dna_sequence, dna_coordinates_nm=positions[72900:],
            n_duplexes=3, temperature_k=310.0, ionic_strength_m=salt)
        topology, groups, edges, _ = build_topology(sequence, dna_sequence)
    topology.setPeriodicBoxVectors(tuple(np.diag(box)) * unit.nanometer)
    with (directory / "topology.pdb").open("w") as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)
    if not with_dna:
        positions = app.PDBFile(str(directory / "topology.pdb")).positions.value_in_unit(unit.nanometer)
    return model.system, topology, positions, box, groups, edges


def make_simulation(system, topology, platform, seed, temperature=310.0, timestep=0.01):
    from openmm import LangevinMiddleIntegrator, unit
    from .runtime import create_simulation
    integrator = LangevinMiddleIntegrator(temperature * unit.kelvin,
                                         0.01 / unit.picosecond, timestep * unit.picosecond)
    integrator.setRandomNumberSeed(seed)
    return create_simulation(topology, system, integrator, platform)


def install_host_state(context, state):
    """Transfer box, coordinates and velocities to a fresh preparation Context."""
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    context.setPositions(state.getPositions())
    context.setVelocities(state.getVelocities())


def finite_state(simulation):
    import numpy as np
    from openmm import unit
    state = simulation.context.getState(getEnergy=True, getForces=True,
                                         getPositions=True, getVelocities=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    if not all(np.isfinite(x).all() for x in (energy, forces, positions, velocities, box)):
        raise RuntimeError("The simulation contains non-finite state values.")
    if np.any(np.diag(box) <= 0):
        raise RuntimeError("Periodic box lengths must be positive.")


def prepare(system, topology, positions, box, groups, edges, args, with_dna, restored=None):
    import numpy as np
    import openmm as mm
    from openmm import unit
    from .preparation.spec import FROZEN_PROTOCOL as protocol
    from .preparation.system import rescale_box_preserving_molecules
    from .checkpoints import save_snapshot

    if restored is None:
        sim = make_simulation(system, topology, args.platform, args.seed + 4,
                              temperature=50.0 if with_dna else 310.0,
                              timestep=0.001 if with_dna else 0.01)
        sim.context.setPeriodicBoxVectors(*tuple(np.diag(box) * unit.nanometer))
        sim.context.setPositions(positions * unit.nanometer)
        progress = dict(stage="initial", completed=0)
    else:
        sim, metadata = restored
        system, topology = sim.system, sim.topology
        progress = metadata["progress"]

    def checkpoint(role):
        finite_state(sim)
        if with_dna:
            from .preparation.health import validate_geometry
            state = sim.context.getState(getPositions=True, getForces=True)
            validate_geometry(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                expected_forces=14 if role.startswith("compression") else 13,
                actual_forces=sim.system.getNumForces(), final_box=role == "final",
                final_minimization=progress["stage"] == "scaling" and
                                   progress["completed"] == protocol.final_scaling_stages)
        save_snapshot(sim, args.output, "preparation", progress, role)

    def lengths():
        return np.diag(sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))

    tolerance = 10.0 * unit.kilojoule_per_mole / unit.nanometer
    if progress["stage"] == "initial":
        sim.minimizeEnergy(tolerance=tolerance,
                           maxIterations=protocol.initial_minimization_iterations if with_dna else 0)
        if with_dna:
            sim.context.setVelocitiesToTemperature(50.0 * unit.kelvin, args.seed + 5)
        progress = dict(stage="warmup", completed=0)
        checkpoint("initial")
    if progress["stage"] == "warmup":
        stages = protocol.thermal_stages if with_dna else ()
        for index in range(progress["completed"], len(stages)):
            stage = stages[index]
            sim.integrator.setTemperature(stage.temperature_k * unit.kelvin)
            sim.integrator.setStepSize(stage.timestep_ps * unit.picosecond)
            sim.step(stage.steps)
            progress["completed"] = index + 1
            checkpoint("initial")
        progress = dict(stage="pre_equilibration", completed=0)
        checkpoint("initial")
    if progress["stage"] == "pre_equilibration":
        print("Pre-equilibration", flush=True)
        for block in range(progress["completed"], 10):
            sim.step(protocol.pre_equilibration_steps_per_block if with_dna else 1_000_000)
            progress["completed"] = block + 1
            checkpoint("initial")
        barostat = mm.MonteCarloAnisotropicBarostat(mm.Vec3(1, 1, 1) * unit.bar,
                                                   310 * unit.kelvin, False, False, True, 25)
        barostat.setRandomNumberSeed(args.seed + 6)
        transfer = sim.context.getState(getPositions=True, getVelocities=True) if with_dna else None
        system.addForce(barostat)
        if with_dna:
            del sim
            sim = make_simulation(system, topology, args.platform, args.seed + 4)
            install_host_state(sim.context, transfer)
        else:
            sim.context.reinitialize(preserveState=True)
        progress = dict(stage="compression", completed=0, steps=0)
        checkpoint("compression")
    if progress["stage"] == "compression":
        print("Compressing the z dimension to 50 nm", flush=True)
        barostat = system.getForce(system.getNumForces() - 1)
        limit = 5000 if with_dna else 30000
        while lengths()[2] > 50.0:
            block = progress["completed"] + 1
            adaptive = not with_dna and args.protein_protocol == "adaptive"
            if (not adaptive and block > limit) or progress["steps"] >= limit * 1000:
                raise RuntimeError("Compression did not reach the target box.")
            z = lengths()[2]
            batch = (1000 if z > 100 else 250 if z > 70 else 50 if z > 52 else 25) if adaptive else 1000
            pressure = protocol.pressure_for_block(block) if with_dna else 1.0
            sim.context.setParameter(barostat.PressureZ(), pressure)
            sim.step(batch)
            progress.update(completed=block, steps=progress["steps"] + batch)
            if not np.allclose(lengths()[:2], [50, 50], rtol=0, atol=1e-8):
                raise RuntimeError("Z compression changed the x/y box.")
            if block % 100 == 0 or lengths()[2] <= 50:
                print(f"Compression block {block}: z = {lengths()[2]:.3f} nm, Pz = {pressure} bar", flush=True)
                checkpoint("compression")
        barostat.setFrequency(0)
        sim.context.reinitialize(preserveState=True)
        progress = dict(stage="scaling", completed=0, origin_box=lengths().tolist())
        checkpoint("compression_disabled")
    if progress["stage"] == "scaling":
        origin = np.asarray(progress["origin_box"])
        stages = protocol.final_scaling_stages if with_dna else 1
        for stage in range(progress["completed"] + 1, stages + 1):
            target_box = origin + (np.full(3, 50.0) - origin) * stage / stages
            state = sim.context.getState(getPositions=True)
            current_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            scaled = (rescale_box_preserving_molecules(current_pos, lengths(), target_box, groups, edges)
                      if with_dna else current_pos * (target_box / lengths()))
            sim.context.setPositions(scaled * unit.nanometer)
            sim.context.setPeriodicBoxVectors(*tuple(np.diag(target_box) * unit.nanometer))
            sim.minimizeEnergy(tolerance=tolerance if with_dna else 10 * tolerance,
                               maxIterations=(protocol.final_minimization_iterations if stage == stages else
                                              protocol.intermediate_minimization_iterations) if with_dna else 0)
            progress["completed"] = stage
            checkpoint("compression_disabled")
        if with_dna:
            state = sim.context.getState(getPositions=True, getVelocities=True)
            system.removeForce(system.getNumForces() - 1)
            del sim
            system.setDefaultPeriodicBoxVectors(*tuple(np.eye(3) * 50.0 * unit.nanometer))
            sim = make_simulation(system, topology, args.platform, args.seed + 7)
            install_host_state(sim.context, state)
        progress = dict(stage="post_relaxation", completed=0)
        checkpoint("final")
    if progress["stage"] == "post_relaxation":
        for block in range(progress["completed"], protocol.post_relaxation_blocks if with_dna else 0):
            sim.step(protocol.post_relaxation_steps_per_block)
            progress["completed"] = block + 1
            checkpoint("final")
        progress = dict(stage="nvt_equilibration", completed=0, temperatures=[])
        checkpoint("final")
    if progress["stage"] == "nvt_equilibration" and with_dna:
        interval = protocol.nvt_temperature_sample_interval
        samples = protocol.nvt_equilibration_steps // interval
        dof = 3 * sum(system.getParticleMass(i).value_in_unit(unit.dalton) > 0
                      for i in range(system.getNumParticles())) - system.getNumConstraints()
        if any(isinstance(force, mm.CMMotionRemover) for force in system.getForces()):
            dof -= 3
        for block in range(progress["completed"], samples):
            sim.step(interval)
            kinetic = sim.context.getState(getEnergy=True).getKineticEnergy()
            temperature = (2 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)
            progress["temperatures"].append(float(temperature))
            progress["completed"] = block + 1
            if (block + 1) % 10 == 0:
                checkpoint("final")
        if len(progress["temperatures"]) != samples:
            raise RuntimeError("NVT temperature sample continuity differs from the protocol.")
        mean_temperature = float(np.mean(progress["temperatures"][samples // 2:]))
        if not protocol.nvt_last_half_temperature_min_k <= mean_temperature <= protocol.nvt_last_half_temperature_max_k:
            raise RuntimeError(f"Last 0.5 ns mean temperature is outside 305-315 K: {mean_temperature:.6f} K")
    finite_state(sim)
    if not np.allclose(lengths(), [50, 50, 50], rtol=0, atol=1e-8):
        raise RuntimeError("Production requires a 50 nm cubic box.")
    system.setDefaultPeriodicBoxVectors(*tuple(np.eye(3) * 50.0 * unit.nanometer))
    if with_dna or args.protein_protocol == "adaptive":
        sim.context.setTime(0 * unit.picosecond)
        sim.context.setStepCount(0)
    topology.setPeriodicBoxVectors(tuple(np.eye(3) * 50.0) * unit.nanometer)
    return sim


def main(story: str):
    p = parser(story)
    args = p.parse_args()
    target_steps = args.steps
    restored = None
    if args.resume:
        args.output = args.resume.expanduser().resolve()
        settings = json.loads((args.output / "settings.json").read_text())
        if settings["story"] != story:
            p.error("Resume entry point differs from the saved story.")
        for key in ("salt", "construct", "seed", "protein_protocol", "report_interval", "log_interval", "platform"):
            setattr(args, key, settings[key])
        args.steps = settings["steps"] if target_steps is None else target_steps
        if args.build_only:
            p.error("Build-only cannot resume a simulation.")
        from .checkpoints import restore_snapshot
        restored = restore_snapshot(args.output, args.platform)
    else:
        if args.seed is None or args.output is None:
            p.error("Fresh runs require --seed and --output.")
        args.steps = 500_000_000 if args.steps is None else args.steps
    if story != "mutation" and args.construct != "WT":
        p.error("Use simulation/mutation/run.py for mutant constructs.")
    if not 1 <= args.seed <= 2_147_483_630:
        p.error("Seed must be between 1 and 2147483630.")
    if args.steps < 1 or args.report_interval < 1 or args.log_interval < 1:
        p.error("Step count and report interval must be positive.")
    if args.protein_protocol == "adaptive" and (story != "mutation" or args.construct != "R306C"):
        p.error("The adaptive protein protocol applies to R306C n2.")
    args.output = args.output.expanduser().resolve()
    from openmm import XmlSerializer, app, unit
    if restored is None:
        args.output.mkdir(parents=True, exist_ok=False)
        system, topology, positions, box, groups, edges = build_initial(
            args.construct, args.salt, args.seed, args.output, story == "protein_dna")
        (args.output / "initial_system.xml").write_text(XmlSerializer.serialize(system))
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        settings.update(story=story, protein_copies=150, residues_per_chain=486,
                        dna_duplexes=3 if story == "protein_dna" else 0,
                        temperature_k=310.0, timestep_ps=0.01)
        (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        if args.build_only:
            return
    else:
        sim, metadata = restored
        system, topology = sim.system, sim.topology
        positions = box = None
        groups = edges = ()
        if story == "protein_dna" and metadata["phase"] == "preparation":
            from .topology import build_topology
            from .preparation.geometry import normalize_dna_sequence
            sequence = "".join(line.strip() for line in
                               (ROOT / "inputs/protein/WT/MeCP2.fasta").read_text().splitlines()
                               if not line.startswith(">"))
            _, groups, edges, _ = build_topology(sequence, normalize_dna_sequence(
                (ROOT / "inputs/dna/sequence.txt").read_text()))
    progress = restored[1]["progress"] if restored and restored[1]["phase"] == "production" else None
    if progress is None:
        sim = prepare(system, topology, positions, box, groups, edges, args,
                      story == "protein_dna", restored=restored)
    (args.output / "system.xml").write_text(XmlSerializer.serialize(sim.system))
    if progress is None:
        with (args.output / "production_initial.pdb").open("w") as handle:
            app.PDBFile.writeFile(topology, sim.context.getState(getPositions=True).getPositions(), handle)
    from .production import run_production
    run_production(sim, args.output, args.steps, args.report_interval, progress,
                   log_interval=args.log_interval)

