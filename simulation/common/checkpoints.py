"""Atomic checkpoint generations for preparation and production recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_snapshot(simulation, directory, phase, progress, system_role):
    from openmm import XmlSerializer

    directory = Path(directory)
    store = directory / "checkpoints"
    store.mkdir(exist_ok=True)
    system_file = store / (system_role + ".xml")
    if not system_file.exists():
        temporary = system_file.with_suffix(".xml.tmp")
        temporary.write_text(XmlSerializer.serialize(simulation.system))
        os.replace(temporary, system_file)
    generation = store / ("generation-" + uuid.uuid4().hex)
    generation.mkdir()
    checkpoint = generation / "checkpoint.chk"
    simulation.saveCheckpoint(str(checkpoint))
    integrator = generation / "integrator.xml"
    integrator.write_text(XmlSerializer.serialize(simulation.integrator))
    metadata = dict(phase=phase, progress=progress, step=int(simulation.currentStep),
                    platform=simulation.context.getPlatform().getName(),
                    system=str(system_file.relative_to(directory)), system_sha256=digest(system_file),
                    integrator_sha256=digest(integrator), checkpoint_sha256=digest(checkpoint),
                    topology_sha256=digest(directory / "topology.pdb"),
                    settings_sha256=digest(directory / "settings.json"))
    write_json(generation / "metadata.json", metadata)
    pointer_path = store / "current.json"
    previous = json.loads(pointer_path.read_text())["generation"] if pointer_path.exists() else None
    write_json(pointer_path, dict(generation=generation.name))
    # Keep the current and preceding complete generations.
    for obsolete in store.glob("generation-*"):
        if obsolete.name not in (generation.name, previous):
            shutil.rmtree(obsolete)
    return metadata


def restore_snapshot(directory, platform=None):
    from openmm import XmlSerializer, app
    from .runtime import create_simulation

    directory = Path(directory)
    store = directory / "checkpoints"
    name = json.loads((store / "current.json").read_text())["generation"]
    if Path(name).name != name or not name.startswith("generation-"):
        raise ValueError("Invalid checkpoint generation.")
    generation = store / name
    metadata = json.loads((generation / "metadata.json").read_text())
    system_file = (directory / metadata["system"]).resolve()
    if not system_file.is_relative_to(store.resolve()):
        raise ValueError("Checkpoint system is outside the checkpoint store.")
    for path, expected in ((system_file, metadata["system_sha256"]),
                           (generation / "integrator.xml", metadata["integrator_sha256"]),
                           (generation / "checkpoint.chk", metadata["checkpoint_sha256"]),
                           (directory / "topology.pdb", metadata["topology_sha256"]),
                           (directory / "settings.json", metadata["settings_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"Checkpoint input checksum differs: {path.name}")
    system = XmlSerializer.deserialize(system_file.read_text())
    integrator = XmlSerializer.deserialize((generation / "integrator.xml").read_text())
    topology = app.PDBFile(str(directory / "topology.pdb")).topology
    simulation = create_simulation(topology, system, integrator,
                                   platform or metadata.get("platform"))
    simulation.loadCheckpoint(str(generation / "checkpoint.chk"))
    if simulation.currentStep != metadata["step"]:
        raise ValueError("Checkpoint step differs from its metadata.")
    return simulation, metadata
