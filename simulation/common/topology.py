"""Particle ordering and molecular topology for the protein-DNA system."""
from typing import Mapping, Any
from openmm import app, openmm, unit
from .preparation.geometry import reverse_complement
N_PROTEIN = 72900
N_RESIDUES = 486
N_PROTEINS = 150
N_BP = 376
N_DUPLEXES = 3
N_TOTAL = 77412

def _initial_topology_box_vectors() -> object:
    """Return one unit-bearing 50 x 50 x 500 nm topology box.

    OpenMM's PDB writer recognizes a single ``Quantity`` containing three
    vectors.  A tuple of three independently unit-bearing vectors looks
    unitless to ``computeLengthsAndAngles`` and fails while formatting CRYST1.
    """

    return unit.Quantity(
        (
            openmm.Vec3(50.0, 0.0, 0.0),
            openmm.Vec3(0.0, 50.0, 0.0),
            openmm.Vec3(0.0, 0.0, 500.0),
        ),
        unit.nanometer,
    )


def build_topology(
    protein_sequence: str,
    dna_sequence: str,
) -> tuple[object, tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...], Mapping[str, Any]]:
    topology = app.Topology()
    groups: list[tuple[int, ...]] = []
    edges: list[tuple[int, int]] = []
    atom_by_index: list[object] = []
    particle = 0
    for copy_index in range(N_PROTEINS):
        chain = topology.addChain(f"P{copy_index + 1}")
        group = []
        previous = None
        for residue_index, residue_code in enumerate(protein_sequence):
            residue = topology.addResidue(residue_code, chain, id=str(residue_index + 1))
            atom = topology.addAtom("CA", app.element.carbon, residue)
            atom_by_index.append(atom)
            group.append(particle)
            if previous is not None:
                topology.addBond(previous, atom)
                edges.append((particle - 1, particle))
            previous = atom
            particle += 1
        groups.append(tuple(group))
    complement = reverse_complement(dna_sequence)
    dna_registry = []
    for duplex in range(N_DUPLEXES):
        duplex_group = []
        strand_base_indices = []
        for strand_id, sequence in enumerate((dna_sequence, complement)):
            chain = topology.addChain(f"D{duplex + 1}{'A' if strand_id == 0 else 'B'}")
            previous_bb_index = None
            bases = []
            for position, base in enumerate(sequence):
                residue = topology.addResidue("D" + base, chain, id=str(position + 1))
                bb = topology.addAtom("BB", app.element.phosphorus, residue)
                base_atom = topology.addAtom(base, app.element.nitrogen, residue)
                atom_by_index.extend((bb, base_atom))
                bb_index = particle
                base_index = particle + 1
                topology.addBond(bb, base_atom)
                edges.append((bb_index, base_index))
                if previous_bb_index is not None:
                    topology.addBond(atom_by_index[previous_bb_index], bb)
                    edges.append((previous_bb_index, bb_index))
                previous_bb_index = bb_index
                bases.append(base_index)
                duplex_group.extend((bb_index, base_index))
                particle += 2
            strand_base_indices.append(bases)
        for position in range(N_BP):
            first = strand_base_indices[0][position]
            second = strand_base_indices[1][N_BP - 1 - position]
            edges.append((first, second))
            dna_registry.append(
                {
                    "duplex_id": duplex + 1,
                    "strand_id": "A",
                    "position": position,
                    "particle_index": first,
                    "native_partner": second,
                    "base": dna_sequence[position],
                }
            )
            dna_registry.append(
                {
                    "duplex_id": duplex + 1,
                    "strand_id": "B",
                    "position": N_BP - 1 - position,
                    "particle_index": second,
                    "native_partner": first,
                    "base": complement[N_BP - 1 - position],
                }
            )
        groups.append(tuple(duplex_group))
    if particle != N_TOTAL or topology.getNumAtoms() != N_TOTAL:
        raise RuntimeError("assembled topology particle count is not 77,412")
    topology.setPeriodicBoxVectors(_initial_topology_box_vectors())
    registry = {
        "schema": "mecp2-dna-fd-topology-identity-registry-v1",
        "status": "PASS",
        "particles": N_TOTAL,
        "protein_copies": N_PROTEINS,
        "protein_residues_per_copy": N_RESIDUES,
        "dna_duplexes": N_DUPLEXES,
        "base_pairs_per_duplex": N_BP,
        "molecule_ranges": [
            {"kind": "protein", "id": i + 1, "first": i * N_RESIDUES, "count": N_RESIDUES}
            for i in range(N_PROTEINS)
        ]
        + [
            {
                "kind": "dna_duplex",
                "id": i + 1,
                "first": N_PROTEIN + i * (4 * N_BP),
                "count": 4 * N_BP,
            }
            for i in range(N_DUPLEXES)
        ],
        "dna_base_identity": dna_registry,
    }
    return topology, tuple(groups), tuple(edges), registry
