"""Fresh residue-COM protein and antiparallel Kapoor-2B DNA packing.

Only coordinates are constructed here.  No OpenMM Force or System is created
or modified.  Every molecule is placed by a proper rigid transform, checked
against periodic self images, and rejected on an explicit steric clash.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - production CALVADOS env includes scipy
    cKDTree = None  # type: ignore[assignment]


ASCII_WHITESPACE = " \t\r\n\v\f"
KAPOOR_SOURCE_EXAMPLE_SHA256 = (
    "fa24fc35da4b11baa3f921e1b93fe9063f5357925e8b7624eb9cccf0bb8843c0"
)

# Standard elemental masses used by MDAnalysis for the elements present in
# the canonical MeCP2 PDB.  Explicit element-column parsing prevents a CA atom
# from being mistaken for calcium.  Values are sufficient to reproduce the
# all-residue mass-COM mapping; CA-only mapping is not an available option.
ELEMENT_MASS_DA: Mapping[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.06,
    "SE": 78.971,
}


class PackingFailure(RuntimeError):
    """No clash-free rigid placement was found within the frozen attempt cap."""


@dataclass(frozen=True)
class ProteinTemplate:
    positions_nm: np.ndarray
    residue_keys: Tuple[str, ...]
    mapping: str = "canonical_all_residue_mass_COM"


@dataclass(frozen=True)
class DuplexTemplate:
    positions_nm: np.ndarray
    strand_sequences: Tuple[str, str]
    rise_nm: float
    twist_deg: float
    source_example_sha256: str = KAPOOR_SOURCE_EXAMPLE_SHA256


@dataclass(frozen=True)
class RigidPlacement:
    molecule_id: str
    molecule_kind: str
    rotation: np.ndarray
    translation_nm: np.ndarray
    attempts: int
    first_particle: int
    particle_count: int
    strand_ids: Tuple[int, ...] = ()
    strand_sequences: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PackingResult:
    positions_nm: np.ndarray
    unwrapped_positions_nm: np.ndarray
    box_nm: np.ndarray
    particle_diameters_nm: np.ndarray
    placements: Tuple[RigidPlacement, ...]
    protein_particle_count: int
    dna_particle_count: int
    long_dna_qa: Mapping[str, object]

    @property
    def total_particles(self) -> int:
        return int(self.positions_nm.shape[0])


def normalize_dna_sequence(raw: str, *, expected_length: int | None = 376) -> str:
    """Remove ASCII whitespace only, then require uppercase A/C/G/T."""

    sequence = raw.translate({ord(char): None for char in ASCII_WHITESPACE}).upper()
    if expected_length is not None and len(sequence) != expected_length:
        raise ValueError(
            f"normalized DNA length must be {expected_length}; observed {len(sequence)}"
        )
    invalid = sorted(set(sequence) - set("ACGT"))
    if invalid:
        raise ValueError(f"DNA sequence contains non-ACGT symbols: {invalid}")
    return sequence


def reverse_complement(sequence: str) -> str:
    sequence = normalize_dna_sequence(sequence, expected_length=None)
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _infer_pdb_element(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    atom_name = line[12:16].strip().upper()
    atom_name = atom_name.lstrip("0123456789")
    if not atom_name:
        raise ValueError("cannot infer element from blank PDB atom name")
    if len(atom_name) >= 2 and atom_name[:2] in ELEMENT_MASS_DA:
        return atom_name[:2]
    return atom_name[0]


def load_canonical_residue_com_template(
    pdb_path: Path, *, expected_residues: int = 486
) -> ProteinTemplate:
    """Reproduce frozen CALVADOS ``geometry_from_pdb(use_com=True)``.

    All atoms are first translated by the all-atom global center of mass, then
    each residue bead is its mass-weighted atom COM.  There is deliberately no
    CA fallback or mapping flag.
    """

    # CALVADOS geometry_from_pdb(use_com=True) uses MDAnalysis topology
    # masses. Calling the same implementation avoids drift from rounded,
    # hand-maintained elemental mass tables.
    try:
        from MDAnalysis import Universe
    except ImportError as exc:  # pragma: no cover - production requirement
        raise RuntimeError(
            "MDAnalysis is required for the frozen use_com=True mapping"
        ) from exc
    universe = Universe(str(Path(pdb_path)))
    if len(universe.residues) != expected_residues:
        raise RuntimeError(
            f"canonical PDB must contain {expected_residues} residues; "
            f"observed {len(universe.residues)}"
        )
    global_com = np.asarray(universe.atoms.center_of_mass(), dtype=np.float64)
    if global_com.shape != (3,) or not np.isfinite(global_com).all():
        raise RuntimeError("canonical PDB has no finite all-atom center of mass")
    universe.atoms.translate(-global_com)
    result = np.asarray(
        [residue.atoms.center_of_mass() for residue in universe.residues],
        dtype=np.float64,
    ) * 0.1
    keys = tuple(
        f"{residue.segid}:{residue.resid}:"
        f"{getattr(residue, 'icode', '')}:{residue.resname}"
        for residue in universe.residues
    )
    if result.shape != (expected_residues, 3) or not np.isfinite(result).all():
        raise RuntimeError("canonical residue-COM geometry is non-finite or malformed")
    result.setflags(write=False)
    return ProteinTemplate(result, keys)


def load_protein_particle_diameters(
    fasta_path: Path, residue_csv_path: Path, *, expected_residues: int = 486
) -> np.ndarray:
    """Return the frozen CALVADOS3 sigma for every residue bead."""

    lines = Path(fasta_path).read_text(encoding="ascii").splitlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    if len(sequence) != expected_residues:
        raise RuntimeError(
            f"protein sequence must contain {expected_residues} residues; observed {len(sequence)}"
        )
    sigma_by_residue: dict[str, float] = {}
    with Path(residue_csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sigma_by_residue[str(row["one"])] = float(row["sigmas"])
    try:
        result = np.asarray([sigma_by_residue[residue] for residue in sequence], dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError(f"residue sigma missing for {exc.args[0]!r}") from exc
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("protein particle diameters must be finite and positive")
    return result


def build_kapoor_2b_duplex_template(sequence: str) -> DuplexTemplate:
    """Build a source-consistent, bond-aware antiparallel two-bead helix.

    The source example has a 0.44-nm radial backbone-to-base offset, 0.34-nm
    B-DNA rise and approximately 36-degree twist.  A shared axial offset makes
    the frozen 0.575-nm BB-base bond exact, while the backbone radius is solved
    so the frozen 0.550-nm BB-BB bond is exact.  Strand B is stored 5'->3' as
    the reverse complement and runs in the opposite spatial direction.

    These are initial coordinates only; Hamiltonian values remain exclusively
    owned by the frozen builder.
    """

    seq_a = normalize_dna_sequence(sequence, expected_length=None)
    seq_b = reverse_complement(seq_a)
    n_bp = len(seq_a)
    if n_bp < 2:
        raise ValueError("a duplex requires at least two base pairs")
    rise_nm = 0.34
    twist_deg = 36.0
    twist = math.radians(twist_deg)
    bb_step_nm = 0.550
    bb_base_nm = 0.575
    radial_bb_base_nm = 0.440
    tangential_chord_factor = 2.0 * math.sin(0.5 * twist)
    radial_step_sq = bb_step_nm**2 - rise_nm**2
    if radial_step_sq <= 0.0:
        raise AssertionError("invalid frozen helix geometry")
    backbone_radius = math.sqrt(radial_step_sq) / tangential_chord_factor
    base_radius = backbone_radius - radial_bb_base_nm
    base_axial_offset = math.sqrt(bb_base_nm**2 - radial_bb_base_nm**2)
    if base_radius <= 0.0:
        raise AssertionError("invalid source-derived base radius")

    coords = np.empty((2, n_bp, 2, 3), dtype=np.float64)
    midpoint = 0.5 * (n_bp - 1)
    for i in range(n_bp):
        angle = i * twist
        z = (i - midpoint) * rise_nm
        radial = np.array([math.cos(angle), math.sin(angle), 0.0])
        bb_a = backbone_radius * radial + np.array([0.0, 0.0, z])
        base_a = base_radius * radial + np.array([0.0, 0.0, z + base_axial_offset])
        coords[0, i, 0] = bb_a
        coords[0, i, 1] = base_a

        # Native partner A(i) <-> B(n-1-i).  Filling B at reversed storage
        # index makes its 5'->3' backbone direction antiparallel to A.
        j = n_bp - 1 - i
        opposite = -radial
        coords[1, j, 0] = backbone_radius * opposite + np.array([0.0, 0.0, z])
        coords[1, j, 1] = base_radius * opposite + np.array(
            [0.0, 0.0, z + base_axial_offset]
        )

    bb_steps_a = np.linalg.norm(np.diff(coords[0, :, 0], axis=0), axis=1)
    bb_steps_b = np.linalg.norm(np.diff(coords[1, :, 0], axis=0), axis=1)
    bb_base = np.linalg.norm(coords[:, :, 0] - coords[:, :, 1], axis=2)
    if not np.allclose(bb_steps_a, bb_step_nm, rtol=0.0, atol=1e-12):
        raise AssertionError("strand A backbone geometry drift")
    if not np.allclose(bb_steps_b, bb_step_nm, rtol=0.0, atol=1e-12):
        raise AssertionError("strand B backbone geometry drift")
    if not np.allclose(bb_base, bb_base_nm, rtol=0.0, atol=1e-12):
        raise AssertionError("BB-base geometry drift")
    coords -= np.mean(coords.reshape(-1, 3), axis=0)
    coords.setflags(write=False)
    return DuplexTemplate(coords, (seq_a, seq_b), rise_nm, twist_deg)


def uniform_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Draw a proper rotation uniformly from SO(3) (Shoemake quaternion)."""

    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
        raise AssertionError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("rotation is not proper (det != +1)")
    return rotation


def _self_image_safe(
    unwrapped_nm: np.ndarray, diameters_nm: np.ndarray, box_nm: np.ndarray
) -> bool:
    """Sufficient, explicit self-image separation check for an orthorhombic box."""

    spans = np.ptp(unwrapped_nm, axis=0)
    max_diameter = float(np.max(diameters_nm))
    return bool(np.all(spans + max_diameter < box_nm))


def _has_periodic_clash(
    candidate_wrapped_nm: np.ndarray,
    candidate_diameters_nm: np.ndarray,
    existing_wrapped_nm: np.ndarray,
    existing_diameters_nm: np.ndarray,
    box_nm: np.ndarray,
) -> bool:
    if existing_wrapped_nm.size == 0:
        return False
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for full-system packing")
    tree = cKDTree(existing_wrapped_nm, boxsize=box_nm)
    search_radii = 0.5 * (candidate_diameters_nm + float(np.max(existing_diameters_nm)))
    neighbor_lists = tree.query_ball_point(candidate_wrapped_nm, search_radii)
    for point_index, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        delta = existing_wrapped_nm[np.asarray(neighbors, dtype=int)] - candidate_wrapped_nm[
            point_index
        ]
        delta -= box_nm * np.rint(delta / box_nm)
        distances = np.linalg.norm(delta, axis=1)
        thresholds = 0.5 * (
            candidate_diameters_nm[point_index]
            + existing_diameters_nm[np.asarray(neighbors, dtype=int)]
        )
        if np.any(distances < thresholds):
            return True
    return False


def _place_one(
    *,
    molecule_id: str,
    molecule_kind: str,
    template_nm: np.ndarray,
    diameters_nm: np.ndarray,
    rotation_rng: np.random.Generator,
    translation_rng: np.random.Generator,
    existing_positions: np.ndarray,
    existing_diameters: np.ndarray,
    box_nm: np.ndarray,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    centered = np.asarray(template_nm, dtype=np.float64) - np.mean(template_nm, axis=0)
    for attempt in range(1, max_attempts + 1):
        rotation = uniform_rotation_matrix(rotation_rng)
        translation = translation_rng.random(3) * box_nm
        unwrapped = centered @ rotation.T + translation
        if not _self_image_safe(unwrapped, diameters_nm, box_nm):
            continue
        wrapped = np.mod(unwrapped, box_nm)
        if _has_periodic_clash(
            wrapped,
            diameters_nm,
            existing_positions,
            existing_diameters,
            box_nm,
        ):
            continue
        return wrapped, unwrapped, rotation, translation, attempt
    raise PackingFailure(
        f"hard STOP: no PBC/self-image/clash-free placement for {molecule_kind} "
        f"{molecule_id} after {max_attempts} attempts"
    )


def _strict_distance_violation_count(
    first_tree: object, second_tree: object, gate_nm: float
) -> int:
    strict_radius = float(np.nextafter(float(gate_nm), -np.inf))
    return int(
        sum(len(items) for items in first_tree.query_ball_tree(second_tree, strict_radius))
    )


def long_dna_periodic_qa(
    wrapped_duplexes_nm: Sequence[np.ndarray],
    unwrapped_duplexes_nm: Sequence[np.ndarray],
    box_nm: Sequence[float],
    *,
    packing_gate_nm: float = 0.55,
) -> Mapping[str, object]:
    """Exact initial inter-duplex/self-image evidence for LONG_DNA."""

    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for LONG_DNA QA")
    box = np.asarray(box_nm, dtype=np.float64)
    wrapped = [np.asarray(value, dtype=np.float64) for value in wrapped_duplexes_nm]
    unwrapped = [np.asarray(value, dtype=np.float64) for value in unwrapped_duplexes_nm]
    if len(wrapped) != 3 or len(unwrapped) != 3:
        raise RuntimeError("LONG_DNA QA requires exactly three duplexes")
    inter_records = []
    for first in range(3):
        tree_first = cKDTree(wrapped[first], boxsize=box)
        for second in range(first + 1, 3):
            tree_second = cKDTree(wrapped[second], boxsize=box)
            distances, _ = tree_first.query(wrapped[second], k=1)
            inter_records.append(
                {
                    "duplex_a": first + 1,
                    "duplex_b": second + 1,
                    "minimum_center_distance_nm": float(np.min(distances)),
                    "violation_count": _strict_distance_violation_count(
                        tree_first, tree_second, packing_gate_nm
                    ),
                }
            )
    self_records = []
    lattice_shifts = [
        np.asarray((i, j, k), dtype=np.float64) * box
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ]
    for index, raw in enumerate(unwrapped):
        tree = cKDTree(raw)
        minimum = math.inf
        violations = 0
        for shift in lattice_shifts:
            shifted = cKDTree(raw + shift)
            distances, _ = tree.query(raw + shift, k=1)
            minimum = min(minimum, float(np.min(distances)))
            violations += _strict_distance_violation_count(tree, shifted, packing_gate_nm)
        self_records.append(
            {
                "duplex": index + 1,
                "minimum_center_distance_nm": minimum,
                "violation_count": int(violations),
                "pbc_spans_nm": np.ptp(raw, axis=0).tolist(),
            }
        )
    inter_violations = sum(int(item["violation_count"]) for item in inter_records)
    self_violations = sum(int(item["violation_count"]) for item in self_records)
    if inter_violations or self_violations:
        raise PackingFailure(
            "hard STOP: initial LONG_DNA 0.55-nm periodic packing gate has violations"
        )
    return {
        "schema": "mecp2-dna-fd-long-dna-initial-packing-qa-v1",
        "status": "PASS",
        "packing_gate_nm": float(packing_gate_nm),
        "distance_definition": "bead-center minimum; strict violation if distance < gate",
        "inter_duplex": inter_records,
        "self_image": self_records,
        "initial_inter_duplex_min_distance_nm": min(
            float(item["minimum_center_distance_nm"]) for item in inter_records
        ),
        "initial_self_image_min_distance_nm": min(
            float(item["minimum_center_distance_nm"]) for item in self_records
        ),
        "inter_duplex_violation_count": inter_violations,
        "self_image_violation_count": self_violations,
        "total_violation_count": inter_violations + self_violations,
    }


def pack_project_geometry(
    protein_template: ProteinTemplate,
    protein_diameters_nm: Sequence[float],
    dna_template: DuplexTemplate,
    *,
    protein_rotation_seed: int,
    packing_translation_seed: int,
    dna_orientation_seed: int,
    dna_placement_seed: int,
    box_nm: Sequence[float] = (50.0, 50.0, 500.0),
    protein_copies: int = 150,
    dna_duplexes: int = 3,
    dna_diameter_nm: float = 0.55,
    max_attempts_per_molecule: int = 10_000,
) -> PackingResult:
    """Pack the complete 77,412-particle initial geometry deterministically."""

    box = np.asarray(box_nm, dtype=np.float64)
    if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0.0):
        raise ValueError("box_nm must be three finite positive lengths")
    if tuple(float(value) for value in box) != (50.0, 50.0, 500.0):
        raise ValueError("production pre-compression box is frozen at 50x50x500 nm")
    protein_coords = np.asarray(protein_template.positions_nm, dtype=np.float64)
    protein_diameters = np.asarray(protein_diameters_nm, dtype=np.float64)
    if protein_template.mapping != "canonical_all_residue_mass_COM":
        raise RuntimeError("CA or mixed protein mapping is forbidden")
    if protein_coords.shape != (486, 3) or protein_diameters.shape != (486,):
        raise RuntimeError("canonical protein geometry must be exactly 486 beads")
    if protein_copies != 150 or dna_duplexes != 3:
        raise ValueError("production composition is frozen at 150 proteins and 3 duplexes")
    if max_attempts_per_molecule < 1:
        raise ValueError("max_attempts_per_molecule must be positive")

    protein_rot_rng = np.random.default_rng(int(protein_rotation_seed))
    protein_trans_rng = np.random.default_rng(int(packing_translation_seed))
    dna_rot_rng = np.random.default_rng(int(dna_orientation_seed))
    dna_trans_rng = np.random.default_rng(int(dna_placement_seed))
    existing_positions = np.empty((0, 3), dtype=np.float64)
    existing_diameters = np.empty((0,), dtype=np.float64)
    dna_flat = np.asarray(dna_template.positions_nm, dtype=np.float64).reshape(-1, 3)
    dna_diameters = np.full(dna_flat.shape[0], float(dna_diameter_nm), dtype=np.float64)
    dna_results = []
    dna_unwrapped_results = []
    dna_placements = []

    # Long duplexes are packed first.  Their uniform SO(3) draws are accepted
    # only when the full explicit geometry passes the 50-nm x/y self-image gate.
    for duplex_id in range(dna_duplexes):
        placed, unwrapped, rotation, translation, attempts = _place_one(
            molecule_id=f"duplex{duplex_id + 1}",
            molecule_kind="dna_duplex",
            template_nm=dna_flat,
            diameters_nm=dna_diameters,
            rotation_rng=dna_rot_rng,
            translation_rng=dna_trans_rng,
            existing_positions=existing_positions,
            existing_diameters=existing_diameters,
            box_nm=box,
            max_attempts=max_attempts_per_molecule,
        )
        dna_results.append(placed)
        dna_unwrapped_results.append(unwrapped)
        dna_placements.append((rotation, translation, attempts))
        existing_positions = np.concatenate((existing_positions, placed), axis=0)
        existing_diameters = np.concatenate((existing_diameters, dna_diameters), axis=0)

    protein_results = []
    protein_unwrapped_results = []
    protein_placements = []
    for copy_index in range(protein_copies):
        placed, unwrapped, rotation, translation, attempts = _place_one(
            molecule_id=f"protein{copy_index + 1}",
            molecule_kind="protein",
            template_nm=protein_coords,
            diameters_nm=protein_diameters,
            rotation_rng=protein_rot_rng,
            translation_rng=protein_trans_rng,
            existing_positions=existing_positions,
            existing_diameters=existing_diameters,
            box_nm=box,
            max_attempts=max_attempts_per_molecule,
        )
        protein_results.append(placed)
        protein_unwrapped_results.append(unwrapped)
        protein_placements.append((rotation, translation, attempts))
        existing_positions = np.concatenate((existing_positions, placed), axis=0)
        existing_diameters = np.concatenate((existing_diameters, protein_diameters), axis=0)

    # The builder's required particle order is all proteins first, then each
    # duplex with strand A followed by antiparallel reverse-complement strand B.
    final_positions = np.concatenate((*protein_results, *dna_results), axis=0)
    final_unwrapped_positions = np.concatenate(
        (*protein_unwrapped_results, *dna_unwrapped_results), axis=0
    )
    final_diameters = np.concatenate(
        (*([protein_diameters] * protein_copies), *([dna_diameters] * dna_duplexes)),
        axis=0,
    )
    placements = []
    first = 0
    for copy_index, (rotation, translation, attempts) in enumerate(protein_placements):
        placements.append(
            RigidPlacement(
                molecule_id=f"protein{copy_index + 1}",
                molecule_kind="protein",
                rotation=rotation,
                translation_nm=translation,
                attempts=attempts,
                first_particle=first,
                particle_count=486,
            )
        )
        first += 486
    dna_particles_per_duplex = int(dna_flat.shape[0])
    for duplex_id, (rotation, translation, attempts) in enumerate(dna_placements):
        placements.append(
            RigidPlacement(
                molecule_id=f"duplex{duplex_id + 1}",
                molecule_kind="dna_duplex",
                rotation=rotation,
                translation_nm=translation,
                attempts=attempts,
                first_particle=first,
                particle_count=dna_particles_per_duplex,
                strand_ids=(2 * duplex_id, 2 * duplex_id + 1),
                strand_sequences=dna_template.strand_sequences,
            )
        )
        first += dna_particles_per_duplex
    if final_positions.shape != (77_412, 3) or first != 77_412:
        raise RuntimeError("hard STOP: packed production particle count is not 77,412")
    if not np.isfinite(final_positions).all() or not np.isfinite(
        final_unwrapped_positions
    ).all():
        raise RuntimeError("hard STOP: packed positions are non-finite")
    if not np.allclose(
        np.mod(final_unwrapped_positions, box),
        final_positions,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("hard STOP: wrapped/unwrapped packed coordinates disagree")
    long_dna_qa = long_dna_periodic_qa(
        dna_results,
        dna_unwrapped_results,
        box,
        packing_gate_nm=float(dna_diameter_nm),
    )

    protein_rotations = [p.rotation for p in placements if p.molecule_kind == "protein"]
    protein_translations = [p.translation_nm for p in placements if p.molecule_kind == "protein"]
    dna_rotations = [p.rotation for p in placements if p.molecule_kind == "dna_duplex"]
    dna_translations = [p.translation_nm for p in placements if p.molecule_kind == "dna_duplex"]
    if len({matrix.tobytes() for matrix in protein_rotations}) != 150:
        raise RuntimeError("hard STOP: protein rigid rotations are not independent")
    if len({vector.tobytes() for vector in protein_translations}) != 150:
        raise RuntimeError("hard STOP: protein translations are not independent")
    if len({matrix.tobytes() for matrix in dna_rotations}) != 3:
        raise RuntimeError("hard STOP: DNA orientations are not independent")
    if len({vector.tobytes() for vector in dna_translations}) != 3:
        raise RuntimeError("hard STOP: DNA placements are not independent")
    return PackingResult(
        positions_nm=final_positions,
        unwrapped_positions_nm=final_unwrapped_positions,
        box_nm=box,
        particle_diameters_nm=final_diameters,
        placements=tuple(placements),
        protein_particle_count=72_900,
        dna_particle_count=4_512,
        long_dna_qa=long_dna_qa,
    )


def minimum_periodic_distance(
    first_nm: np.ndarray, second_nm: np.ndarray, box_nm: Sequence[float]
) -> np.ndarray:
    """Vectorized orthorhombic minimum-image distances used by packing QA."""

    box = np.asarray(box_nm, dtype=np.float64)
    delta = np.asarray(first_nm, dtype=np.float64) - np.asarray(second_nm, dtype=np.float64)
    delta -= box * np.rint(delta / box)
    return np.linalg.norm(delta, axis=-1)

