"""OpenMM port of the Kapoor two-bead DNA model for frozen CALVADOS3 proteins.

This module appends DNA particles and DNA/DNA + protein/DNA forces to an
already constructed protein ``openmm.System``.

Canonical DNA particle order
----------------------------
For every duplex, strand A is followed by strand B.  Within each strand, every
nucleotide is stored as ``[backbone, base]``.  Strand B is the reverse
complement of strand A and native partners are therefore A(i) <-> B(N-1-i).

Provenance
----------
Kapoor source repository commit:
``5a93fc279f6a6c498c55ed6992493d6cb6c0d952``.
Frozen local CALVADOS commit:
``dc5beac158b72262f585947487c1dd8ef1602a18``.

All public numeric inputs and outputs use OpenMM units: nm, ps, kJ/mol, Da,
and elementary charge.  Source ``units real`` parameters are converted once at
module definition time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from openmm import openmm, unit
except ImportError:  # pragma: no cover - compatibility with older OpenMM only
    from simtk import openmm, unit  # type: ignore


KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1

# Kapoor 2B mapping.  The masses follow the source 2-bead in.data type map:
# type 1=backbone, 2=G, 3=C, 4=T, 5=A.
DNA_MASS_DA: Mapping[str, float] = {
    "BB": 178.08,
    "A": 134.10,
    "T": 125.10,
    "C": 110.10,
    "G": 150.10,
}
DNA_CHARGE_E: Mapping[str, float] = {"BB": -1.0, "A": 0.0, "T": 0.0, "C": 0.0, "G": 0.0}
DNA_LAMBDA: Mapping[str, float] = {"BB": 0.38, "A": 0.40, "T": 0.54, "C": 0.59, "G": 0.35}
DNA_PD_SIGMA_NM = 0.55

# Bond source convention is U=K(r-r0)^2.  OpenMM HarmonicBondForce uses
# U=0.5*k(r-r0)^2, hence the explicit factor of two.
SOURCE_BOND_K_KCAL_PER_A2 = 50.0
OPENMM_BOND_K_KJ_PER_NM2 = 2.0 * SOURCE_BOND_K_KCAL_PER_A2 * KCAL_TO_KJ / (ANGSTROM_TO_NM**2)
BB_BB_R0_NM = 0.55
BB_BASE_R0_NM = 0.575

ANGLE_K_KJ_PER_MOL = 40.0 * KCAL_TO_KJ
ANGLE_THETA0_RAD = math.pi

WCA_EPSILON_KJ_PER_MOL = 4.0 * KCAL_TO_KJ
WCA_SIGMA_NM = 0.55
WCA_CUTOFF_NM = (2.0 ** (1.0 / 6.0)) * WCA_SIGMA_NM

STACK_SIGMA_NM = 0.36
STACK_CUTOFF_NM = 0.62
STACK_DELTA_KJ_PER_MOL: Mapping[str, float] = {
    "AA": 9.7 * KCAL_TO_KJ,
    "AT": 8.6 * KCAL_TO_KJ,
    "AC": 7.8 * KCAL_TO_KJ,
    "AG": 10.3 * KCAL_TO_KJ,
    "TT": 7.7 * KCAL_TO_KJ,
    "TC": 6.9 * KCAL_TO_KJ,
    "TG": 9.2 * KCAL_TO_KJ,
    "CC": 6.2 * KCAL_TO_KJ,
    "CG": 8.3 * KCAL_TO_KJ,
    "GG": 11.0 * KCAL_TO_KJ,
}

HB_CUTOFF_NM = 0.95
HB_PARAMETERS: Mapping[str, Tuple[float, float]] = {
    "AT": (0.60, 2.7 * KCAL_TO_KJ),
    "CG": (0.55, 3.3 * KCAL_TO_KJ),
}

PD_AH_EPSILON_KJ_PER_MOL = 0.2 * KCAL_TO_KJ
PD_AH_CUTOFF_NM = 2.0
DH_CUTOFF_NM = 4.0

# Source comparator constants; they are not used by the formal C3-DH builder.
KAPOOR_DH_COULOMB_KJ_MOL_NM = 1.7366932033
KAPOOR_DH_KAPPA_NM_INV = 1.0
KAPOOR_DH_CUTOFF_NM = 3.5

DEFAULT_FORCE_GROUPS: Mapping[str, int] = {
    "dna_bonds": 10,
    "dna_angles": 11,
    "dna_short_range": 12,
    "dd_dh": 13,
    "pd_ah": 14,
    "pd_dh": 15,
}

PROJECT_DNA_EXTRA_EXCLUSION_COUNT = 4_506 + 2_250 + 2_244


class PairOwnership(str, Enum):
    """Mutually exclusive short-range ownership classes for DNA pairs."""

    EXCLUDED_12 = "excluded_1_2"
    WCA_BACKBONE = "wca_backbone_or_backbone_base"
    STACKING = "stacking_same_strand_adjacent_base"
    WCA_NEXT_NEAREST = "wca_same_strand_i_i_plus_2_base"
    HYDROGEN_BOND = "dynamic_watson_crick_hb"
    WCA_BASE_GENERIC = "wca_other_noncomplementary_base"


@dataclass(frozen=True)
class DNASite:
    """Metadata for one appended DNA particle."""

    particle_index: int
    dna_id: int
    strand_id: int
    strand_role: str
    nucleotide_index: int
    base_identity: str
    bead_kind: str
    native_partner: Optional[int]
    mass_da: float
    charge_e: float
    lambda_dna: float
    sigma_pd_nm: float = DNA_PD_SIGMA_NM

    @property
    def is_base(self) -> bool:
        return self.bead_kind == "base"

    @property
    def is_backbone(self) -> bool:
        return self.bead_kind == "backbone"


@dataclass(frozen=True)
class Kapoor2BManifest:
    """Topology, coordinate-order, and force handles returned by the builder."""

    protein_particle_count: int
    total_particle_count: int
    n_duplexes: int
    strand_sequences: Tuple[str, ...]
    dna_positions_nm: np.ndarray
    site_records: Tuple[DNASite, ...]
    dna_indices: Tuple[int, ...]
    backbone_indices: Tuple[int, ...]
    base_indices: Tuple[int, ...]
    bonds: Tuple[Tuple[int, int, str], ...]
    angles: Tuple[Tuple[int, int, int], ...]
    exclusions_12: Tuple[Tuple[int, int], ...]
    protein_nonbonded_exclusions: Tuple[Tuple[int, int], ...]
    global_nonbonded_exclusions: Tuple[Tuple[int, int], ...]
    stacking_pairs: Tuple[Tuple[int, int], ...]
    next_nearest_base_pairs: Tuple[Tuple[int, int], ...]
    native_base_pairs: Tuple[Tuple[int, int], ...]
    forces: Mapping[str, object]
    temperature_k: float
    ionic_strength_m: float
    dh_cutoff_nm: float

    @property
    def nucleotide_count(self) -> int:
        return len(self.base_indices)

    @property
    def dna_particle_count(self) -> int:
        return len(self.dna_indices)

    @property
    def total_dna_charge_e(self) -> float:
        return float(sum(site.charge_e for site in self.site_records))

    @property
    def strand_count(self) -> int:
        return len(self.strand_sequences)

    def metadata_dict(self) -> Dict[str, object]:
        """Return JSON-friendly topology and force-field metadata."""

        return {
            "model": "CALVADOS3-Kapoor-2B hybrid adaptation",
            "kapoor_source_commit": "5a93fc279f6a6c498c55ed6992493d6cb6c0d952",
            "calvados_source_commit": "dc5beac158b72262f585947487c1dd8ef1602a18",
            "temperature_k": self.temperature_k,
            "ionic_strength_m": self.ionic_strength_m,
            "dh_cutoff_nm": self.dh_cutoff_nm,
            "protein_particles": self.protein_particle_count,
            "dna_particles": self.dna_particle_count,
            "total_particles": self.total_particle_count,
            "duplexes": self.n_duplexes,
            "strands": self.strand_count,
            "nucleotides": self.nucleotide_count,
            "bonds": len(self.bonds),
            "angles": len(self.angles),
            "fixed_stacking_pairs": len(self.stacking_pairs),
            "fixed_next_nearest_base_wca_pairs": len(self.next_nearest_base_pairs),
            "native_base_pairs_metadata_only": len(self.native_base_pairs),
            "protein_nonbonded_exclusions": len(self.protein_nonbonded_exclusions),
            "global_nonbonded_exclusions": len(self.global_nonbonded_exclusions),
            "dna_total_charge_e": self.total_dna_charge_e,
            "particle_order": "duplex-major; strand A then B; per nucleotide [backbone, base]",
            "strand_b": "reverse complement; native partner A(i)<->B(N-1-i)",
            "electrostatics": (
                f"C3-DH, {self.temperature_k} K, {self.ionic_strength_m} M, "
                f"shifted at {self.dh_cutoff_nm} nm"
            ),
        }


def reverse_complement(sequence: str) -> str:
    """Return an uppercase DNA reverse complement after strict validation."""

    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("DNA sequence is empty.")
    invalid = sorted(set(seq) - set("ATCG"))
    if invalid:
        raise ValueError(f"DNA sequence contains unsupported symbols: {invalid}")
    return seq.translate(str.maketrans("ATCG", "TAGC"))[::-1]


def canonical_strand_sequences(sequence: str, n_duplexes: int = 3) -> Tuple[str, ...]:
    """Return A,B strands for each duplex in canonical particle order."""

    seq = "".join(str(sequence).split()).upper()
    rev = reverse_complement(seq)
    if int(n_duplexes) != n_duplexes or n_duplexes < 1:
        raise ValueError("n_duplexes must be a positive integer.")
    strands = []
    for _ in range(int(n_duplexes)):
        strands.extend((seq, rev))
    return tuple(strands)


def expected_topology_counts(sequence_length: int, n_duplexes: int, protein_particles: int = 0) -> Dict[str, int]:
    """Return exact count invariants for the canonical two-bead topology."""

    n = int(sequence_length)
    d = int(n_duplexes)
    if n < 1 or d < 1:
        raise ValueError("sequence_length and n_duplexes must be positive.")
    strands = 2 * d
    nucleotides = strands * n
    dna_particles = 2 * nucleotides
    bonds = strands * (n + max(n - 1, 0))
    angles = strands * max(n - 2, 0)
    stacking = strands * max(n - 1, 0)
    next_nearest = strands * max(n - 2, 0)
    return {
        "protein_particles": int(protein_particles),
        "duplexes": d,
        "strands": strands,
        "nucleotides": nucleotides,
        "dna_particles": dna_particles,
        "total_particles": int(protein_particles) + dna_particles,
        "bonds": bonds,
        "angles": angles,
        "stacking_pairs": stacking,
        "next_nearest_base_pairs": next_nearest,
        "dna_total_charge_e": -nucleotides,
    }


def assert_project_topology(
    manifest: Kapoor2BManifest,
    *,
    expected_protein_exclusion_count: int,
) -> None:
    """Hard-stop unless the manifest matches the frozen MeCP2 production counts."""

    if (
        type(expected_protein_exclusion_count) is not int
        or expected_protein_exclusion_count <= 0
    ):
        raise ValueError(
            "expected_protein_exclusion_count must be an explicit positive integer"
        )
    expected_global_exclusion_count = (
        expected_protein_exclusion_count + PROJECT_DNA_EXTRA_EXCLUSION_COUNT
    )

    expected = expected_topology_counts(376, 3, 72_900)
    actual = {
        "protein_particles": manifest.protein_particle_count,
        "duplexes": manifest.n_duplexes,
        "strands": manifest.strand_count,
        "nucleotides": manifest.nucleotide_count,
        "dna_particles": manifest.dna_particle_count,
        "total_particles": manifest.total_particle_count,
        "bonds": len(manifest.bonds),
        "angles": len(manifest.angles),
        "stacking_pairs": len(manifest.stacking_pairs),
        "next_nearest_base_pairs": len(manifest.next_nearest_base_pairs),
        "dna_total_charge_e": int(round(manifest.total_dna_charge_e)),
    }
    mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise AssertionError(f"Frozen MeCP2 topology/count mismatch: {mismatches}")
    if (
        len(manifest.protein_nonbonded_exclusions)
        != expected_protein_exclusion_count
    ):
        raise AssertionError(
            f"Frozen PP exclusion count mismatch: {len(manifest.protein_nonbonded_exclusions)} "
            f"!= {expected_protein_exclusion_count}"
        )
    if (
        len(manifest.global_nonbonded_exclusions)
        != expected_global_exclusion_count
    ):
        raise AssertionError(
            f"Global exclusion count mismatch: {len(manifest.global_nonbonded_exclusions)} "
            f"!= {expected_global_exclusion_count}"
        )


def c3_dh_parameters(temperature_k: float = 310.0, ionic_strength_m: float = 0.100) -> Tuple[float, float]:
    """Reproduce frozen CALVADOS3 ``genParamsDH`` (epsilon, kappa)."""

    temperature_k = float(temperature_k)
    ionic_strength_m = float(ionic_strength_m)
    if temperature_k <= 0.0 or ionic_strength_m < 0.0:
        raise ValueError("temperature must be positive and ionic strength nonnegative.")
    kt = 8.3145 * temperature_k * 1.0e-3
    epsw = (
        5321.0 / temperature_k
        + 233.76
        - 0.9297 * temperature_k
        + 0.1417e-2 * temperature_k**2
        - 0.8292e-6 * temperature_k**3
    )
    bjerrum_nm = 1.6021766**2 / (4.0 * math.pi * 8.854188 * epsw) * 6.02214076 * 1000.0 / kt
    epsilon_yu = bjerrum_nm * kt
    kappa_nm_inv = math.sqrt(8.0 * math.pi * bjerrum_nm * ionic_strength_m * 6.02214076 / 10.0)
    return epsilon_yu, kappa_nm_inv


def source_bond_energy_force(delta_r_nm: float) -> Tuple[float, float]:
    """Return source-convention bond energy and |force| for displacement dr."""

    dr = float(delta_r_nm)
    energy = SOURCE_BOND_K_KCAL_PER_A2 * (dr / ANGSTROM_TO_NM) ** 2 * KCAL_TO_KJ
    force = 2.0 * SOURCE_BOND_K_KCAL_PER_A2 * (dr / ANGSTROM_TO_NM) * KCAL_TO_KJ / ANGSTROM_TO_NM
    return energy, abs(force)


def source_angle_energy(theta_rad: float) -> float:
    return ANGLE_K_KJ_PER_MOL * (math.cos(float(theta_rad)) - math.cos(ANGLE_THETA0_RAD)) ** 2


def wca_energy(r_nm: float, epsilon_kj: float = WCA_EPSILON_KJ_PER_MOL, sigma_nm: float = WCA_SIGMA_NM) -> float:
    r = float(r_nm)
    if r <= 0.0:
        return math.inf
    cutoff = (2.0 ** (1.0 / 6.0)) * float(sigma_nm)
    if r >= cutoff:
        return 0.0
    sr6 = (float(sigma_nm) / r) ** 6
    return 4.0 * float(epsilon_kj) * (sr6 * sr6 - sr6) + float(epsilon_kj)


def twelve_ten_energy(r_nm: float, sigma_nm: float, delta_kj: float, cutoff_nm: Optional[float] = None) -> float:
    r = float(r_nm)
    if r <= 0.0:
        return math.inf
    if cutoff_nm is not None and r >= float(cutoff_nm):
        return 0.0
    sr10 = (float(sigma_nm) / r) ** 10
    return float(delta_kj) * (5.0 * sr10 * (float(sigma_nm) / r) ** 2 - 6.0 * sr10)


def mix_pd_parameters(protein_sigma_nm: float, protein_lambda: float, dna_site_type: str) -> Tuple[float, float]:
    """Arithmetic sigma/lambda mixing for one C3 residue and one DNA site."""

    site = str(dna_site_type).upper()
    if site not in DNA_LAMBDA:
        raise ValueError(f"Unknown DNA site type {dna_site_type!r}.")
    return 0.5 * (float(protein_sigma_nm) + DNA_PD_SIGMA_NM), 0.5 * (float(protein_lambda) + DNA_LAMBDA[site])


def c3_shifted_ah_energy(
    r_nm: float,
    sigma_nm: float,
    lambda_mixed: float,
    epsilon_kj: float = PD_AH_EPSILON_KJ_PER_MOL,
    cutoff_nm: float = PD_AH_CUTOFF_NM,
) -> float:
    """Frozen C3 energy-shifted Ashbaugh-Hatch pair energy."""

    r = float(r_nm)
    sigma = float(sigma_nm)
    lam = float(lambda_mixed)
    rc = float(cutoff_nm)
    if r <= 0.0:
        return math.inf
    if r >= rc:
        return 0.0
    lj = (sigma / r) ** 12 - (sigma / r) ** 6
    shift = (sigma / rc) ** 12 - (sigma / rc) ** 6
    if r >= (2.0 ** (1.0 / 6.0)) * sigma:
        return 4.0 * float(epsilon_kj) * lam * (lj - shift)
    return float(epsilon_kj) * (4.0 * (lj - lam * shift) + (1.0 - lam))


def shifted_dh_energy(
    r_nm: float,
    charge_product: float,
    epsilon_yu_kj_mol_nm: float,
    kappa_nm_inv: float,
    cutoff_nm: float = DH_CUTOFF_NM,
) -> float:
    r = float(r_nm)
    rc = float(cutoff_nm)
    if r <= 0.0:
        return math.copysign(math.inf, float(charge_product))
    if r >= rc:
        return 0.0
    return float(charge_product) * float(epsilon_yu_kj_mol_nm) * (
        math.exp(-float(kappa_nm_inv) * r) / r
        - math.exp(-float(kappa_nm_inv) * rc) / rc
    )


def kapoor_source_dh_energy(r_nm: float, charge_product: float = 1.0) -> float:
    """Unshifted epsilon_r=80 source-comparator Debye-Huckel energy."""

    r = float(r_nm)
    if r <= 0.0:
        return math.copysign(math.inf, float(charge_product))
    if r >= KAPOOR_DH_CUTOFF_NM:
        return 0.0
    return float(charge_product) * KAPOOR_DH_COULOMB_KJ_MOL_NM * math.exp(-KAPOOR_DH_KAPPA_NM_INV * r) / r


def kapoor_source_dh_force_magnitude(r_nm: float, charge_product: float = 1.0) -> float:
    r = float(r_nm)
    if r <= 0.0 or r >= KAPOOR_DH_CUTOFF_NM:
        return 0.0 if r >= KAPOOR_DH_CUTOFF_NM else math.inf
    prefactor = abs(float(charge_product)) * KAPOOR_DH_COULOMB_KJ_MOL_NM
    return prefactor * math.exp(-KAPOOR_DH_KAPPA_NM_INV * r) * (
        KAPOOR_DH_KAPPA_NM_INV / r + 1.0 / (r * r)
    )


def _canonical_pair(a: int, b: int) -> Tuple[int, int]:
    return (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))


def _ordered_exclusions(force: object) -> Tuple[Tuple[int, int], ...]:
    """Return a CustomNonbondedForce exclusion list without reordering it."""

    return tuple(
        (int(force.getExclusionParticles(i)[0]), int(force.getExclusionParticles(i)[1]))
        for i in range(int(force.getNumExclusions()))
    )


def _validate_ordered_exclusion_list(
    exclusions: Sequence[Tuple[int, int]], total_particle_count: int
) -> Tuple[Tuple[int, int], ...]:
    ordered = tuple((int(a), int(b)) for a, b in exclusions)
    canonical = tuple(_canonical_pair(a, b) for a, b in ordered)
    if any(a == b or a < 0 or b >= int(total_particle_count) for a, b in canonical):
        raise ValueError("Nonbonded exclusions contain an invalid particle pair.")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Nonbonded exclusions contain duplicate particle pairs.")
    return ordered


def _add_ordered_exclusions(
    force: object, exclusions: Sequence[Tuple[int, int]], total_particle_count: int
) -> None:
    for a, b in _validate_ordered_exclusion_list(exclusions, total_particle_count):
        force.addExclusion(int(a), int(b))


def are_complementary(base_a: str, base_b: str) -> bool:
    return (str(base_a).upper(), str(base_b).upper()) in {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def pair_ownership(site_a: DNASite, site_b: DNASite, covalent_bonds: Iterable[Tuple[int, int]] = ()) -> PairOwnership:
    """Pure-Python reference for the mutually exclusive Kapoor pair routing."""

    if site_a.particle_index == site_b.particle_index:
        raise ValueError("A particle cannot own an interaction with itself.")
    bond_set = {_canonical_pair(*pair) for pair in covalent_bonds}
    if _canonical_pair(site_a.particle_index, site_b.particle_index) in bond_set:
        return PairOwnership.EXCLUDED_12
    if site_a.is_backbone or site_b.is_backbone:
        return PairOwnership.WCA_BACKBONE
    same_strand = site_a.strand_id == site_b.strand_id
    separation = abs(site_a.nucleotide_index - site_b.nucleotide_index)
    if same_strand and separation == 1:
        return PairOwnership.STACKING
    if same_strand and separation == 2:
        return PairOwnership.WCA_NEXT_NEAREST
    if are_complementary(site_a.base_identity, site_b.base_identity):
        return PairOwnership.HYDROGEN_BOND
    return PairOwnership.WCA_BASE_GENERIC


def _to_float(value: object, target_unit: Optional[object] = None) -> float:
    if target_unit is not None and hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(target_unit))
    return float(value)


def _normalize_coordinates(
    coordinates_nm: object,
    n_strands: int,
    sequence_length: int,
) -> np.ndarray:
    if hasattr(coordinates_nm, "value_in_unit"):
        coordinates_nm = coordinates_nm.value_in_unit(unit.nanometer)
    arr = np.asarray(coordinates_nm, dtype=float)
    expected = n_strands * sequence_length * 2
    accepted = {
        (n_strands, sequence_length, 2, 3),
        (n_strands, sequence_length * 2, 3),
        (expected, 3),
    }
    if arr.shape not in accepted:
        raise ValueError(
            "dna_coordinates_nm must have canonical shape "
            f"({n_strands},{sequence_length},2,3), ({n_strands},{2*sequence_length},3), "
            f"or ({expected},3); got {arr.shape}."
        )
    flat = np.asarray(arr.reshape(expected, 3), dtype=float)
    if not np.isfinite(flat).all():
        raise ValueError("DNA coordinates contain NaN or infinite values.")
    return flat


def _validate_force_groups(force_groups: Optional[Mapping[str, int]]) -> Dict[str, int]:
    groups = dict(DEFAULT_FORCE_GROUPS if force_groups is None else force_groups)
    if set(groups) != set(DEFAULT_FORCE_GROUPS):
        raise ValueError(f"force_groups must define exactly {sorted(DEFAULT_FORCE_GROUPS)}.")
    values = [int(groups[key]) for key in DEFAULT_FORCE_GROUPS]
    if len(set(values)) != len(values) or any(value < 0 or value > 31 for value in values):
        raise ValueError("New force groups must be unique integers in [0,31].")
    return {key: int(groups[key]) for key in DEFAULT_FORCE_GROUPS}


def _parameter_names(force: object) -> Tuple[str, ...]:
    return tuple(force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters()))


def _validate_frozen_pp_inputs(
    system: object,
    protein_indices: Sequence[int],
    protein_charges_e: Sequence[float],
    protein_sigmas_nm: Sequence[float],
    protein_lambdas: Sequence[float],
    protein_ah_force: object,
    protein_dh_force: object,
) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    n_old = int(system.getNumParticles())
    indices = tuple(int(i) for i in protein_indices)
    if tuple(sorted(indices)) != tuple(range(n_old)):
        raise ValueError("For this frozen-project adapter, protein_indices must cover every pre-existing particle exactly once.")
    charges = np.asarray(protein_charges_e, dtype=float)
    sigmas = np.asarray(protein_sigmas_nm, dtype=float)
    lambdas = np.asarray(protein_lambdas, dtype=float)
    if any(arr.shape != (n_old,) for arr in (charges, sigmas, lambdas)):
        raise ValueError("protein charges/sigmas/lambdas must each have one value per pre-existing protein particle.")
    if not all(np.isfinite(arr).all() for arr in (charges, sigmas, lambdas)):
        raise ValueError("Protein nonbonded parameters contain NaN or infinite values.")
    if int(protein_ah_force.getNumParticles()) != n_old or int(protein_dh_force.getNumParticles()) != n_old:
        raise ValueError("Frozen PP AH/DH force particle counts do not match the existing System.")
    if _parameter_names(protein_ah_force) != ("s", "l", "id"):
        raise ValueError("Frozen C3 AH force must expose per-particle parameters exactly (s,l,id).")
    if _parameter_names(protein_dh_force) != ("q",):
        raise ValueError("Frozen C3 DH force must expose per-particle parameter q.")
    for local, particle in enumerate(indices):
        ah_params = protein_ah_force.getParticleParameters(particle)
        dh_params = protein_dh_force.getParticleParameters(particle)
        sigma_existing = _to_float(ah_params[0], unit.nanometer)
        lambda_existing = _to_float(ah_params[1])
        id_existing = _to_float(ah_params[2])
        charge_existing = _to_float(dh_params[0])
        if not math.isclose(id_existing, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Pre-existing particle {particle} is not a frozen protein AH id=1 bead.")
        if not math.isclose(sigma_existing, sigmas[local], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Protein sigma mismatch at particle {particle}.")
        if not math.isclose(lambda_existing, lambdas[local], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Protein lambda mismatch at particle {particle}.")
        if not math.isclose(charge_existing, charges[local], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Protein charge mismatch at particle {particle}.")
    return indices, charges, sigmas, lambdas


def _ensure_c3_ah_dummy_mask(protein_ah_force: object) -> None:
    """Make id=0 dummy pairs exactly zero while leaving id=1 PP algebra intact."""

    expression = str(protein_ah_force.getEnergyFunction())
    compact = expression.replace(" ", "")
    if "pp_dummy_mask=id1*id2" in compact or "mask=id1*id2" in compact:
        return
    if ";" not in expression:
        raise ValueError("Unexpected frozen C3 AH expression: missing intermediate definitions.")
    energy_term, remainder = expression.split(";", 1)
    protein_ah_force.setEnergyFunction(
        f"pp_dummy_mask*({energy_term});pp_dummy_mask=id1*id2;{remainder}"
    )


def create_dna_bond_force(
    bonds: Sequence[Tuple[int, int, str]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_bonds"],
) -> object:
    force = openmm.HarmonicBondForce()
    force.setName("Kapoor2B_DNA_Bonds")
    force.setForceGroup(int(force_group))
    force.setUsesPeriodicBoundaryConditions(True)
    for a, b, kind in bonds:
        if kind == "BB-BB":
            r0 = BB_BB_R0_NM
        elif kind == "BB-base":
            r0 = BB_BASE_R0_NM
        else:
            raise ValueError(f"Unknown DNA bond kind {kind!r}.")
        force.addBond(
            int(a),
            int(b),
            r0 * unit.nanometer,
            OPENMM_BOND_K_KJ_PER_NM2 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
    return force


def create_dna_angle_force(
    angles: Sequence[Tuple[int, int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_angles"],
) -> object:
    force = openmm.CustomAngleForce("kangle*(cos(theta)-cos(theta0))^2")
    force.setName("Kapoor2B_DNA_BackboneAngles")
    force.setForceGroup(int(force_group))
    force.addGlobalParameter("kangle", ANGLE_K_KJ_PER_MOL)
    force.addGlobalParameter("theta0", ANGLE_THETA0_RAD)
    force.setUsesPeriodicBoundaryConditions(True)
    for a, b, c in angles:
        force.addAngle(int(a), int(b), int(c), [])
    return force


def _stack_pair_key(base_a: str, base_b: str) -> str:
    a, b = str(base_a).upper(), str(base_b).upper()
    for key in STACK_DELTA_KJ_PER_MOL:
        if (a == key[0] and b == key[1]) or (a == key[1] and b == key[0]):
            return key
    raise ValueError(f"No stacking parameter for bases {a!r}, {b!r}.")


def create_dna_stacking_force(
    site_records: Sequence[DNASite],
    stacking_pairs: Sequence[Tuple[int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_short_range"],
) -> object:
    force = openmm.CustomBondForce(
        "dstack*(5*(stackSigma/r)^12-6*(stackSigma/r)^10)*step(stackCut-r)"
    )
    force.setName("Kapoor2B_DNA_Stacking")
    force.setForceGroup(int(force_group))
    force.addPerBondParameter("dstack")
    force.addGlobalParameter("stackSigma", STACK_SIGMA_NM)
    force.addGlobalParameter("stackCut", STACK_CUTOFF_NM)
    force.setUsesPeriodicBoundaryConditions(True)
    site_map = {site.particle_index: site for site in site_records}
    for a, b in stacking_pairs:
        sa, sb = site_map[int(a)], site_map[int(b)]
        if not (sa.is_base and sb.is_base):
            raise ValueError("Stacking pairs must contain base beads only.")
        key = _stack_pair_key(sa.base_identity, sb.base_identity)
        force.addBond(int(a), int(b), [STACK_DELTA_KJ_PER_MOL[key]])
    return force


def create_dna_next_nearest_wca_force(
    next_nearest_pairs: Sequence[Tuple[int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_short_range"],
) -> object:
    force = openmm.CustomBondForce(
        "(4*wcaEps*((wcaSigma/r)^12-(wcaSigma/r)^6)+wcaEps)*step(wcaCut-r)"
    )
    force.setName("Kapoor2B_DNA_NextNearestBase_WCA")
    force.setForceGroup(int(force_group))
    force.addGlobalParameter("wcaEps", WCA_EPSILON_KJ_PER_MOL)
    force.addGlobalParameter("wcaSigma", WCA_SIGMA_NM)
    force.addGlobalParameter("wcaCut", WCA_CUTOFF_NM)
    force.setUsesPeriodicBoundaryConditions(True)
    for a, b in next_nearest_pairs:
        force.addBond(int(a), int(b), [])
    return force


def _base_generic_expression() -> str:
    # Stacking and same-strand i/i+2 pairs are explicit exclusions from this
    # force.  Every remaining base pair is therefore routed dynamically to
    # Watson-Crick HB (AT/CG) or generic WCA, never both.
    at = "(isA1*isT2+isT1*isA2)"
    cg = "(isC1*isG2+isG1*isC2)"
    return (
        f"{at}*{HB_PARAMETERS['AT'][1]}*(5*({HB_PARAMETERS['AT'][0]}/r)^12-6*({HB_PARAMETERS['AT'][0]}/r)^10)*step({HB_CUTOFF_NM}-r)+"
        f"{cg}*{HB_PARAMETERS['CG'][1]}*(5*({HB_PARAMETERS['CG'][0]}/r)^12-6*({HB_PARAMETERS['CG'][0]}/r)^10)*step({HB_CUTOFF_NM}-r)+"
        f"(1-{at}-{cg})*(4*{WCA_EPSILON_KJ_PER_MOL}*(({WCA_SIGMA_NM}/r)^12-({WCA_SIGMA_NM}/r)^6)+{WCA_EPSILON_KJ_PER_MOL})*step({WCA_CUTOFF_NM}-r)"
    )


def create_dna_base_nonbonded_force(
    total_particle_count: int,
    site_records: Sequence[DNASite],
    exclusions: Sequence[Tuple[int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_short_range"],
) -> object:
    force = openmm.CustomNonbondedForce(_base_generic_expression())
    force.setName("Kapoor2B_DNA_Base_DynamicHB_or_WCA")
    force.setForceGroup(int(force_group))
    for name in ("isA", "isT", "isC", "isG"):
        force.addPerParticleParameter(name)
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(HB_CUTOFF_NM * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    site_map = {site.particle_index: site for site in site_records}
    base_indices = []
    for particle in range(int(total_particle_count)):
        site = site_map.get(particle)
        if site is None or not site.is_base:
            params = [0.0, 0.0, 0.0, 0.0]
        else:
            base_indices.append(particle)
            params = [float(site.base_identity == base) for base in "ATCG"]
        force.addParticle(params)
    force.addInteractionGroup(set(base_indices), set(base_indices))
    _add_ordered_exclusions(force, exclusions, total_particle_count)
    return force


def create_dna_backbone_wca_force(
    total_particle_count: int,
    site_records: Sequence[DNASite],
    exclusions: Sequence[Tuple[int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_short_range"],
) -> object:
    expression = (
        f"(4*{WCA_EPSILON_KJ_PER_MOL}*(({WCA_SIGMA_NM}/r)^12-({WCA_SIGMA_NM}/r)^6)+"
        f"{WCA_EPSILON_KJ_PER_MOL})*step({WCA_CUTOFF_NM}-r)"
    )
    force = openmm.CustomNonbondedForce(expression)
    force.setName("Kapoor2B_DNA_BackboneContaining_WCA")
    force.setForceGroup(int(force_group))
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(WCA_CUTOFF_NM * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    for _ in range(int(total_particle_count)):
        force.addParticle([])
    dna_indices = [site.particle_index for site in site_records]
    backbone_indices = [site.particle_index for site in site_records if site.is_backbone]
    force.addInteractionGroup(set(backbone_indices), set(dna_indices))
    _add_ordered_exclusions(force, exclusions, total_particle_count)
    return force


def create_dd_short_range_forces(
    total_particle_count: int,
    site_records: Sequence[DNASite],
    global_exclusions: Sequence[Tuple[int, int]],
    stacking_pairs: Sequence[Tuple[int, int]],
    next_nearest_pairs: Sequence[Tuple[int, int]],
    force_group: int = DEFAULT_FORCE_GROUPS["dna_short_range"],
) -> Dict[str, object]:
    """Build a stable, exactly partitioned set of Kapoor short-range forces."""

    return {
        "dna_stacking": create_dna_stacking_force(site_records, stacking_pairs, force_group),
        "dna_next_nearest_wca": create_dna_next_nearest_wca_force(next_nearest_pairs, force_group),
        "dna_base_nonbonded": create_dna_base_nonbonded_force(
            total_particle_count, site_records, global_exclusions, force_group
        ),
        "dna_backbone_wca": create_dna_backbone_wca_force(
            total_particle_count, site_records, global_exclusions, force_group
        ),
    }


def create_shifted_dh_force(
    total_particle_count: int,
    charges_by_particle: Mapping[int, float],
    group_a: Sequence[int],
    group_b: Sequence[int],
    exclusions: Sequence[Tuple[int, int]],
    epsilon_yu_kj_mol_nm: float,
    kappa_nm_inv: float,
    cutoff_nm: float,
    name: str,
    force_group: int,
) -> object:
    expression = "q1*q2*epsYu*(exp(-kappa*r)/r-dhShift)"
    force = openmm.CustomNonbondedForce(expression)
    force.setName(str(name))
    force.setForceGroup(int(force_group))
    force.addPerParticleParameter("q")
    force.addGlobalParameter("epsYu", float(epsilon_yu_kj_mol_nm))
    force.addGlobalParameter("kappa", float(kappa_nm_inv))
    force.addGlobalParameter(
        "dhShift", math.exp(-float(kappa_nm_inv) * float(cutoff_nm)) / float(cutoff_nm)
    )
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(float(cutoff_nm) * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    for particle in range(int(total_particle_count)):
        force.addParticle([float(charges_by_particle.get(particle, 0.0))])
    force.addInteractionGroup(set(map(int, group_a)), set(map(int, group_b)))
    _add_ordered_exclusions(force, exclusions, total_particle_count)
    return force


def create_pd_ah_force(
    total_particle_count: int,
    protein_indices: Sequence[int],
    protein_sigmas_nm: Sequence[float],
    protein_lambdas: Sequence[float],
    site_records: Sequence[DNASite],
    exclusions: Sequence[Tuple[int, int]] = (),
    epsilon_kj: float = PD_AH_EPSILON_KJ_PER_MOL,
    cutoff_nm: float = PD_AH_CUTOFF_NM,
    force_group: int = DEFAULT_FORCE_GROUPS["pd_ah"],
) -> object:
    rc = float(cutoff_nm)
    eps = float(epsilon_kj)
    expression = (
        f"{eps}*select(step(r-2^(1/6)*s),"
        "4*l*((s/r)^12-(s/r)^6-shift),"
        "4*((s/r)^12-(s/r)^6-l*shift)+(1-l));"
        "l=0.5*(l1+l2);"
        f"shift=(s/{rc})^12-(s/{rc})^6;"
        "s=0.5*(s1+s2)"
    )
    force = openmm.CustomNonbondedForce(expression)
    force.setName("CALVADOS3_Kapoor2B_Protein_DNA_AH")
    force.setForceGroup(int(force_group))
    force.addPerParticleParameter("s")
    force.addPerParticleParameter("l")
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(rc * unit.nanometer)
    force.setUseLongRangeCorrection(False)
    parameters: Dict[int, Tuple[float, float]] = {}
    for idx, sigma, lam in zip(protein_indices, protein_sigmas_nm, protein_lambdas):
        parameters[int(idx)] = (float(sigma), float(lam))
    for site in site_records:
        dna_type = "BB" if site.is_backbone else site.base_identity
        parameters[site.particle_index] = (DNA_PD_SIGMA_NM, DNA_LAMBDA[dna_type])
    for particle in range(int(total_particle_count)):
        sigma, lam = parameters.get(particle, (0.0, 0.0))
        force.addParticle([sigma * unit.nanometer, lam])
    dna_indices = [site.particle_index for site in site_records]
    force.addInteractionGroup(set(map(int, protein_indices)), set(dna_indices))
    # Global exclusions contain no protein-DNA cross pair, so none
    # suppresses an interaction owned by this PD force.
    _add_ordered_exclusions(force, exclusions, total_particle_count)
    return force


def _build_topology_records(
    first_dna_particle: int,
    strand_sequences: Sequence[str],
) -> Tuple[
    Tuple[DNASite, ...],
    Tuple[Tuple[int, int, str], ...],
    Tuple[Tuple[int, int, int], ...],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
    Tuple[Tuple[int, int], ...],
]:
    n = len(strand_sequences[0])
    records = []
    bonds = []
    angles = []
    stacking = []
    next_nearest = []
    strand_base_indices: Dict[int, list[int]] = {}
    strand_bb_indices: Dict[int, list[int]] = {}
    cursor = int(first_dna_particle)
    for strand_id, sequence in enumerate(strand_sequences):
        dna_id = strand_id // 2
        role = "A" if strand_id % 2 == 0 else "B"
        bb_indices = []
        base_indices = []
        for nt_index, base in enumerate(sequence):
            bb_index = cursor
            base_index = cursor + 1
            cursor += 2
            bb_indices.append(bb_index)
            base_indices.append(base_index)
            records.append(
                DNASite(
                    particle_index=bb_index,
                    dna_id=dna_id,
                    strand_id=strand_id,
                    strand_role=role,
                    nucleotide_index=nt_index,
                    base_identity=base,
                    bead_kind="backbone",
                    native_partner=None,
                    mass_da=DNA_MASS_DA["BB"],
                    charge_e=DNA_CHARGE_E["BB"],
                    lambda_dna=DNA_LAMBDA["BB"],
                )
            )
            records.append(
                DNASite(
                    particle_index=base_index,
                    dna_id=dna_id,
                    strand_id=strand_id,
                    strand_role=role,
                    nucleotide_index=nt_index,
                    base_identity=base,
                    bead_kind="base",
                    native_partner=None,
                    mass_da=DNA_MASS_DA[base],
                    charge_e=DNA_CHARGE_E[base],
                    lambda_dna=DNA_LAMBDA[base],
                )
            )
            bonds.append((bb_index, base_index, "BB-base"))
            if nt_index:
                bonds.append((bb_indices[nt_index - 1], bb_index, "BB-BB"))
                stacking.append(_canonical_pair(base_indices[nt_index - 1], base_index))
            if nt_index >= 2:
                angles.append((bb_indices[nt_index - 2], bb_indices[nt_index - 1], bb_index))
                next_nearest.append(_canonical_pair(base_indices[nt_index - 2], base_index))
        strand_base_indices[strand_id] = base_indices
        strand_bb_indices[strand_id] = bb_indices

    native_pairs = []
    partner_by_base: Dict[int, int] = {}
    for dna_id in range(len(strand_sequences) // 2):
        strand_a = 2 * dna_id
        strand_b = strand_a + 1
        for i in range(n):
            a = strand_base_indices[strand_a][i]
            b = strand_base_indices[strand_b][n - 1 - i]
            pair = _canonical_pair(a, b)
            native_pairs.append(pair)
            partner_by_base[a] = b
            partner_by_base[b] = a
    finalized = tuple(
        replace(site, native_partner=partner_by_base.get(site.particle_index))
        if site.is_base
        else site
        for site in records
    )
    return (
        finalized,
        tuple(bonds),
        tuple(angles),
        tuple(sorted(stacking)),
        tuple(sorted(next_nearest)),
        tuple(sorted(native_pairs)),
    )


def append_kapoor2b_to_system(
    system: object,
    *,
    protein_indices: Sequence[int],
    protein_charges_e: Sequence[float],
    protein_sigmas_nm: Sequence[float],
    protein_lambdas: Sequence[float],
    protein_ah_force: object,
    protein_dh_force: object,
    dna_sequence: str,
    dna_coordinates_nm: object,
    n_duplexes: int = 3,
    temperature_k: float = 310.0,
    ionic_strength_m: float = 0.100,
    pd_ah_epsilon_kj: float = PD_AH_EPSILON_KJ_PER_MOL,
    pd_ah_cutoff_nm: float = PD_AH_CUTOFF_NM,
    dh_cutoff_nm: float = DH_CUTOFF_NM,
    force_groups: Optional[Mapping[str, int]] = None,
    strict_project_counts: bool = False,
    expected_protein_exclusion_count: Optional[int] = None,
) -> Kapoor2BManifest:
    """Append three (by default) Kapoor 2B duplexes and hybrid PD/DD forces.

    The input ``System`` must contain only the already-built frozen protein
    particles.  ``protein_ah_force`` and ``protein_dh_force`` must be the exact
    C3 forces already present in that System.  The function extends them with
    AH ``(s,l,id)=(0,0,0)`` and DH ``q=0`` dummy entries; it never adds a PP
    interaction group.  A multiplicative id mask is added once to the frozen
    AH expression so every pair containing a dummy is exactly zero while all
    id=1 protein-protein energies and forces are unchanged.

    ``dna_coordinates_nm`` is returned in the manifest but is not installed in
    a Context (Systems do not store positions).  The caller must concatenate
    existing protein positions with ``manifest.dna_positions_nm`` in exactly
    this order.
    """

    strands = canonical_strand_sequences(dna_sequence, n_duplexes=n_duplexes)
    n = len(strands[0])
    coords = _normalize_coordinates(dna_coordinates_nm, len(strands), n)
    groups = _validate_force_groups(force_groups)
    indices, protein_charges, protein_sigmas, protein_lams = _validate_frozen_pp_inputs(
        system,
        protein_indices,
        protein_charges_e,
        protein_sigmas_nm,
        protein_lambdas,
        protein_ah_force,
        protein_dh_force,
    )
    first_dna = int(system.getNumParticles())
    protein_ah_exclusions = _ordered_exclusions(protein_ah_force)
    protein_dh_exclusions = _ordered_exclusions(protein_dh_force)
    if protein_ah_exclusions != protein_dh_exclusions:
        raise ValueError(
            "Frozen PP AH and DH exclusion lists must be identical in count, order, and particle orientation."
        )
    _validate_ordered_exclusion_list(protein_ah_exclusions, first_dna)
    if strict_project_counts and expected_protein_exclusion_count is None:
        raise ValueError(
            "strict_project_counts requires an explicit "
            "expected_protein_exclusion_count"
        )
    if expected_protein_exclusion_count is not None and (
        type(expected_protein_exclusion_count) is not int
        or expected_protein_exclusion_count <= 0
    ):
        raise ValueError(
            "expected_protein_exclusion_count must be an explicit positive integer"
        )
    if (
        expected_protein_exclusion_count is not None
        and len(protein_ah_exclusions) != expected_protein_exclusion_count
    ):
        raise ValueError(
            "Frozen PP System must contain the construct-bound number of AH/DH "
            f"exclusions ({expected_protein_exclusion_count}); "
            f"found {len(protein_ah_exclusions)}."
        )
    records, bonds, angles, stacking, next_nearest, native_pairs = _build_topology_records(first_dna, strands)
    if len(records) != len(coords):
        raise AssertionError("Internal coordinate/topology ordering mismatch.")

    dna_indices = tuple(site.particle_index for site in records)
    backbone_indices = tuple(site.particle_index for site in records if site.is_backbone)
    base_indices = tuple(site.particle_index for site in records if site.is_base)
    exclusions = tuple(sorted({_canonical_pair(a, b) for a, b, _ in bonds}))
    dna_extra_exclusions = tuple(
        sorted(set(exclusions).union(stacking).union(next_nearest))
    )
    expected_dna_extra = len(bonds) + len(stacking) + len(next_nearest)
    if len(dna_extra_exclusions) != expected_dna_extra:
        raise AssertionError(
            "DNA covalent, stacking, and i/i+2 exclusion ownership sets must be mutually disjoint."
        )
    global_exclusions = tuple(protein_ah_exclusions) + dna_extra_exclusions
    _validate_ordered_exclusion_list(global_exclusions, first_dna + len(records))
    if any((a < first_dna) != (b < first_dna) for a, b in global_exclusions):
        raise AssertionError("Global exclusions must not contain any protein-DNA cross pair.")

    # All validation above happens before mutation.  From here the append is
    # deterministic and must complete before a Context is constructed.
    _ensure_c3_ah_dummy_mask(protein_ah_force)
    for site in records:
        system.addParticle(site.mass_da * unit.dalton)
        protein_ah_force.addParticle([0.0 * unit.nanometer, 0.0, 0.0])
        protein_dh_force.addParticle([0.0])

    total_particles = int(system.getNumParticles())
    if int(protein_ah_force.getNumParticles()) != total_particles or int(protein_dh_force.getNumParticles()) != total_particles:
        raise AssertionError("Frozen PP dummy extension did not match the appended System particle count.")

    # OpenMM requires every CustomNonbondedForce in one System to carry
    # the same ordered exclusion list.  These appended DNA-only pairs were
    # exactly zero in frozen PP forces because of the AH id mask and DH q=0.
    for a, b in dna_extra_exclusions:
        protein_ah_force.addExclusion(int(a), int(b))
        protein_dh_force.addExclusion(int(a), int(b))
    if _ordered_exclusions(protein_ah_force) != global_exclusions:
        raise AssertionError("Frozen PP AH global exclusion extension failed.")
    if _ordered_exclusions(protein_dh_force) != global_exclusions:
        raise AssertionError("Frozen PP DH global exclusion extension failed.")

    dna_bonds = create_dna_bond_force(bonds, groups["dna_bonds"])
    dna_angles = create_dna_angle_force(angles, groups["dna_angles"])
    dna_short_forces = create_dd_short_range_forces(
        total_particles, records, global_exclusions, stacking, next_nearest, groups["dna_short_range"]
    )

    epsilon_yu, kappa = c3_dh_parameters(temperature_k, ionic_strength_m)
    dna_charges = {site.particle_index: site.charge_e for site in records}
    dd_dh = create_shifted_dh_force(
        total_particles,
        dna_charges,
        backbone_indices,
        backbone_indices,
        global_exclusions,
        epsilon_yu,
        kappa,
        dh_cutoff_nm,
        "CALVADOS3_Kapoor2B_DNA_DNA_DH",
        groups["dd_dh"],
    )
    pd_ah = create_pd_ah_force(
        total_particles,
        indices,
        protein_sigmas,
        protein_lams,
        records,
        exclusions=global_exclusions,
        epsilon_kj=pd_ah_epsilon_kj,
        cutoff_nm=pd_ah_cutoff_nm,
        force_group=groups["pd_ah"],
    )
    pd_charges: Dict[int, float] = {int(i): float(q) for i, q in zip(indices, protein_charges)}
    pd_charges.update(dna_charges)
    pd_dh = create_shifted_dh_force(
        total_particles,
        pd_charges,
        indices,
        backbone_indices,
        global_exclusions,
        epsilon_yu,
        kappa,
        dh_cutoff_nm,
        "CALVADOS3_Kapoor2B_Protein_DNA_DH",
        groups["pd_dh"],
    )
    new_forces = {
        "dna_bonds": dna_bonds,
        "dna_angles": dna_angles,
        **dna_short_forces,
        "dd_dh": dd_dh,
        "pd_ah": pd_ah,
        "pd_dh": pd_dh,
    }
    for force in new_forces.values():
        system.addForce(force)

    manifest = Kapoor2BManifest(
        protein_particle_count=first_dna,
        total_particle_count=total_particles,
        n_duplexes=int(n_duplexes),
        strand_sequences=tuple(strands),
        dna_positions_nm=np.array(coords, dtype=float, copy=True),
        site_records=tuple(records),
        dna_indices=dna_indices,
        backbone_indices=backbone_indices,
        base_indices=base_indices,
        bonds=tuple(bonds),
        angles=tuple(angles),
        exclusions_12=exclusions,
        protein_nonbonded_exclusions=tuple(protein_ah_exclusions),
        global_nonbonded_exclusions=tuple(global_exclusions),
        stacking_pairs=tuple(stacking),
        next_nearest_base_pairs=tuple(next_nearest),
        native_base_pairs=tuple(native_pairs),
        forces=new_forces,
        temperature_k=float(temperature_k),
        ionic_strength_m=float(ionic_strength_m),
        dh_cutoff_nm=float(dh_cutoff_nm),
    )

    # Internal invariants independent of the project-size hard gate.
    expected = expected_topology_counts(n, int(n_duplexes), first_dna)
    observed = {
        "strands": manifest.strand_count,
        "nucleotides": manifest.nucleotide_count,
        "dna_particles": manifest.dna_particle_count,
        "total_particles": manifest.total_particle_count,
        "bonds": len(manifest.bonds),
        "angles": len(manifest.angles),
        "stacking_pairs": len(manifest.stacking_pairs),
        "next_nearest_base_pairs": len(manifest.next_nearest_base_pairs),
        "dna_total_charge_e": int(round(manifest.total_dna_charge_e)),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise AssertionError(f"Kapoor2B topology invariant failed for {key}: {value} != {expected[key]}")
    if len(manifest.exclusions_12) != len(manifest.bonds):
        raise AssertionError("Only and all true 1-2 covalent DNA pairs must be nonbonded exclusions.")
    custom_nonbonded_forces = tuple(
        force for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    )
    if not custom_nonbonded_forces:
        raise AssertionError("System unexpectedly contains no CustomNonbondedForce objects.")
    for force in custom_nonbonded_forces:
        if _ordered_exclusions(force) != global_exclusions:
            raise AssertionError(
                f"CustomNonbondedForce {force.getName()!r} does not carry the identical ordered global exclusions."
            )
    if any((a < first_dna) != (b < first_dna) for a, b in global_exclusions):
        raise AssertionError("Global exclusions must not suppress any protein-DNA interaction.")
    if strict_project_counts:
        assert_project_topology(
            manifest,
            expected_protein_exclusion_count=expected_protein_exclusion_count,
        )
    return manifest


__all__ = [
    "ANGLE_K_KJ_PER_MOL",
    "ANGLE_THETA0_RAD",
    "BB_BASE_R0_NM",
    "BB_BB_R0_NM",
    "DEFAULT_FORCE_GROUPS",
    "DH_CUTOFF_NM",
    "DNA_CHARGE_E",
    "DNA_LAMBDA",
    "DNA_MASS_DA",
    "DNA_PD_SIGMA_NM",
    "DNASite",
    "HB_CUTOFF_NM",
    "HB_PARAMETERS",
    "KAPOOR_DH_COULOMB_KJ_MOL_NM",
    "KAPOOR_DH_CUTOFF_NM",
    "KAPOOR_DH_KAPPA_NM_INV",
    "Kapoor2BManifest",
    "OPENMM_BOND_K_KJ_PER_NM2",
    "PD_AH_CUTOFF_NM",
    "PD_AH_EPSILON_KJ_PER_MOL",
    "PROJECT_DNA_EXTRA_EXCLUSION_COUNT",
    "PairOwnership",
    "STACK_CUTOFF_NM",
    "STACK_DELTA_KJ_PER_MOL",
    "STACK_SIGMA_NM",
    "WCA_CUTOFF_NM",
    "WCA_EPSILON_KJ_PER_MOL",
    "WCA_SIGMA_NM",
    "append_kapoor2b_to_system",
    "are_complementary",
    "assert_project_topology",
    "c3_dh_parameters",
    "c3_shifted_ah_energy",
    "canonical_strand_sequences",
    "create_dd_short_range_forces",
    "create_dna_backbone_wca_force",
    "create_dna_base_nonbonded_force",
    "create_dna_next_nearest_wca_force",
    "create_dna_stacking_force",
    "create_dna_angle_force",
    "create_dna_bond_force",
    "create_pd_ah_force",
    "create_shifted_dh_force",
    "expected_topology_counts",
    "kapoor_source_dh_energy",
    "kapoor_source_dh_force_magnitude",
    "mix_pd_parameters",
    "pair_ownership",
    "reverse_complement",
    "shifted_dh_energy",
    "source_angle_energy",
    "source_bond_energy_force",
    "twelve_ten_energy",
    "wca_energy",
]
