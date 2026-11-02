#!/usr/bin/env python3
"""CPU/Reference unit tests for the CALVADOS3--Kapoor 2B force builder.

Run with ``python -m unittest discover -s tests -v``.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.common.kapoor2b_openmm import (  # noqa: E402
    ANGLE_K_KJ_PER_MOL,
    BB_BB_R0_NM,
    DEFAULT_FORCE_GROUPS,
    DNA_LAMBDA,
    DNA_MASS_DA,
    DNA_PD_SIGMA_NM,
    DNASite,
    HB_CUTOFF_NM,
    HB_PARAMETERS,
    OPENMM_BOND_K_KJ_PER_NM2,
    PD_AH_CUTOFF_NM,
    PROJECT_DNA_EXTRA_EXCLUSION_COUNT,
    PairOwnership,
    STACK_CUTOFF_NM,
    STACK_DELTA_KJ_PER_MOL,
    STACK_SIGMA_NM,
    WCA_CUTOFF_NM,
    WCA_EPSILON_KJ_PER_MOL,
    WCA_SIGMA_NM,
    append_kapoor2b_to_system,
    c3_dh_parameters,
    c3_shifted_ah_energy,
    canonical_strand_sequences,
    create_dd_short_range_forces,
    create_dna_angle_force,
    create_dna_bond_force,
    create_pd_ah_force,
    create_shifted_dh_force,
    expected_topology_counts,
    kapoor_source_dh_energy,
    kapoor_source_dh_force_magnitude,
    mix_pd_parameters,
    pair_ownership,
    reverse_complement,
    shifted_dh_energy,
    source_angle_energy,
    source_bond_energy_force,
    twelve_ten_energy,
    wca_energy,
    openmm,
    unit,
)


ENERGY_RTOL = 1.0e-6
FORCE_RTOL = 1.0e-5
ENERGY_ATOL = 1.0e-8
FORCE_ATOL = 1.0e-7


def assert_close(testcase, actual, expected, *, rtol=ENERGY_RTOL, atol=ENERGY_ATOL, message=""):
    testcase.assertTrue(
        math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
        message or f"{actual!r} != {expected!r} (rtol={rtol}, atol={atol})",
    )


def make_system(n_particles: int, box_nm: float = 10.0):
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(100.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_nm) * unit.nanometer,
    )
    return system


def evaluate(system, positions_nm, groups=None, platform_name="Reference"):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
    kwargs = {"getEnergy": True, "getForces": True}
    if groups is not None:
        kwargs["groups"] = set(groups)
    state = context.getState(**kwargs)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    del context, integrator
    return float(energy), np.asarray(forces, dtype=float)


def ordered_exclusions(force):
    return tuple(
        tuple(map(int, force.getExclusionParticles(i)))
        for i in range(force.getNumExclusions())
    )


def make_site(index, kind, base, strand=0, nt=0, dna_id=0):
    key = "BB" if kind == "backbone" else base
    return DNASite(
        particle_index=index,
        dna_id=dna_id,
        strand_id=strand,
        strand_role="A" if strand % 2 == 0 else "B",
        nucleotide_index=nt,
        base_identity=base,
        bead_kind=kind,
        native_partner=None,
        mass_da=DNA_MASS_DA[key],
        charge_e=-1.0 if kind == "backbone" else 0.0,
        lambda_dna=DNA_LAMBDA[key],
    )


def dd_pair_state(site_a, site_b, positions_nm, exclusions=()):
    system = make_system(2)
    owner = pair_ownership(site_a, site_b, exclusions)
    stacking = ((0, 1),) if owner is PairOwnership.STACKING else ()
    next_nearest = ((0, 1),) if owner is PairOwnership.WCA_NEXT_NEAREST else ()
    global_exclusions = tuple(sorted(set(exclusions).union(stacking).union(next_nearest)))
    forces_by_name = create_dd_short_range_forces(
        2, (site_a, site_b), global_exclusions, stacking, next_nearest, force_group=12
    )
    for force in forces_by_name.values():
        system.addForce(force)
    energy, forces = evaluate(system, positions_nm, groups={12})
    diagnostics = {
        "pbc": all(force.usesPeriodicBoundaryConditions() for force in forces_by_name.values()),
        "exclusions": {
            name: force.getNumExclusions() if hasattr(force, "getNumExclusions") else None
            for name, force in forces_by_name.items()
        },
    }
    return energy, forces, diagnostics


class TestAnalyticSourceAnchors(unittest.TestCase):
    def test_unit_conversions_and_lammps_anchors(self):
        energy, force = source_bond_energy_force(0.010)
        assert_close(self, energy, 2.092, atol=1e-12)
        assert_close(self, force, 418.4, atol=1e-10)
        assert_close(self, OPENMM_BOND_K_KJ_PER_NM2, 41840.0, atol=1e-10)

        angle_energy = source_angle_energy(math.radians(170.0))
        assert_close(self, angle_energy, 0.038627419118532146, atol=1e-13)
        assert_close(self, ANGLE_K_KJ_PER_MOL, 167.36, atol=1e-12)

        assert_close(self, wca_energy(WCA_SIGMA_NM), 16.736, atol=1e-12)
        self.assertEqual(wca_energy(WCA_CUTOFF_NM), 0.0)

        assert_close(self, kapoor_source_dh_energy(1.0), 0.6388937251, atol=2e-11)
        assert_close(self, kapoor_source_dh_force_magnitude(1.0), 1.2777874502, atol=4e-11)

    def test_c3_dh_frozen_values_and_shift(self):
        epsilon_yu, kappa = c3_dh_parameters(310.0, 0.100)
        assert_close(self, epsilon_yu, 1.8727351727, atol=3e-11)
        assert_close(self, kappa, 1.0486598703, atol=3e-11)
        self.assertEqual(shifted_dh_energy(4.0, -1.0, epsilon_yu, kappa, 4.0), 0.0)
        self.assertLess(shifted_dh_energy(1.0, -1.0, epsilon_yu, kappa, 4.0), 0.0)

    def test_stack_and_hb_lammps_cpp_form(self):
        for pair, delta in STACK_DELTA_KJ_PER_MOL.items():
            assert_close(
                self,
                twelve_ten_energy(STACK_SIGMA_NM, STACK_SIGMA_NM, delta, STACK_CUTOFF_NM),
                -delta,
                atol=1e-10,
                message=f"stack minimum failed for {pair}",
            )
        for pair, (sigma, delta) in HB_PARAMETERS.items():
            assert_close(
                self,
                twelve_ten_energy(sigma, sigma, delta, HB_CUTOFF_NM),
                -delta,
                atol=1e-10,
                message=f"HB minimum failed for {pair}",
            )
            self.assertEqual(twelve_ten_energy(HB_CUTOFF_NM, sigma, delta, HB_CUTOFF_NM), 0.0)


class TestTopologyAndOwnership(unittest.TestCase):
    def test_reverse_complement_and_project_counts(self):
        self.assertEqual(reverse_complement("ATGC"), "GCAT")
        self.assertEqual(canonical_strand_sequences("ATGC", 3), ("ATGC", "GCAT") * 3)
        expected = expected_topology_counts(376, 3, 72_900)
        self.assertEqual(
            expected,
            {
                "protein_particles": 72_900,
                "duplexes": 3,
                "strands": 6,
                "nucleotides": 2_256,
                "dna_particles": 4_512,
                "total_particles": 77_412,
                "bonds": 4_506,
                "angles": 2_244,
                "stacking_pairs": 2_250,
                "next_nearest_base_pairs": 2_244,
                "dna_total_charge_e": -2_256,
            },
        )
        self.assertEqual(
            PROJECT_DNA_EXTRA_EXCLUSION_COUNT,
            expected["bonds"] + expected["stacking_pairs"] + expected["next_nearest_base_pairs"],
        )
        self.assertEqual(PROJECT_DNA_EXTRA_EXCLUSION_COUNT, 9_000)

    def test_mutually_exclusive_pair_ownership(self):
        bb0 = make_site(0, "backbone", "A", strand=0, nt=0)
        base0 = make_site(1, "base", "A", strand=0, nt=0)
        base1_t = make_site(2, "base", "T", strand=0, nt=1)
        base2_t = make_site(3, "base", "T", strand=0, nt=2)
        base3_t = make_site(4, "base", "T", strand=0, nt=3)
        other_t = make_site(5, "base", "T", strand=1, nt=0)
        other_a = make_site(6, "base", "A", strand=1, nt=0)

        self.assertEqual(pair_ownership(bb0, base0, ((0, 1),)), PairOwnership.EXCLUDED_12)
        self.assertEqual(pair_ownership(bb0, other_t), PairOwnership.WCA_BACKBONE)
        self.assertEqual(pair_ownership(base0, base1_t), PairOwnership.STACKING)
        self.assertEqual(pair_ownership(base0, base2_t), PairOwnership.WCA_NEXT_NEAREST)
        self.assertEqual(pair_ownership(base0, base3_t), PairOwnership.HYDROGEN_BOND)
        self.assertEqual(pair_ownership(base0, other_t), PairOwnership.HYDROGEN_BOND)
        self.assertEqual(pair_ownership(base0, other_a), PairOwnership.WCA_BASE_GENERIC)

    def test_every_small_topology_pair_has_exactly_one_owner(self):
        # Six sites cover backbone, bonded, adjacent, next-nearest, dynamic-HB,
        # and generic base WCA branches without relying on global particle tags.
        sites = (
            make_site(0, "backbone", "A", 0, 0),
            make_site(1, "base", "A", 0, 0),
            make_site(2, "backbone", "T", 0, 1),
            make_site(3, "base", "T", 0, 1),
            make_site(4, "base", "T", 0, 2),
            make_site(5, "base", "A", 1, 0),
        )
        bonds = ((0, 1), (0, 2), (2, 3))
        valid = set(PairOwnership)
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                self.assertIn(pair_ownership(sites[i], sites[j], bonds), valid)


class TestOpenMMBondAngleAndShortRange(unittest.TestCase):
    def test_bond_energy_force_and_pbc(self):
        system = make_system(2)
        force = create_dna_bond_force(((0, 1, "BB-BB"),), force_group=10)
        self.assertTrue(force.usesPeriodicBoundaryConditions())
        system.addForce(force)
        # Minimum-image distance is 0.56 nm across a 10-nm box: dr=0.01.
        energy, forces = evaluate(system, ((0.10, 5, 5), (9.54, 5, 5)), groups={10})
        expected_energy, expected_force = source_bond_energy_force(0.010)
        assert_close(self, energy, expected_energy, atol=1e-8)
        assert_close(self, np.linalg.norm(forces[1]), expected_force, rtol=FORCE_RTOL, atol=FORCE_ATOL)
        self.assertEqual(BB_BB_R0_NM, 0.55)

    def test_cosine_squared_angle_energy(self):
        system = make_system(3)
        force = create_dna_angle_force(((0, 1, 2),), force_group=11)
        self.assertTrue(force.usesPeriodicBoundaryConditions())
        system.addForce(force)
        theta = math.radians(170.0)
        positions = (
            (6.0, 5.0, 5.0),
            (5.0, 5.0, 5.0),
            (5.0 + math.cos(theta), 5.0 + math.sin(theta), 5.0),
        )
        energy, _ = evaluate(system, positions, groups={11})
        assert_close(self, energy, source_angle_energy(theta), atol=1e-9)

    def test_actual_short_range_ownership_energies(self):
        # Same-strand adjacent A/T is stacking only, despite complementarity.
        a = make_site(0, "base", "A", strand=0, nt=0)
        t_adj = make_site(1, "base", "T", strand=0, nt=1)
        energy, _, force = dd_pair_state(a, t_adj, ((2, 2, 2), (2.40, 2, 2)))
        expected = twelve_ten_energy(0.40, STACK_SIGMA_NM, STACK_DELTA_KJ_PER_MOL["AT"], STACK_CUTOFF_NM)
        assert_close(self, energy, expected, atol=1e-7)
        self.assertTrue(force["pbc"])

        # Same-strand i/i+2 A/T is WCA only, not HB.
        t_next = make_site(1, "base", "T", strand=0, nt=2)
        energy, _, _ = dd_pair_state(a, t_next, ((2, 2, 2), (2.55, 2, 2)))
        assert_close(self, energy, WCA_EPSILON_KJ_PER_MOL, atol=1e-7)

        # Other complementary A/T is a dynamic HB, including inter-strand.
        t_other = make_site(1, "base", "T", strand=5, nt=19, dna_id=2)
        energy, _, _ = dd_pair_state(a, t_other, ((2, 2, 2), (2.60, 2, 2)))
        assert_close(self, energy, -HB_PARAMETERS["AT"][1], atol=1e-7)

        # Other non-complementary base pair is generic WCA.
        a_other = make_site(1, "base", "A", strand=5, nt=19, dna_id=2)
        energy, _, _ = dd_pair_state(a, a_other, ((2, 2, 2), (2.55, 2, 2)))
        assert_close(self, energy, WCA_EPSILON_KJ_PER_MOL, atol=1e-7)

        # Any eligible pair containing backbone is WCA.
        bb = make_site(0, "backbone", "A", strand=0, nt=0)
        energy, _, _ = dd_pair_state(bb, a_other, ((2, 2, 2), (2.55, 2, 2)))
        assert_close(self, energy, WCA_EPSILON_KJ_PER_MOL, atol=1e-7)

        # A true 1-2 bond is excluded from all DD short-range branches.
        energy, forces, force = dd_pair_state(bb, a_other, ((2, 2, 2), (2.55, 2, 2)), exclusions=((0, 1),))
        assert_close(self, energy, 0.0, atol=1e-12)
        self.assertTrue(np.allclose(forces, 0.0, atol=1e-12))
        self.assertEqual(force["exclusions"]["dna_backbone_wca"], 1)

    def test_cutoffs_and_minimum_image_pbc(self):
        a = make_site(0, "base", "A", strand=0, nt=0)
        t = make_site(1, "base", "T", strand=1, nt=7)
        inside, _, _ = dd_pair_state(a, t, ((2, 2, 2), (2 + HB_CUTOFF_NM - 1e-4, 2, 2)))
        outside, _, _ = dd_pair_state(a, t, ((2, 2, 2), (2 + HB_CUTOFF_NM + 1e-4, 2, 2)))
        self.assertNotEqual(inside, 0.0)
        assert_close(self, outside, 0.0, atol=1e-12)

        # Non-complementary A/A across the boundary has minimum-image r=0.55.
        a2 = make_site(1, "base", "A", strand=1, nt=7)
        pbc_energy, _, _ = dd_pair_state(a, a2, ((0.10, 2, 2), (9.55, 2, 2)))
        assert_close(self, pbc_energy, WCA_EPSILON_KJ_PER_MOL, atol=2e-6)

    def test_short_range_finite_difference_gradient(self):
        a = make_site(0, "base", "A", strand=0, nt=0)
        a2 = make_site(1, "base", "A", strand=1, nt=7)
        r = 0.60
        h = 1.0e-5
        e0, forces, _ = dd_pair_state(a, a2, ((2, 2, 2), (2 + r, 2, 2)))
        ep, _, _ = dd_pair_state(a, a2, ((2, 2, 2), (2 + r + h, 2, 2)))
        em, _, _ = dd_pair_state(a, a2, ((2, 2, 2), (2 + r - h, 2, 2)))
        fd_force_on_second = -(ep - em) / (2.0 * h)
        self.assertTrue(math.isfinite(e0))
        assert_close(self, forces[1, 0], fd_force_on_second, rtol=FORCE_RTOL, atol=2e-5)


class TestProteinDNAAndFrozenPP(unittest.TestCase):
    C3_RESIDUES = (
        ("R", 0.656, 0.7407902764839954),
        ("D", 0.558, 0.0925875575361580),
        ("N", 0.568, 0.3706962163690402),
        ("E", 0.592, 0.0002495905394260),
        ("K", 0.636, 0.1380602542039267),
        ("H", 0.608, 0.4087176216525476),
        ("Q", 0.602, 0.3143449791669133),
        ("S", 0.518, 0.4473142572693176),
        ("C", 0.548, 0.5922529084601322),
        ("G", 0.450, 0.7538308115197386),
        ("T", 0.562, 0.2672387936544146),
        ("A", 0.504, 0.3377244362031627),
        ("M", 0.618, 0.5170874160398543),
        ("Y", 0.646, 0.9506286873011070),
        ("V", 0.586, 0.2936174211771383),
        ("W", 0.678, 1.0334501235745120),
        ("L", 0.618, 0.5548615312993875),
        ("I", 0.618, 0.5130398874425708),
        ("P", 0.556, 0.3469777523519372),
        ("F", 0.636, 0.8906449355499866),
    )

    def test_20_by_5_pd_arithmetic_mixing(self):
        checked = 0
        for aa, sigma, lam in self.C3_RESIDUES:
            for dna_type in ("BB", "A", "T", "C", "G"):
                mixed_sigma, mixed_lambda = mix_pd_parameters(sigma, lam, dna_type)
                assert_close(self, mixed_sigma, 0.5 * (sigma + 0.55), atol=1e-14, message=f"sigma {aa}-{dna_type}")
                assert_close(
                    self,
                    mixed_lambda,
                    0.5 * (lam + DNA_LAMBDA[dna_type]),
                    atol=1e-14,
                    message=f"lambda {aa}-{dna_type}",
                )
                self.assertEqual(c3_shifted_ah_energy(PD_AH_CUTOFF_NM, mixed_sigma, mixed_lambda), 0.0)
                checked += 1
        self.assertEqual(checked, 100)

    def test_pd_ah_openmm_energy_and_gradient(self):
        system = make_system(2)
        site = make_site(1, "base", "G", strand=0, nt=0)
        force = create_pd_ah_force(2, (0,), (0.656,), (0.7407902764839954,), (site,), force_group=14)
        system.addForce(force)
        self.assertEqual(force.getNumInteractionGroups(), 1)
        r = 0.80
        h = 1.0e-5
        positions = np.array(((2, 2, 2), (2 + r, 2, 2)), float)
        energy, forces = evaluate(system, positions, groups={14})
        mixed_sigma, mixed_lambda = mix_pd_parameters(0.656, 0.7407902764839954, "G")
        expected = c3_shifted_ah_energy(r, mixed_sigma, mixed_lambda)
        assert_close(self, energy, expected, atol=1e-8)
        ep, _ = evaluate(system, ((2, 2, 2), (2 + r + h, 2, 2)), groups={14})
        em, _ = evaluate(system, ((2, 2, 2), (2 + r - h, 2, 2)), groups={14})
        fd_force_on_second = -(ep - em) / (2 * h)
        assert_close(self, forces[1, 0], fd_force_on_second, rtol=FORCE_RTOL, atol=2e-6)

    def test_pd_dh_openmm_energy_gradient_standalone_pair(self):
        system = make_system(2)
        epsilon_yu, kappa = c3_dh_parameters()
        force = create_shifted_dh_force(
            2,
            {0: 1.0, 1: -1.0},
            (0,),
            (1,),
            (),
            epsilon_yu,
            kappa,
            4.0,
            "test_PD_DH",
            15,
        )
        system.addForce(force)
        r = 1.0
        h = 1.0e-5
        energy, forces = evaluate(system, ((2, 2, 2), (2 + r, 2, 2)), groups={15})
        expected = shifted_dh_energy(r, -1.0, epsilon_yu, kappa, 4.0)
        assert_close(self, energy, expected, atol=1e-9)
        ep, _ = evaluate(system, ((2, 2, 2), (2 + r + h, 2, 2)), groups={15})
        em, _ = evaluate(system, ((2, 2, 2), (2 + r - h, 2, 2)), groups={15})
        fd_force_on_second = -(ep - em) / (2 * h)
        assert_close(self, forces[1, 0], fd_force_on_second, rtol=FORCE_RTOL, atol=2e-7)

    @staticmethod
    def _frozen_two_protein_system(n_proteins=2):
        system = make_system(n_proteins)
        eps = 0.2 * 4.184
        rc = 2.0
        expression = (
            f"{eps}*select(step(r-2^(1/6)*s),4*l*((s/r)^12-(s/r)^6-shift),"
            f"4*((s/r)^12-(s/r)^6-l*shift)+(1-l));"
            f"l=select(id1+id2,(id1*id2)*0.5*(l1+l2),0);"
            f"shift=(s/{rc})^12-(s/{rc})^6;s=0.5*(s1+s2)"
        )
        ah = openmm.CustomNonbondedForce(expression)
        for name in ("s", "l", "id"):
            ah.addPerParticleParameter(name)
        sigmas = (0.656, 0.558)[:n_proteins]
        lambdas = (0.7407902764839954, 0.0925875575361580)[:n_proteins]
        for sigma, lam in zip(sigmas, lambdas):
            ah.addParticle((sigma * unit.nanometer, lam, 1.0))
        ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        ah.setCutoffDistance(2.0 * unit.nanometer)
        ah.setForceGroup(0)
        ah.setName("Frozen_C3_AH")

        epsilon_yu, kappa = c3_dh_parameters()
        shift = math.exp(-kappa * 4.0) / 4.0
        dh = openmm.CustomNonbondedForce(f"q*{epsilon_yu}*(exp(-{kappa}*r)/r-{shift});q=q1*q2")
        dh.addPerParticleParameter("q")
        charges = (1.0, -1.0)[:n_proteins]
        for charge in charges:
            dh.addParticle((charge,))
        dh.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        dh.setCutoffDistance(4.0 * unit.nanometer)
        dh.setForceGroup(1)
        dh.setName("Frozen_C3_DH")
        system.addForce(ah)
        system.addForce(dh)
        return system, ah, dh, charges, sigmas, lambdas

    def test_mismatched_frozen_pp_exclusion_lists_rejected_before_mutation(self):
        system, ah, dh, charges, sigmas, lambdas = self._frozen_two_protein_system()
        ah.addExclusion(0, 1)
        dh.addExclusion(1, 0)
        dna_coords = np.array(
            [
                [[[2.8, 2.0, 2.0], [3.0, 2.0, 2.0]]],
                [[[2.0, 2.9, 2.0], [2.0, 3.1, 2.0]]],
            ],
            dtype=float,
        )
        with self.assertRaisesRegex(ValueError, "identical in count, order"):
            append_kapoor2b_to_system(
                system,
                protein_indices=(0, 1),
                protein_charges_e=charges,
                protein_sigmas_nm=sigmas,
                protein_lambdas=lambdas,
                protein_ah_force=ah,
                protein_dh_force=dh,
                dna_sequence="A",
                dna_coordinates_nm=dna_coords,
                n_duplexes=1,
            )
        self.assertEqual(system.getNumParticles(), 2)
        self.assertEqual(ah.getNumParticles(), 2)
        self.assertEqual(dh.getNumParticles(), 2)

    def test_pp_before_after_energy_force_and_dummy_zero_regression(self):
        system, ah, dh, charges, sigmas, lambdas = self._frozen_two_protein_system()
        protein_positions = np.array(((2.0, 2.0, 2.0), (2.9, 2.0, 2.0)), float)
        energy_before, forces_before = evaluate(system, protein_positions, groups={0, 1})
        self.assertEqual(ah.getNumInteractionGroups(), 0)

        # Dummy DNA beads are intentionally close to proteins.  Any missing AH
        # mask would produce readily detectable PP-group energy/forces.
        dna_coords = np.array(
            [
                [[[2.35, 2.20, 2.00], [2.45, 2.20, 2.00]]],
                [[[2.55, 2.40, 2.00], [2.65, 2.40, 2.00]]],
            ],
            dtype=float,
        )
        manifest = append_kapoor2b_to_system(
            system,
            protein_indices=(0, 1),
            protein_charges_e=charges,
            protein_sigmas_nm=sigmas,
            protein_lambdas=lambdas,
            protein_ah_force=ah,
            protein_dh_force=dh,
            dna_sequence="A",
            dna_coordinates_nm=dna_coords,
            n_duplexes=1,
        )
        combined = np.vstack((protein_positions, manifest.dna_positions_nm))
        energy_after, forces_after = evaluate(system, combined, groups={0, 1})

        assert_close(self, energy_after, energy_before, rtol=1e-10, atol=1e-10)
        self.assertTrue(np.allclose(forces_after[:2], forces_before, rtol=1e-10, atol=1e-10))
        self.assertTrue(np.allclose(forces_after[2:], 0.0, rtol=0.0, atol=1e-12))
        self.assertEqual(ah.getNumInteractionGroups(), 0, "PP AH must not receive a protein self interaction group")
        self.assertIn("pp_dummy_mask=id1*id2", ah.getEnergyFunction().replace(" ", ""))
        for particle in manifest.dna_indices:
            ah_params = ah.getParticleParameters(particle)
            self.assertEqual(float(ah_params[0]), 0.0)
            self.assertEqual(float(ah_params[1]), 0.0)
            self.assertEqual(float(ah_params[2]), 0.0)
            self.assertEqual(float(dh.getParticleParameters(particle)[0]), 0.0)

    def test_isolated_protein_dna_pp_zero_and_independent_pd_nonzero(self):
        system, ah, dh, charges, sigmas, lambdas = self._frozen_two_protein_system(n_proteins=1)
        protein_positions = np.array(((2.0, 2.0, 2.0),), dtype=float)
        dna_coords = np.array(
            [
                [[[2.80, 2.00, 2.00], [3.00, 2.00, 2.00]]],
                [[[2.00, 2.90, 2.00], [2.00, 3.10, 2.00]]],
            ],
            dtype=float,
        )
        manifest = append_kapoor2b_to_system(
            system,
            protein_indices=(0,),
            protein_charges_e=charges,
            protein_sigmas_nm=sigmas,
            protein_lambdas=lambdas,
            protein_ah_force=ah,
            protein_dh_force=dh,
            dna_sequence="A",
            dna_coordinates_nm=dna_coords,
            n_duplexes=1,
        )
        positions = np.vstack((protein_positions, manifest.dna_positions_nm))

        pp_energy, pp_forces = evaluate(system, positions, groups={0, 1})
        assert_close(self, pp_energy, 0.0, atol=1e-12)
        self.assertTrue(np.allclose(pp_forces, 0.0, rtol=0.0, atol=1e-12))

        pd_ah_energy, pd_ah_forces = evaluate(system, positions, groups={14})
        self.assertNotEqual(pd_ah_energy, 0.0)
        self.assertGreater(float(np.linalg.norm(pd_ah_forces)), 0.0)
        pd_dh_energy, pd_dh_forces = evaluate(system, positions, groups={15})
        self.assertNotEqual(pd_dh_energy, 0.0)
        self.assertGreater(float(np.linalg.norm(pd_dh_forces)), 0.0)

    def test_all_custom_nonbonded_exclusions_identical_and_cpu_full_context(self):
        system, ah, dh, charges, sigmas, lambdas = self._frozen_two_protein_system()
        seq = "ATG"
        coords = np.zeros((2, len(seq), 2, 3), dtype=float)
        for strand, y0 in enumerate((2.0, 6.0)):
            for nt in range(len(seq)):
                x = 2.0 + 0.55 * nt
                coords[strand, nt, 0] = (x, y0, 2.0)
                coords[strand, nt, 1] = (x, y0 + 0.575, 2.0)
        manifest = append_kapoor2b_to_system(
            system,
            protein_indices=(0, 1),
            protein_charges_e=charges,
            protein_sigmas_nm=sigmas,
            protein_lambdas=lambdas,
            protein_ah_force=ah,
            protein_dh_force=dh,
            dna_sequence=seq,
            dna_coordinates_nm=coords,
            n_duplexes=1,
        )

        custom_nonbonded = [
            force for force in system.getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        self.assertEqual(len(custom_nonbonded), 7)
        expected_exclusions = tuple(manifest.global_nonbonded_exclusions)
        self.assertTrue(expected_exclusions)
        for force in custom_nonbonded:
            self.assertEqual(
                ordered_exclusions(force), expected_exclusions,
                msg=f"ordered exclusions differ for {force.getName()}",
            )
        self.assertFalse(
            any((a < manifest.protein_particle_count) != (b < manifest.protein_particle_count)
                for a, b in expected_exclusions)
        )

        dd_a, dd_b = manifest.forces["dd_dh"].getInteractionGroupParameters(0)
        self.assertEqual(set(dd_a), set(manifest.backbone_indices))
        self.assertEqual(set(dd_b), set(manifest.backbone_indices))
        pd_a, pd_b = manifest.forces["pd_dh"].getInteractionGroupParameters(0)
        self.assertEqual(set(pd_a), {0, 1})
        self.assertEqual(set(pd_b), set(manifest.backbone_indices))

        protein_positions = np.array(((7.0, 2.0, 2.0), (8.0, 2.0, 2.0)), dtype=float)
        positions = np.vstack((protein_positions, manifest.dna_positions_nm))
        energy, forces = evaluate(system, positions, platform_name="CPU")
        self.assertTrue(math.isfinite(energy))
        self.assertTrue(np.isfinite(forces).all())

    def test_builder_order_native_partner_counts_and_force_partition(self):
        system, ah, dh, charges, sigmas, lambdas = self._frozen_two_protein_system()
        seq = "ATGC"
        # Canonical (duplex, strand, nt, [BB/base], xyz) layout.
        coords = np.zeros((3, 2, len(seq), 2, 3), dtype=float)
        cursor = 0
        for duplex in range(3):
            for strand in range(2):
                for nt in range(len(seq)):
                    coords[duplex, strand, nt, 0] = (1.0 + duplex * 2, 1.0 + strand, 1.0 + nt * 0.7)
                    coords[duplex, strand, nt, 1] = (1.2 + duplex * 2, 1.0 + strand, 1.0 + nt * 0.7)
                    cursor += 2
        manifest = append_kapoor2b_to_system(
            system,
            protein_indices=(0, 1),
            protein_charges_e=charges,
            protein_sigmas_nm=sigmas,
            protein_lambdas=lambdas,
            protein_ah_force=ah,
            protein_dh_force=dh,
            dna_sequence=seq,
            dna_coordinates_nm=coords.reshape(6, len(seq), 2, 3),
            n_duplexes=3,
        )
        expected = expected_topology_counts(4, 3, 2)
        self.assertEqual(manifest.strand_sequences, ("ATGC", "GCAT") * 3)
        self.assertEqual(manifest.dna_particle_count, expected["dna_particles"])
        self.assertEqual(manifest.total_particle_count, expected["total_particles"])
        self.assertEqual(len(manifest.bonds), expected["bonds"])
        self.assertEqual(len(manifest.angles), expected["angles"])
        self.assertEqual(len(manifest.stacking_pairs), expected["stacking_pairs"])
        self.assertEqual(len(manifest.next_nearest_base_pairs), expected["next_nearest_base_pairs"])
        self.assertEqual(manifest.total_dna_charge_e, expected["dna_total_charge_e"])
        self.assertEqual(len(manifest.exclusions_12), len(manifest.bonds))

        # Protein count=2. First duplex A bases are 3,5,7,9; B bases are
        # 11,13,15,17.  Native antiparallel mapping is i <-> N-1-i.
        by_index = {site.particle_index: site for site in manifest.site_records}
        self.assertEqual(by_index[3].native_partner, 17)
        self.assertEqual(by_index[5].native_partner, 15)
        self.assertEqual(by_index[2].native_partner, None)
        self.assertEqual(by_index[3].dna_id, 0)
        self.assertEqual(by_index[3].strand_id, 0)
        self.assertEqual(by_index[11].strand_id, 1)
        self.assertEqual(by_index[19].dna_id, 1)

        forces = manifest.forces
        expected_force_keys = {
            "dna_bonds",
            "dna_angles",
            "dna_stacking",
            "dna_next_nearest_wca",
            "dna_base_nonbonded",
            "dna_backbone_wca",
            "dd_dh",
            "pd_ah",
            "pd_dh",
        }
        self.assertEqual(set(forces), expected_force_keys)
        expected_global = len(manifest.global_nonbonded_exclusions)
        self.assertEqual(
            expected_global,
            len(manifest.protein_nonbonded_exclusions)
            + len(manifest.bonds)
            + len(manifest.stacking_pairs)
            + len(manifest.next_nearest_base_pairs),
        )
        for name in ("dna_base_nonbonded", "dna_backbone_wca", "dd_dh", "pd_ah", "pd_dh"):
            self.assertEqual(forces[name].getNumExclusions(), expected_global)
        self.assertEqual(ah.getNumExclusions(), expected_global)
        self.assertEqual(dh.getNumExclusions(), expected_global)
        self.assertFalse(
            any((a < manifest.protein_particle_count) != (b < manifest.protein_particle_count)
                for a, b in manifest.global_nonbonded_exclusions)
        )
        for force in forces.values():
            self.assertTrue(force.usesPeriodicBoundaryConditions())


if __name__ == "__main__":
    unittest.main(verbosity=2)
