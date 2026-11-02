"""Frozen preparation protocol for MeCP2_DNA_FD production_v1.

The preparation schedule is stored as a frozen dataclass so that protocol
changes are detected before an OpenMM Context is created.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Tuple


PROTOCOL_SCHEMA = "mecp2-dna-fd-preparation-protocol-v1"

@dataclass(frozen=True)
class ThermalStage:
    temperature_k: float
    timestep_ps: float
    steps: int


@dataclass(frozen=True)
class PreparationProtocol:
    """The single authorized production_v1 preparation schedule."""

    initial_box_nm: Tuple[float, float, float] = (50.0, 50.0, 500.0)
    final_box_nm: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    thermal_stages: Tuple[ThermalStage, ...] = (
        ThermalStage(50.0, 0.001, 1_000),
        ThermalStage(100.0, 0.002, 1_000),
        ThermalStage(200.0, 0.005, 1_000),
        ThermalStage(310.0, 0.010, 2_000),
    )
    pre_equilibration_blocks: int = 10
    pre_equilibration_steps_per_block: int = 5_000
    compression_steps_per_block: int = 1_000
    compression_max_blocks: int = 5_000
    compression_initial_pz_bar: float = 1.0
    compression_pz_increment_bar: float = 1.0
    compression_max_pz_bar: float = 1_000.0
    barostat_frequency: int = 25
    final_scaling_stages: int = 6
    intermediate_minimization_iterations: int = 25
    final_minimization_iterations: int = 50
    initial_minimization_iterations: int = 200
    post_relaxation_blocks: int = 5
    post_relaxation_steps_per_block: int = 1_000
    nvt_equilibration_steps: int = 100_000
    nvt_timestep_ps: float = 0.010
    nvt_temperature_k: float = 310.0
    nvt_temperature_sample_interval: int = 1_000
    nvt_last_half_temperature_min_k: float = 305.0
    nvt_last_half_temperature_max_k: float = 315.0
    friction_per_ps: float = 0.01
    clean_force_count: int = 13
    compression_force_count: int = 14
    total_particles: int = 77_412
    protein_particles: int = 72_900
    dna_particles: int = 4_512
    protein_copies: int = 150
    residues_per_protein: int = 486
    dna_duplexes: int = 3
    dna_bp_per_duplex: int = 376
    frozen_restraints_total: int = 43_200
    frozen_restraints_per_copy: int = 288

    def validate(self) -> None:
        """Validate the reference preparation schedule."""

        expected = {
            "initial_box_nm": (50.0, 50.0, 500.0),
            "final_box_nm": (50.0, 50.0, 50.0),
            "thermal_stages": (
                ThermalStage(50.0, 0.001, 1_000),
                ThermalStage(100.0, 0.002, 1_000),
                ThermalStage(200.0, 0.005, 1_000),
                ThermalStage(310.0, 0.010, 2_000),
            ),
            "pre_equilibration_blocks": 10,
            "pre_equilibration_steps_per_block": 5_000,
            "compression_steps_per_block": 1_000,
            "compression_max_blocks": 5_000,
            "compression_initial_pz_bar": 1.0,
            "compression_pz_increment_bar": 1.0,
            "compression_max_pz_bar": 1_000.0,
            "barostat_frequency": 25,
            "final_scaling_stages": 6,
            "intermediate_minimization_iterations": 25,
            "final_minimization_iterations": 50,
            "initial_minimization_iterations": 200,
            "post_relaxation_blocks": 5,
            "post_relaxation_steps_per_block": 1_000,
            "nvt_equilibration_steps": 100_000,
            "nvt_timestep_ps": 0.010,
            "nvt_temperature_k": 310.0,
            "nvt_temperature_sample_interval": 1_000,
            "nvt_last_half_temperature_min_k": 305.0,
            "nvt_last_half_temperature_max_k": 315.0,
            "friction_per_ps": 0.01,
            "clean_force_count": 13,
            "compression_force_count": 14,
            "total_particles": 77_412,
            "protein_particles": 72_900,
            "dna_particles": 4_512,
            "protein_copies": 150,
            "residues_per_protein": 486,
            "dna_duplexes": 3,
            "dna_bp_per_duplex": 376,
            "frozen_restraints_total": 43_200,
            "frozen_restraints_per_copy": 288,
        }
        field_names = tuple(field.name for field in fields(self))
        missing_or_extra = sorted(set(field_names).symmetric_difference(expected))
        failed = missing_or_extra + sorted(
            name
            for name in field_names
            if name in expected and getattr(self, name) != expected[name]
        )
        if failed:
            raise RuntimeError(
                "FINAL preparation protocol drift: " + ", ".join(failed)
            )

    def pressure_for_block(self, block_index: int) -> float:
        """Return the pressure applied *before* a one-based compression block."""

        if type(block_index) is not int or block_index < 1:
            raise ValueError("block_index must be a positive one-based integer")
        return min(
            self.compression_initial_pz_bar
            + (block_index - 1) * self.compression_pz_increment_bar,
            self.compression_max_pz_bar,
        )


FROZEN_PROTOCOL = PreparationProtocol()
FROZEN_PROTOCOL.validate()

