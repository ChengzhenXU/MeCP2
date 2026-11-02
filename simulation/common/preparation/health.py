"""DNA preparation acceptance limits from the original QA gate specification."""

from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def bond_pairs():
    protein = np.arange(72900).reshape(150, 486)
    backbone = np.arange(72900, 77412, 2).reshape(6, 376)
    return {
        "protein_chain": (np.column_stack((protein[:, :-1].ravel(), protein[:, 1:].ravel())), 1.0),
        "dna_backbone": (np.column_stack((backbone[:, :-1].ravel(), backbone[:, 1:].ravel())), .85),
        "dna_backbone_base": (np.column_stack((backbone.ravel(), backbone.ravel() + 1)), .85),
    }


def validate_geometry(positions, forces, box, *, expected_forces, actual_forces,
                      final_box=False, final_minimization=False):
    if (positions.shape != (77412, 3) or forces.shape != positions.shape or
            not np.isfinite(positions).all() or not np.isfinite(forces).all() or not np.isfinite(box).all()):
        raise RuntimeError("DNA preparation must contain 77,412 particles.")
    if actual_forces != expected_forces:
        raise RuntimeError("DNA preparation force count differs from its stage.")
    lengths = np.diag(box)
    if not np.allclose(box, np.diag(lengths), rtol=0, atol=1e-8) or np.any(lengths <= 0):
        raise RuntimeError("DNA preparation requires a positive orthorhombic box.")
    for name, (pairs, limit) in bond_pairs().items():
        delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        delta -= lengths * np.rint(delta / lengths)
        maximum = np.linalg.norm(delta, axis=1).max()
        if maximum > limit:
            raise RuntimeError(f"{name} bond exceeds the original acceptance limit: {maximum:.6f} nm")
    if final_box and not np.allclose(lengths, [50, 50, 50], rtol=0, atol=1e-8):
        raise RuntimeError("Final DNA box differs from the 50 nm cube.")
    if final_minimization and np.sqrt(np.mean(np.sum(forces * forces, axis=1))) > 10:
        raise RuntimeError("Final DNA minimization RMS force exceeds 10 kJ mol^-1 nm^-1.")
