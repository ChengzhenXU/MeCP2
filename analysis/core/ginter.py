from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


AA_SIGMA = {
    "ALA": 0.504, "ARG": 0.656, "ASN": 0.568, "ASP": 0.558,
    "CYS": 0.548, "GLN": 0.602, "GLU": 0.592, "GLY": 0.450,
    "HIS": 0.608, "ILE": 0.618, "LEU": 0.618, "LYS": 0.636,
    "MET": 0.618, "PHE": 0.636, "PRO": 0.556, "SER": 0.518,
    "THR": 0.562, "TRP": 0.678, "TYR": 0.646, "VAL": 0.586,
}

PATCHES = (
    ("IDR", 1, 90),
    ("FD", 91, 162),
    ("P1", 163, 205),
    ("M1", 206, 241),
    ("P2", 242, 313),
    ("M2", 314, 375),
    ("N1", 376, 409),
    ("CTR", 410, 486),
)


def residue_sigma(residue_name: str) -> float:
    name = residue_name.upper()
    if name in AA_SIGMA:
        return AA_SIGMA[name]
    if name.startswith("H"):
        return AA_SIGMA["HIS"]
    if name == "GLH":
        return AA_SIGMA["GLU"]
    if name == "ASH":
        return AA_SIGMA["ASP"]
    return 0.5


def sigma_matrix(residue_names: Sequence[str]) -> np.ndarray:
    sigmas = np.asarray([residue_sigma(name) for name in residue_names], dtype=np.float32)
    return (sigmas[:, None] + sigmas[None, :]) / 2.0


def minimum_image(delta: np.ndarray, box_lengths: np.ndarray | None) -> np.ndarray:
    if box_lengths is None:
        return delta
    box = np.asarray(box_lengths, dtype=np.float64)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
        raise ValueError(f"invalid orthorhombic box lengths: {box_lengths}")
    return delta - box * np.rint(delta / box)


def contact_frame_reference(
    xyz: np.ndarray,
    residue_sigma_matrix: np.ndarray,
    box_lengths: np.ndarray | None,
    beta_q: float = 50.0,
    gamma_q: float = 1.5,
    exclude_intra_lt: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference for one frame using the original contact algebra."""
    coords = np.asarray(xyz, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("xyz must have shape (chains,residues,3)")
    n_chains, n_residues, _ = coords.shape
    if residue_sigma_matrix.shape != (n_residues, n_residues):
        raise ValueError("sigma matrix shape mismatch")

    diff_intra = coords[:, :, None, :] - coords[:, None, :, :]
    diff_intra = minimum_image(diff_intra, box_lengths)
    distance_intra = np.linalg.norm(diff_intra, axis=-1)
    exponent = np.clip(beta_q * (distance_intra - gamma_q * residue_sigma_matrix), -100.0, 100.0)
    q_intra = 1.0 / (1.0 + np.exp(exponent))
    indices = np.arange(n_residues)
    q_intra[:, np.abs(indices[:, None] - indices[None, :]) < exclude_intra_lt] = 0.0
    intra_sum = q_intra.sum(axis=0)

    inter_acc = np.zeros((n_residues, n_residues), dtype=np.float64)
    for chain_i in range(n_chains - 1):
        diff = coords[chain_i][None, :, None, :] - coords[chain_i + 1 :, None, :, :]
        diff = minimum_image(diff, box_lengths)
        distance = np.linalg.norm(diff, axis=-1)
        exponent = np.clip(beta_q * (distance - gamma_q * residue_sigma_matrix), -100.0, 100.0)
        inter_acc += (1.0 / (1.0 + np.exp(exponent))).sum(axis=0)
    inter_sym = (inter_acc + inter_acc.T) / 2.0
    return intra_sum, inter_sym


class DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]

    def component_sizes(self) -> list[int]:
        counts: dict[int, int] = {}
        for value in range(len(self.parent)):
            root = self.find(value)
            counts[root] = counts.get(root, 0) + 1
        return sorted(counts.values(), reverse=True)


def component_sizes_from_pairs(chain_pairs: np.ndarray, num_chains: int) -> list[int]:
    dsu = DSU(num_chains)
    for pair in np.asarray(chain_pairs):
        dsu.union(int(pair[0]), int(pair[1]))
    return dsu.component_sizes()


def chain_pairs_within_cutoffs(
    xyz: np.ndarray,
    box_lengths: np.ndarray | None,
    cutoffs_nm: Iterable[float],
    num_chains: int,
    residues_per_chain: int,
) -> dict[float, np.ndarray]:
    from scipy.spatial import cKDTree

    cutoffs = tuple(sorted(float(value) for value in cutoffs_nm))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("cutoffs must be positive")
    coords = np.asarray(xyz, dtype=np.float64).reshape(num_chains * residues_per_chain, 3)
    box: np.ndarray | None
    if box_lengths is None:
        box = None
        wrapped = coords
        tree = cKDTree(wrapped)
    else:
        box = np.asarray(box_lengths, dtype=np.float64)
        if box.shape != (3,) or np.any(box <= 2.0 * max(cutoffs)):
            raise ValueError("box is invalid or too small for periodic cutoff")
        wrapped = np.mod(coords, box)
        tree = cKDTree(wrapped, boxsize=box)

    atom_pairs = tree.query_pairs(max(cutoffs), output_type="ndarray")
    if atom_pairs.size == 0:
        return {cutoff: np.empty((0, 2), dtype=np.int16) for cutoff in cutoffs}
    delta = wrapped[atom_pairs[:, 0]] - wrapped[atom_pairs[:, 1]]
    delta = minimum_image(delta, box)
    distance = np.linalg.norm(delta, axis=1)
    chain_i = atom_pairs[:, 0] // residues_per_chain
    chain_j = atom_pairs[:, 1] // residues_per_chain
    inter = chain_i != chain_j
    all_chain_pairs = np.stack((chain_i[inter], chain_j[inter]), axis=1).astype(np.int16)
    inter_distance = distance[inter]
    result: dict[float, np.ndarray] = {}
    for cutoff in cutoffs:
        selected = all_chain_pairs[inter_distance <= cutoff]
        if selected.size == 0:
            result[cutoff] = np.empty((0, 2), dtype=np.int16)
        else:
            selected.sort(axis=1)
            result[cutoff] = np.unique(selected, axis=0)
    return result


def patch_arrays(
    residues_per_chain: int = 486,
    patches: Sequence[tuple[str, int, int]] = PATCHES,
) -> tuple[np.ndarray, tuple[tuple[str, str], ...], np.ndarray, dict[str, int]]:
    patch_ids = np.full(residues_per_chain, -1, dtype=np.int16)
    lengths: dict[str, int] = {}
    for patch_id, (name, start, stop) in enumerate(patches):
        if not (1 <= start <= stop <= residues_per_chain):
            raise ValueError(f"invalid patch {name}: {(start, stop)}")
        patch_ids[start - 1 : stop] = patch_id
        lengths[name] = stop - start + 1
    if np.any(patch_ids < 0):
        raise ValueError("patch scheme does not cover every residue")
    names = [entry[0] for entry in patches]
    pairs: list[tuple[str, str]] = []
    lookup = np.full((len(names), len(names)), -1, dtype=np.int16)
    for left in range(len(names)):
        for right in range(left, len(names)):
            pair_id = len(pairs)
            pairs.append((names[left], names[right]))
            lookup[left, right] = pair_id
            lookup[right, left] = pair_id
    return patch_ids, tuple(pairs), lookup, lengths


PATCH_ID_BY_RESIDUE, PATCH_PAIRS, PATCH_PAIR_ID, PATCH_LENGTHS = patch_arrays()


def possible_patch_pairs_per_chain_pair(pair: tuple[str, str], lengths: dict[str, int] = PATCH_LENGTHS) -> int:
    left, right = pair
    if left == right:
        return lengths[left] * lengths[right]
    return 2 * lengths[left] * lengths[right]


def ginter_frame_histograms(
    xyz: np.ndarray,
    box_lengths: np.ndarray,
    bin_edges: np.ndarray,
    num_chains: int,
    residues_per_chain: int,
    patch_id_by_residue: np.ndarray = PATCH_ID_BY_RESIDUE,
    patch_pair_id: np.ndarray = PATCH_PAIR_ID,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    coords = np.asarray(xyz, dtype=np.float64).reshape(num_chains * residues_per_chain, 3)
    box = np.asarray(box_lengths, dtype=np.float64)
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 2.0 * float(bin_edges[-1])):
        raise ValueError("invalid periodic box for ginter")
    if patch_id_by_residue.shape != (residues_per_chain,):
        raise ValueError("patch lookup and residue count differ")
    wrapped = np.mod(coords, box)
    atom_pairs = cKDTree(wrapped, boxsize=box).query_pairs(float(bin_edges[-1]), output_type="ndarray")
    n_bins = len(bin_edges) - 1
    n_patch_pairs = int(patch_pair_id.max()) + 1
    total_hist = np.zeros(n_bins, dtype=np.float64)
    pair_hist = np.zeros((n_patch_pairs, n_bins), dtype=np.float64)
    if atom_pairs.size == 0:
        return total_hist, pair_hist
    delta = minimum_image(wrapped[atom_pairs[:, 0]] - wrapped[atom_pairs[:, 1]], box)
    distance = np.linalg.norm(delta, axis=1)
    bins = np.searchsorted(bin_edges, distance, side="right") - 1
    valid = (bins >= 0) & (bins < n_bins)
    atom_i = atom_pairs[valid, 0]
    atom_j = atom_pairs[valid, 1]
    bins = bins[valid]
    inter = atom_i // residues_per_chain != atom_j // residues_per_chain
    if not np.any(inter):
        return total_hist, pair_hist
    atom_i = atom_i[inter]
    atom_j = atom_j[inter]
    bins = bins[inter]
    pair_ids = patch_pair_id[
        patch_id_by_residue[atom_i % residues_per_chain],
        patch_id_by_residue[atom_j % residues_per_chain],
    ]
    total_hist += np.bincount(bins, minlength=n_bins)
    flat = pair_ids.astype(np.int64) * n_bins + bins.astype(np.int64)
    pair_hist += np.bincount(flat, minlength=n_patch_pairs * n_bins).reshape(n_patch_pairs, n_bins)
    return total_hist, pair_hist


def normalize_histogram(
    histogram: np.ndarray,
    possible_pairs_per_frame: int,
    shell_volumes: np.ndarray,
    inverse_volume_sum: float,
) -> np.ndarray:
    expected = possible_pairs_per_frame * shell_volumes * inverse_volume_sum
    result = np.full(np.asarray(histogram).shape, np.nan, dtype=np.float64)
    valid = expected > 0
    result[valid] = np.asarray(histogram)[valid] / expected[valid]
    return result


def summarize_ginter_curve(r_nm: np.ndarray, values: np.ndarray, cutoff_nm: float = 2.0) -> dict[str, float]:
    mask = (r_nm <= cutoff_nm) & np.isfinite(values)
    if not np.any(mask):
        return {key: math.nan for key in ("auc_excess_2nm", "weighted_auc_excess_2nm", "peak_2nm", "peak_r_nm")}
    r_cut = r_nm[mask]
    g_cut = values[mask]
    excess = np.clip(g_cut - 1.0, 0.0, None)
    peak_index = int(np.nanargmax(g_cut))
    return {
        "auc_excess_2nm": float(np.trapz(excess, r_cut)),
        "weighted_auc_excess_2nm": float(np.trapz(excess * r_cut**2, r_cut)),
        "peak_2nm": float(g_cut[peak_index]),
        "peak_r_nm": float(r_cut[peak_index]),
    }


def update_size_histogram(
    histogram: dict[tuple[int, int], int],
    block_id: int,
    component_sizes: Sequence[int],
) -> None:
    for size, count in Counter(component_sizes).items():
        histogram[(0, size)] = histogram.get((0, size), 0) + count
        histogram[(block_id, size)] = histogram.get((block_id, size), 0) + count

