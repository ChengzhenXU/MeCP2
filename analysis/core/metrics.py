from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.spatial import cKDTree

from .io_utils import AnalysisError, require_hash, write_json_atomic


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _reader(path: str | Path):
    from MDAnalysis.coordinates.DCD import DCDReader

    if Path(path).suffix.lower() == ".json":
        from .segments import SegmentReader
        return SegmentReader(path)
    return DCDReader(str(path), convert_units=True)


def _orthorhombic_box(ts: Any) -> np.ndarray:
    dimensions = np.asarray(ts.dimensions, dtype=np.float64)
    _require(dimensions.shape == (6,), "DCD frame has no six-value periodic box")
    _require(np.all(np.isfinite(dimensions)), "DCD frame box is non-finite")
    _require(np.all(dimensions[:3] > 0), "DCD frame box length is non-positive")
    _require(
        np.allclose(dimensions[3:], 90.0, atol=1.0e-3),
        f"Only orthorhombic boxes are supported, observed angles={dimensions[3:].tolist()}",
    )
    return dimensions[:3]


def _frames(path: str | Path, frame_slice: slice = slice(None)) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    with _reader(path) as reader:
        for index, ts in enumerate(reader[frame_slice]):
            positions_nm = np.asarray(ts.positions, dtype=np.float64) / 10.0
            _require(np.all(np.isfinite(positions_nm)), f"Non-finite coordinates in frame {index}")
            yield index, positions_nm, _orthorhombic_box(ts) / 10.0


def connected_component_sizes(chain_count: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(chain_count, dtype=np.int32)
    size = np.ones(chain_count, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in edges:
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
    roots = [node for node in range(chain_count) if find(node) == node]
    return np.asarray([size[node] for node in roots], dtype=np.int32)


def cluster_metrics(
    dcd_path: str | Path,
    identity_npz: str | Path,
    output_npz: str | Path,
    output_json: str | Path,
    *,
    dcd_sha256: str,
    cutoffs_nm: list[float],
    frame_slice: slice = slice(None),
) -> dict[str, Any]:
    require_hash(dcd_path, dcd_sha256, "input trajectory")
    identity = np.load(identity_npz)
    protein_indices = identity["protein_indices"].astype(np.int64, copy=False)
    _require(protein_indices.size % 486 == 0, "Protein identities are not 486-aa chains")
    chain_count = protein_indices.size // 486
    chain_local = np.repeat(np.arange(chain_count, dtype=np.int32), 486)
    event_counts = np.zeros((len(cutoffs_nm), chain_count + 1), dtype=np.int64)
    largest_rows: list[list[int]] = []

    for _frame_index, positions_nm, box_nm in _frames(dcd_path, frame_slice):
        protein_xyz = np.mod(positions_nm[protein_indices], box_nm)
        tree = cKDTree(protein_xyz, boxsize=box_nm)
        frame_largest: list[int] = []
        for cutoff_index, cutoff in enumerate(cutoffs_nm):
            bead_pairs = tree.query_pairs(float(cutoff), output_type="ndarray")
            if bead_pairs.size:
                chain_pairs = np.column_stack(
                    (chain_local[bead_pairs[:, 0]], chain_local[bead_pairs[:, 1]])
                )
                chain_pairs = chain_pairs[chain_pairs[:, 0] != chain_pairs[:, 1]]
                chain_pairs = np.unique(np.sort(chain_pairs, axis=1), axis=0)
            else:
                chain_pairs = np.empty((0, 2), dtype=np.int32)
            sizes = connected_component_sizes(chain_count, chain_pairs)
            event_counts[cutoff_index] += np.bincount(
                sizes, minlength=chain_count + 1
            )[: chain_count + 1]
            frame_largest.append(int(sizes.max()))
        largest_rows.append(frame_largest)

    largest = np.asarray(largest_rows, dtype=np.int16)
    _require(largest.shape[0] > 0, "No DCD frames were read")
    primary_index = int(np.argmin(np.abs(np.asarray(cutoffs_nm) - 0.55)))
    primary_largest = largest[:, primary_index]
    proportions = {
        str(threshold): float(np.mean(primary_largest >= threshold))
        for threshold in (60, 75, 90)
    }
    probabilities = np.zeros_like(event_counts, dtype=np.float64)
    for index in range(len(cutoffs_nm)):
        total = int(event_counts[index, 1:].sum())
        if total:
            probabilities[index] = event_counts[index] / total
    target = Path(output_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        cutoffs_nm=np.asarray(cutoffs_nm, dtype=np.float64),
        largest_cluster_by_frame=largest,
        cluster_event_counts=event_counts,
        cluster_event_probability=probabilities,
    )
    summary = {
        "schema": "mecp2-dna-fd-cluster-metrics-v1",
        "status": "PASS",
        "analysis_scope": "selected_frames",
        "frames": int(largest.shape[0]),
        "chain_count": chain_count,
        "primary_cutoff_nm": float(cutoffs_nm[primary_index]),
        "largest_cluster_mean": float(primary_largest.mean()),
        "largest_cluster_max": int(primary_largest.max()),
        "threshold_frame_proportions": proportions,
        "cutoff_sensitivity_mean_largest": {
            f"{cutoff:.2f}": float(largest[:, index].mean())
            for index, cutoff in enumerate(cutoffs_nm)
        },
        "interpretation_guard": "Operational connected components; not evidence of LLPS or phase coexistence.",
        "npz_path": str(target),
    }
    write_json_atomic(output_json, summary)
    return summary


def switching_weight(distance_nm: np.ndarray, sigma_nm: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    argument = np.clip(beta * (distance_nm - gamma * sigma_nm), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(argument))


def pp_contact_metrics(
    dcd_path: str | Path,
    identity_npz: str | Path,
    output_npz: str | Path,
    output_json: str | Path,
    *,
    dcd_sha256: str,
    beta_per_nm: float,
    gamma: float,
    kernel_tolerance: float,
    patches: list[dict[str, Any]],
    construct: str,
    frame_slice: slice = slice(None),
    frames_per_block: int | None = None,
    contact_engine: str = "sparse",
) -> dict[str, Any]:
    require_hash(dcd_path, dcd_sha256, "input trajectory")
    identity = np.load(identity_npz)
    protein_indices = identity["protein_indices"].astype(np.int64, copy=False)
    sigma = identity["sigma_nm"].astype(np.float64, copy=False)[protein_indices]
    _require(protein_indices.size % 486 == 0, "Protein identities are not 486-aa chains")
    chain_count = protein_indices.size // 486
    chain_local = np.repeat(np.arange(chain_count, dtype=np.int32), 486)
    residue_local = np.tile(np.arange(486, dtype=np.int16), chain_count)
    pair_denominator = math.comb(chain_count, 2)
    q_accumulator = np.zeros((486, 486), dtype=np.float64)
    frame_count = 0
    block_matrices = []
    previous_accumulator = q_accumulator.copy()
    tail = math.log((1.0 - kernel_tolerance) / kernel_tolerance) / beta_per_nm
    search_cutoff = gamma * float(sigma.max()) + tail

    for _frame_index, positions_nm, box_nm in _frames(dcd_path, frame_slice):
        xyz = np.mod(positions_nm[protein_indices], box_nm)
        if contact_engine == "reference":
            from .ginter import contact_frame_reference
            residue_sigma = sigma[:486]
            sigma_matrix = (residue_sigma[:, None] + residue_sigma[None, :]) / 2
            q_accumulator += contact_frame_reference(xyz.reshape(chain_count, 486, 3), sigma_matrix,
                                                      box_nm, beta_q=beta_per_nm, gamma_q=gamma)[1]
            pairs = np.empty((0, 2), dtype=np.int64)
        elif contact_engine == "sparse":
            tree = cKDTree(xyz, boxsize=box_nm)
            pairs = tree.query_pairs(search_cutoff, output_type="ndarray")
        else:
            raise ValueError("Contact engine must be reference or sparse.")
        if pairs.size:
            cross = chain_local[pairs[:, 0]] != chain_local[pairs[:, 1]]
            pairs = pairs[cross]
        if pairs.size:
            delta = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
            delta -= box_nm * np.rint(delta / box_nm)
            distance = np.linalg.norm(delta, axis=1)
            sigma_ij = 0.5 * (sigma[pairs[:, 0]] + sigma[pairs[:, 1]])
            weight = switching_weight(distance, sigma_ij, beta_per_nm, gamma)
            keep = weight >= kernel_tolerance
            pairs = pairs[keep]
            weight = weight[keep]
            left_residue = residue_local[pairs[:, 0]]
            right_residue = residue_local[pairs[:, 1]]
            np.add.at(q_accumulator, (left_residue, right_residue), 0.5 * weight)
            np.add.at(q_accumulator, (right_residue, left_residue), 0.5 * weight)
        frame_count += 1
        if frames_per_block and frame_count % frames_per_block == 0:
            block_matrices.append((q_accumulator - previous_accumulator) /
                                  (frames_per_block * pair_denominator))
            previous_accumulator = q_accumulator.copy()

    _require(frame_count > 0, "No DCD frames were read")
    qij = q_accumulator / (frame_count * pair_denominator)
    patch_count = len(patches)
    wab = np.zeros((patch_count, patch_count), dtype=np.float64)
    for left_index, left in enumerate(patches):
        left_slice = slice(int(left["start"]) - 1, int(left["end"]))
        for right_index, right in enumerate(patches):
            right_slice = slice(int(right["start"]) - 1, int(right["end"]))
            wab[left_index, right_index] = qij[left_slice, right_slice].sum()
    wa = wab.sum(axis=1)
    wtot = float(qij.sum())
    _require(np.isclose(wab.sum(), wtot, rtol=1.0e-10, atol=1.0e-12), "sum(W_AB) != W_tot")
    _require(np.allclose(wab, wab.T, rtol=1.0e-10, atol=1.0e-12), "W_AB is not symmetric")

    mutation_sites = {"R133C": 133, "T158M": 158, "P225R": 225, "R306C": 306}
    site = mutation_sites.get(construct)
    site_partner = qij[site - 1].copy() if site else np.asarray([], dtype=np.float64)
    site_to_patch = (
        np.asarray(
            [
                site_partner[int(patch["start"]) - 1 : int(patch["end"])].sum()
                for patch in patches
            ],
            dtype=np.float64,
        )
        if site
        else np.asarray([], dtype=np.float64)
    )
    target = Path(output_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        qij=qij,
        wab=wab,
        wa=wa,
        wtot=np.asarray(wtot),
        mutation_site=np.asarray(site if site else -1),
        mutation_site_partner=site_partner,
        mutation_site_to_patch=site_to_patch,
        qij_blocks=np.asarray(block_matrices, dtype=np.float64).reshape(-1, 486, 486),
    )
    summary = {
        "schema": "mecp2-dna-fd-pp-contact-metrics-v1",
        "status": "PASS",
        "analysis_scope": "selected_frames",
        "frames": frame_count,
        "chain_count": chain_count,
        "beta_per_nm": beta_per_nm,
        "gamma": gamma,
        "kernel_tolerance": kernel_tolerance if contact_engine == "sparse" else 0.0,
        "contact_engine": contact_engine,
        "unordered_chain_pair_denominator": pair_denominator,
        "w_tot": wtot,
        "w_a": {patch["name"]: float(wa[index]) for index, patch in enumerate(patches)},
        "mutation_site": site,
        "mutation_site_weight": float(site_partner.sum()) if site else None,
        "sum_wab_equals_wtot": True,
        "interpretation_guard": "Continuous geometric contact weight; not energy or affinity.",
        "npz_path": str(target),
    }
    write_json_atomic(output_json, summary)
    return summary


def _class_contact_summary(
    pair_protein_local: np.ndarray,
    pair_dna_local: np.ndarray,
    dna_duplex: np.ndarray,
    *,
    chain_count: int,
    residue_count: int,
) -> tuple[int, int, np.ndarray]:
    if pair_protein_local.size == 0:
        return 0, 0, np.zeros(residue_count, dtype=np.int64)
    chains = pair_protein_local // residue_count
    residues = pair_protein_local % residue_count
    contact_chains = int(np.unique(chains).size)
    events = int(np.unique(chains * 3 + dna_duplex[pair_dna_local]).size)
    chain_residue = np.unique(chains * residue_count + residues)
    residue_hits = np.bincount(chain_residue % residue_count, minlength=residue_count)
    return contact_chains, events, residue_hits


def protein_dna_metrics(
    dcd_path: str | Path,
    identity_npz: str | Path,
    output_npz: str | Path,
    output_json: str | Path,
    *,
    dcd_sha256: str,
    gamma: float,
    patches: list[dict[str, Any]],
    frame_slice: slice = slice(None),
    frames_per_block: int | None = None,
) -> dict[str, Any]:
    require_hash(dcd_path, dcd_sha256, "input trajectory")
    identity = np.load(identity_npz)
    protein_indices = identity["protein_indices"].astype(np.int64, copy=False)
    dna_indices = identity["dna_indices"].astype(np.int64, copy=False)
    base_indices = identity["base_indices"].astype(np.int64, copy=False)
    duplex_by_global = identity["duplex_by_global"].astype(np.int16, copy=False)
    sigma_all = identity["sigma_nm"].astype(np.float64, copy=False)
    chain_count = protein_indices.size // 486
    _require(chain_count * 486 == protein_indices.size, "Protein identity shape mismatch")
    dna_is_base = np.isin(dna_indices, base_indices)
    dna_duplex = duplex_by_global[dna_indices]
    _require(np.all(dna_duplex >= 0), "DNA particle without duplex identity")
    max_cutoff = gamma * 0.5 * (
        float(sigma_all[protein_indices].max()) + float(sigma_all[dna_indices].max())
    )
    classes = ("backbone", "base", "any")
    chain_counts: dict[str, list[int]] = {name: [] for name in classes}
    event_counts: dict[str, list[int]] = {name: [] for name in classes}
    per_duplex_event_counts: dict[str, list[list[int]]] = {
        name: [] for name in classes
    }
    residue_hits = {name: np.zeros(486, dtype=np.int64) for name in classes}
    previous_hits = {name: value.copy() for name, value in residue_hits.items()}
    block_frequencies = {name: [] for name in classes}
    frame_count = 0

    for _frame_index, positions_nm, box_nm in _frames(dcd_path, frame_slice):
        protein_xyz = np.mod(positions_nm[protein_indices], box_nm)
        dna_xyz = np.mod(positions_nm[dna_indices], box_nm)
        protein_tree = cKDTree(protein_xyz, boxsize=box_nm)
        dna_tree = cKDTree(dna_xyz, boxsize=box_nm)
        sparse = protein_tree.sparse_distance_matrix(
            dna_tree, max_cutoff, output_type="coo_matrix"
        )
        protein_local = sparse.row.astype(np.int64, copy=False)
        dna_local = sparse.col.astype(np.int64, copy=False)
        distance = sparse.data
        if distance.size:
            threshold = gamma * 0.5 * (
                sigma_all[protein_indices[protein_local]]
                + sigma_all[dna_indices[dna_local]]
            )
            accepted = distance <= threshold
            protein_local = protein_local[accepted]
            dna_local = dna_local[accepted]
        masks = {
            "backbone": ~dna_is_base[dna_local],
            "base": dna_is_base[dna_local],
            "any": np.ones(dna_local.size, dtype=bool),
        }
        for name in classes:
            count, events, hits = _class_contact_summary(
                protein_local[masks[name]],
                dna_local[masks[name]],
                dna_duplex,
                chain_count=chain_count,
                residue_count=486,
            )
            chain_counts[name].append(count)
            event_counts[name].append(events)
            residue_hits[name] += hits
            class_chains = protein_local[masks[name]] // 486
            class_duplex = dna_duplex[dna_local[masks[name]]]
            per_duplex_event_counts[name].append(
                [
                    int(np.unique(class_chains[class_duplex == duplex]).size)
                    for duplex in range(3)
                ]
            )
        frame_count += 1

        if frames_per_block and frame_count % frames_per_block == 0:
            for name in classes:
                block_frequencies[name].append((residue_hits[name] - previous_hits[name]) /
                                               (frames_per_block * chain_count))
                previous_hits[name] = residue_hits[name].copy()

    _require(frame_count > 0, "No DCD frames were read")
    residue_frequency = {
        name: residue_hits[name].astype(np.float64) / (frame_count * chain_count)
        for name in classes
    }
    patch_mean: dict[str, np.ndarray] = {}
    patch_sum: dict[str, np.ndarray] = {}
    for name in classes:
        patch_mean[name] = np.asarray(
            [
                residue_frequency[name][int(patch["start"]) - 1 : int(patch["end"])].mean()
                for patch in patches
            ]
        )
        patch_sum[name] = np.asarray(
            [
                residue_frequency[name][int(patch["start"]) - 1 : int(patch["end"])].sum()
                for patch in patches
            ]
        )
    target = Path(output_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        residue_frequency_backbone=residue_frequency["backbone"],
        residue_frequency_base=residue_frequency["base"],
        residue_frequency_any=residue_frequency["any"],
        contact_chain_count_backbone=np.asarray(chain_counts["backbone"]),
        contact_chain_count_base=np.asarray(chain_counts["base"]),
        contact_chain_count_any=np.asarray(chain_counts["any"]),
        chain_duplex_events_backbone=np.asarray(event_counts["backbone"]),
        chain_duplex_events_base=np.asarray(event_counts["base"]),
        chain_duplex_events_any=np.asarray(event_counts["any"]),
        chain_duplex_events_backbone_by_duplex=np.asarray(
            per_duplex_event_counts["backbone"]
        ),
        chain_duplex_events_base_by_duplex=np.asarray(
            per_duplex_event_counts["base"]
        ),
        chain_duplex_events_any_by_duplex=np.asarray(
            per_duplex_event_counts["any"]
        ),
        patch_residue_frequency_mean_backbone=patch_mean["backbone"],
        patch_residue_frequency_mean_base=patch_mean["base"],
        patch_residue_frequency_mean_any=patch_mean["any"],
        patch_residue_frequency_sum_backbone=patch_sum["backbone"],
        patch_residue_frequency_sum_base=patch_sum["base"],
        patch_residue_frequency_sum_any=patch_sum["any"],
        residue_frequency_blocks_backbone=np.asarray(block_frequencies["backbone"]).reshape(-1, 486),
        residue_frequency_blocks_base=np.asarray(block_frequencies["base"]).reshape(-1, 486),
        residue_frequency_blocks_any=np.asarray(block_frequencies["any"]).reshape(-1, 486),
    )
    summary = {
        "schema": "mecp2-dna-fd-protein-dna-metrics-v1",
        "status": "PASS",
        "analysis_scope": "selected_frames",
        "frames": frame_count,
        "chain_count": chain_count,
        "gamma": gamma,
        "contact_definition": "r_ij <= 1.5 * (sigma_i + sigma_j) / 2",
        "mean_contacting_chains": {
            name: float(np.mean(chain_counts[name])) for name in classes
        },
        "mean_chain_duplex_events": {
            name: float(np.mean(event_counts[name])) for name in classes
        },
        "mean_chain_duplex_events_by_duplex": {
            name: {
                str(duplex): float(
                    np.mean(np.asarray(per_duplex_event_counts[name])[:, duplex])
                )
                for duplex in range(3)
            }
            for name in classes
        },
        "patch_residue_frequency_mean": {
            name: {
                patch["name"]: float(patch_mean[name][index])
                for index, patch in enumerate(patches)
            }
            for name in classes
        },
        "patch_residue_frequency_sum": {
            name: {
                patch["name"]: float(patch_sum[name][index])
                for index, patch in enumerate(patches)
            }
            for name in classes
        },
        "any_dna_is_set_union": True,
        "interpretation_guard": "Geometric occupancy; not affinity, lifetime, bound-protein count, or bridging strength.",
        "npz_path": str(target),
    }
    write_json_atomic(output_json, summary)
    return summary
