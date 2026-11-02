"""Analyze MeCP2 trajectories without generating figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def identity_arrays(topology, construct, chains, duplexes, directory):
    import numpy as np
    import MDAnalysis as mda

    u = mda.Universe(str(topology))
    nprotein = chains * 486
    if nprotein > len(u.atoms) or chains < 2:
        raise ValueError("At least two complete 486-residue protein chains are required.")
    protein = u.atoms[:nprotein]
    if len(protein.residues) != nprotein or not np.all(protein.names == "CA"):
        raise ValueError("Topology must start with one CA bead per protein residue.")
    n_dna = len(u.atoms) - nprotein
    if (duplexes == 0 and n_dna != 0) or (duplexes == 3 and n_dna != 4512):
        raise ValueError("Expected protein-only topology or three 376-bp two-bead duplexes.")
    table_path = ROOT / "inputs/protein" / construct / "residues_CALVADOS3.csv"
    with table_path.open() as handle:
        table = {row["one"]: row for row in csv.DictReader(handle)}
    three = {row["three"]: key for key, row in table.items()}
    sequence = "".join(line.strip() for line in
                       (ROOT / "inputs/protein" / construct / "MeCP2.fasta").read_text().splitlines()
                       if not line.startswith(">"))
    expected = list(sequence) * chains
    observed = [three.get(name, name) for name in protein.resnames]
    if observed != expected:
        raise ValueError("Topology sequence does not match the selected construct.")
    sigma = np.full(len(u.atoms), 0.55)
    sigma[:nprotein] = np.tile([float(table[aa]["sigmas"]) for aa in sequence], chains)
    dna = np.arange(nprotein, len(u.atoms), dtype=np.int64)
    if duplexes and not np.all(u.atoms[dna[::2]].names == "BB"):
        raise ValueError("DNA beads must be ordered [backbone, base] within each strand.")
    if duplexes and not np.isin(u.atoms[dna[1::2]].names, list("ATCG")).all():
        raise ValueError("DNA base beads must be named A, T, C, or G.")
    duplex_by_global = np.full(len(u.atoms), -1, dtype=np.int16)
    if duplexes:
        duplex_by_global[dna] = np.repeat(np.arange(3), 1504)
    path = directory / "identity.npz"
    np.savez_compressed(path, protein_indices=np.arange(nprotein), dna_indices=dna,
                        base_indices=dna[1::2], sigma_nm=sigma,
                        duplex_by_global=duplex_by_global)
    return path


def radial_distribution(trajectory, identity, frame_slice, output):
    import numpy as np
    from analysis.core import ginter as g
    from analysis.core.metrics import _frames

    with np.load(identity) as z:
        indices = z["protein_indices"]
    chains = len(indices) // 486
    bins = np.linspace(0.0, 3.0, 61)
    midpoint = (bins[1:] + bins[:-1]) / 2
    shell = 4 * np.pi * midpoint**2 * np.diff(bins)
    total = np.zeros(60)
    patches = np.zeros((36, 60))
    inverse_volume = 0.0
    frames = 0
    for _, xyz, box in _frames(trajectory, frame_slice):
        h, hp = g.ginter_frame_histograms(xyz[indices], box, bins, chains, 486)
        total += h
        patches += hp
        inverse_volume += 1.0 / np.prod(box)
        frames += 1
    if not frames:
        raise ValueError("No frames were selected.")
    chain_pairs = math.comb(chains, 2)
    global_curve = g.normalize_histogram(total, chain_pairs * 486**2, shell, inverse_volume)
    curves = [g.normalize_histogram(hist, chain_pairs * g.possible_patch_pairs_per_chain_pair(pair),
                                    shell, inverse_volume)
              for hist, pair in zip(patches, g.PATCH_PAIRS)]
    weights = np.array([g.possible_patch_pairs_per_chain_pair(p) for p in g.PATCH_PAIRS])
    np.testing.assert_allclose(np.sum(np.asarray(curves) * weights[:, None], axis=0) / 486**2,
                               global_curve, rtol=1e-12, atol=1e-12)
    with (output / "ginter.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "r_nm", "g_inter"])
        for pair, values in [("ALL", global_curve)] + [
            ("-".join(pair), values) for pair, values in zip(g.PATCH_PAIRS, curves)
        ]:
            writer.writerows((pair, radius, value) for radius, value in zip(midpoint, values))


def dna_connectivity(trajectory, identity, frame_slice, output):
    import numpy as np
    from scipy.spatial import cKDTree
    from analysis.core.metrics import _frames

    with np.load(identity) as z:
        protein = z["protein_indices"]
        dna = z["dna_indices"]
        sigma = z["sigma_nm"]
        duplex = z["duplex_by_global"][dna]
    chains = len(protein) // 486
    rows = []
    for _, xyz, box in _frames(trajectory, frame_slice):
        px = np.mod(xyz[protein], box)
        dx = np.mod(xyz[dna], box)
        ptree = cKDTree(px, boxsize=box)
        pp = ptree.query_pairs(0.55, output_type="ndarray") // 486
        pp = pp[pp[:, 0] != pp[:, 1]]
        pd = ptree.sparse_distance_matrix(cKDTree(dx, boxsize=box),
                                         1.5 * (sigma[protein].max() + 0.55) / 2,
                                         output_type="coo_matrix")
        keep = pd.data <= 1.5 * (sigma[protein[pd.row]] + sigma[dna[pd.col]]) / 2
        pd_edges = np.column_stack((pd.row[keep] // 486, chains + duplex[pd.col[keep]]))

        def largest(edges, nodes):
            parent = np.arange(nodes)
            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for a, b in edges:
                parent[root(int(a))] = root(int(b))
            counts = np.bincount([root(i) for i in range(chains)], minlength=nodes)
            return int(counts.max())

        direct = largest(pp, chains)
        combined = largest(np.concatenate((pp, pd_edges)), chains + 3)
        if combined < direct:
            raise RuntimeError("Adding DNA edges reduced the largest component.")
        rows.append((direct, combined))
    values = np.asarray(rows)
    np.savez_compressed(output / "dna_connectivity.npz", largest_pp=values[:, 0],
                        largest_pp_dna=values[:, 1])
    summary = {name: {"mean_largest_fraction": float(values[:, i].mean() / chains),
                      "threshold_fraction_of_time": {
                          str(threshold): float(np.mean(values[:, i] / chains >= threshold))
                          for threshold in (0.4, 0.5, 0.6)}}
               for i, name in enumerate(("pp_only", "pp_and_dna"))}
    (output / "dna_connectivity.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--settings", type=Path, help="Simulation settings.json for salt and seed provenance.")
    p.add_argument("--construct", choices=("WT", "R133C", "T158M", "P225R", "R306C"), default="WT")
    p.add_argument("--chains", type=int, default=150)
    p.add_argument("--dna-duplexes", type=int, choices=(0, 3), default=0)
    p.add_argument("--start-frame", type=int, required=True, help="Zero-based inclusive index.")
    p.add_argument("--stop-frame", type=int, required=True, help="Zero-based exclusive index.")
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--block-ns", type=float, default=500.0)
    p.add_argument("--contact-engine", choices=("reference", "sparse"),
                   help="Default: full-pair reference without DNA, sparse 1e-8 kernel with DNA.")
    p.add_argument("--paper-window", action="store_true",
                   help="Require the final 3000 samples at 1 ns spacing, split into six 500 ns blocks.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.start_frame < 0 or args.stop_frame <= args.start_frame or args.stride < 1:
        p.error("Select a nonempty frame interval and a positive stride.")
    import numpy as np
    from analysis.core import metrics
    from analysis.core.ginter import PATCHES
    from analysis.core.io_utils import sha256_file
    with metrics._reader(args.trajectory) as reader:
        if args.stop_frame > len(reader):
            p.error(f"Stop frame exceeds trajectory length ({len(reader)}).")
        if args.paper_window and args.stop_frame != len(reader):
            p.error("The paper window must end at the final checkpoint-bounded trajectory frame.")
        interval_ns = reader.dt * args.stride / 1000
        selected_frames = len(range(args.start_frame, args.stop_frame, args.stride))
        first_time_ns = float(reader[args.start_frame].time) / 1000
        frame_times = [(index, first_time_ns + offset * interval_ns)
                       for offset, index in enumerate(range(args.start_frame, args.stop_frame, args.stride))]
    if not np.isfinite(interval_ns) or interval_ns <= 0 or not np.isfinite(args.block_ns) or args.block_ns <= 0:
        p.error("Sampling and block intervals must be finite and positive.")
    frames_per_block = int(round(args.block_ns / interval_ns))
    if frames_per_block < 1 or not np.isclose(frames_per_block * interval_ns, args.block_ns, rtol=1e-5):
        p.error("The block duration must be an integer number of sampled frames.")
    if args.paper_window and (selected_frames != 3000 or not np.isclose(interval_ns, 1, rtol=1e-5) or
                              not np.isclose(args.block_ns, 500)):
        p.error("The paper window requires 3000 frames at 1 ns and 500 ns blocks.")
    selection = slice(args.start_frame, args.stop_frame, args.stride)
    if args.contact_engine is None:
        args.contact_engine = "sparse" if args.dna_duplexes else "reference"
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = identity_arrays(args.topology, args.construct, args.chains,
                                args.dna_duplexes, args.output)
    with np.load(identity) as z:
        expected_atoms = len(z["sigma_nm"])
    with metrics._reader(args.trajectory) as reader:
        if reader.n_atoms != expected_atoms:
            raise ValueError("Trajectory and topology atom counts differ.")
    digest = sha256_file(args.trajectory)
    settings = None
    if args.settings:
        settings = json.loads(args.settings.read_text())
        if settings["construct"] != args.construct or bool(settings["dna_duplexes"]) != bool(args.dna_duplexes):
            raise ValueError("Simulation settings differ from the selected analysis system.")
    patches = [{"name": name, "start": start, "end": end} for name, start, end in PATCHES]
    metrics.cluster_metrics(args.trajectory, identity, args.output / "clusters.npz",
                             args.output / "clusters.json", dcd_sha256=digest,
                             cutoffs_nm=[0.5, 0.55, 0.6], frame_slice=selection)
    metrics.pp_contact_metrics(args.trajectory, identity, args.output / "contacts.npz",
                                args.output / "contacts.json", dcd_sha256=digest,
                                beta_per_nm=50.0, gamma=1.5, kernel_tolerance=1e-8,
                                patches=patches, construct=args.construct, frame_slice=selection,
                                frames_per_block=frames_per_block, contact_engine=args.contact_engine)
    radial_distribution(args.trajectory, identity, selection, args.output)
    if args.dna_duplexes:
        metrics.protein_dna_metrics(args.trajectory, identity, args.output / "protein_dna.npz",
                                     args.output / "protein_dna.json", dcd_sha256=digest,
                                     gamma=1.5, patches=patches, frame_slice=selection,
                                     frames_per_block=frames_per_block)
        dna_connectivity(args.trajectory, identity, selection, args.output)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(trajectory_sha256=digest, topology_sha256=sha256_file(args.topology),
                    selected_frames=selected_frames, sampled_interval_ns=interval_ns,
                    frames_per_block=frames_per_block, complete_blocks=selected_frames // frames_per_block)
    metadata["partial_block_frames"] = selected_frames % frames_per_block
    if args.trajectory.suffix.lower() == ".json":
        from analysis.core.segments import segment_records
        metadata["source_trajectory_sha256s"] = [sha256_file(item["path"])
                                                for item in segment_records(args.trajectory)]
    else:
        metadata["source_trajectory_sha256s"] = [digest]
    if args.settings:
        metadata.update(salt_molar=settings["salt"], simulation_seed=settings["seed"],
                        settings_sha256=sha256_file(args.settings))
    with (args.output / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_frame", "trajectory_time_ns"])
        writer.writerows(frame_times)
    (args.output / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
