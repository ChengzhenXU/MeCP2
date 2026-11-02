"""Summarize independent trajectory analyses and mutation contrasts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV columns: directory, construct, salt_molar, background, replicate.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-statistics", action="store_true",
                        help="Require six 500 ns blocks per trajectory for paper error bars.")
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    with args.manifest.open() as handle:
        records = list(csv.DictReader(handle))
    required = {"directory", "construct", "salt_molar", "background", "replicate"}
    if not records or not required.issubset(records[0]):
        parser.error("Manifest is empty or lacks required columns.")
    rows = []
    profiles = []
    seen_trajectories = set()
    seen_seeds = set()
    for record in records:
        directory = Path(record["directory"])
        if not directory.is_absolute():
            directory = args.manifest.resolve().parent / directory
        metadata = json.loads((directory / "analysis.json").read_text())
        digests = set(metadata.get("source_trajectory_sha256s", [metadata["trajectory_sha256"]]))
        if digests & seen_trajectories:
            raise ValueError("The same trajectory cannot count as two independent replicates.")
        seen_trajectories.update(digests)
        if record["construct"] != metadata["construct"]:
            raise ValueError("Manifest construct differs from analysis metadata.")
        if record["background"] not in ("protein", "protein_dna"):
            raise ValueError("Background must be protein or protein_dna.")
        if (record["background"] == "protein_dna") != bool(metadata["dna_duplexes"]):
            raise ValueError("Manifest DNA background differs from analysis metadata.")
        base = {key: record[key] for key in required - {"directory"}}
        base["salt_molar"] = float(base["salt_molar"])
        if not any(np.isclose(base["salt_molar"], salt, rtol=0, atol=1e-8) for salt in (0.01, 0.1, 0.3)):
            raise ValueError("Salt must be 0.01, 0.10, or 0.30 M.")
        if "salt_molar" in metadata and not np.isclose(base["salt_molar"], metadata["salt_molar"]):
            raise ValueError("Manifest salt differs from simulation settings.")
        if "simulation_seed" in metadata:
            seed_key = (base["background"], base["construct"], base["salt_molar"], metadata["simulation_seed"])
            if seed_key in seen_seeds:
                raise ValueError("Independent replicates of a condition cannot reuse the same simulation seed.")
            seen_seeds.add(seed_key)
        with np.load(directory / "clusters.npz") as z:
            index = int(np.flatnonzero(np.isclose(z["cutoffs_nm"], 0.55))[0])
            largest = z["largest_cluster_by_frame"][:, index]
            row = dict(base, mean_largest_fraction=float(largest.mean() / metadata["chains"]))
            row.update({f"fraction_ge_{threshold}": float(np.mean(largest >= threshold))
                        for threshold in (60, 75, 90)})
        with np.load(directory / "contacts.npz") as z:
            qij = z["qij"]
            row["total_contact"] = float(qij.sum())
            native_profile = 2 * qij[:90, 90:162].sum(axis=0)
            row["idr_fd_contact"] = float(qij[:90, 90:162].sum())
            row["idr_fd_complete_pair_contact"] = float(native_profile.sum())
            profiles.extend(dict(base, residue=residue, contact=float(value))
                            for residue, value in zip(range(91, 163), native_profile))
        rows.append(row)
    table = pd.DataFrame(rows)
    keys = ["background", "construct", "salt_molar"]
    if table.duplicated(keys + ["replicate"]).any():
        raise ValueError("Duplicate replicate in manifest.")
    counts = table.groupby(keys).size()
    if not counts.eq(2).all():
        raise ValueError("Each condition requires exactly two independent replicates.")
    metrics = [key for key in table if key not in keys + ["replicate"]]
    summary = table.groupby(keys)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    means = table.groupby(keys)[metrics].mean().reset_index()
    wt = means[means.construct.eq("WT")].drop(columns="construct")
    mutants = means[~means.construct.eq("WT")]
    contrasts = mutants.merge(wt, on=["background", "salt_molar"], suffixes=("_mutant", "_wt"),
                              validate="many_to_one")
    if len(contrasts) != len(mutants):
        raise ValueError("A mutation condition lacks a matching WT reference.")
    for metric in metrics:
        contrasts[metric + "_delta"] = contrasts[metric + "_mutant"] - contrasts[metric + "_wt"]
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / "trajectory_metrics.csv", index=False)
    summary.reset_index().to_csv(args.output / "condition_metrics.csv", index=False)
    pd.DataFrame(profiles).to_csv(args.output / "native_idr_fd_profiles.csv", index=False)
    profile_table = pd.DataFrame(profiles)
    profile_table.groupby(keys + ["residue"]).contact.agg(["mean", "std", "count"]).reset_index().to_csv(
        args.output / "native_idr_fd_condition_profiles.csv", index=False)
    contrasts.to_csv(args.output / "mutation_contrasts.csv", index=False)
    from analysis.aggregation import aggregate_details
    aggregate_details(records, args.manifest, args.output, require_paper=args.paper_statistics)


if __name__ == "__main__":
    main()
