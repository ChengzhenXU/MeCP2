"""Condition means, nested temporal-block statistics, and signed contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core.ginter import PATCHES

KEYS = ["background", "construct", "salt_molar"]


def aggregate_details(records, manifest, output, require_paper=False):
    values, blocks, curves, probabilities, dna_profiles = [], [], [], [], []
    matrices = {}

    for record in records:
        directory = Path(record["directory"])
        if not directory.is_absolute():
            directory = manifest.resolve().parent / directory
        metadata = json.loads((directory / "analysis.json").read_text())
        base = {key: record[key] for key in KEYS + ["replicate"]}
        base["salt_molar"] = float(base["salt_molar"])
        nframes = metadata["selected_frames"]
        size = metadata.get("frames_per_block", nframes + 1)
        nblocks = nframes // size
        if require_paper and (nframes != 3000 or nblocks != 6 or nframes % size or
                              not np.isclose(metadata.get("sampled_interval_ns", 0), 1, rtol=1e-5) or
                              not np.isclose(metadata.get("block_ns", 0), 500)):
            raise ValueError("Paper statistics require six 500 ns blocks per independent trajectory.")

        def add(metric, value, block_values=None):
            values.append(dict(base, metric=metric, value=float(value)))
            if block_values is not None:
                if len(block_values) != nblocks:
                    raise ValueError(f"Block count differs for {metric}.")
                blocks.extend(dict(base, block=index + 1, metric=metric, value=float(item))
                              for index, item in enumerate(block_values))

        def block_mean(series):
            return np.asarray(series)[:nblocks * size].reshape(nblocks, size).mean(axis=1) if nblocks else []

        with np.load(directory / "clusters.npz") as data:
            for index, cutoff in enumerate(data["cutoffs_nm"]):
                series = data["largest_cluster_by_frame"][:, index] / metadata["chains"]
                prefix = f"pp_{cutoff:.2f}"
                add(prefix + ":largest_fraction", series.mean(), block_mean(series))
                for threshold in (0.4, 0.5, 0.6):
                    occupied = series >= threshold
                    add(prefix + f":fraction_ge_{threshold:.1f}", occupied.mean(), block_mean(occupied))
                counts = data["cluster_event_counts"][index]
                probs = data["cluster_event_probability"][index]
                probabilities.extend(dict(base, cutoff_nm=float(cutoff), cluster_size=s,
                                          events=int(count), probability=float(probs[s]))
                                     for s, count in enumerate(counts) if s > 0)
        with np.load(directory / "contacts.npz") as data:
            qij = data["qij"]
            qb = data["qij_blocks"] if "qij_blocks" in data else np.empty((0, 486, 486))
            if len(qb) != nblocks:
                raise ValueError("Contact block matrices are missing; rerun analysis with block output.")
            add("pp_total_contact", qij.sum(), qb.sum(axis=(1, 2)))
            add("idr_fd_one_direction", qij[:90, 90:162].sum(), qb[:, :90, 90:162].sum(axis=(1, 2)))
            for i, (name, start, end) in enumerate(PATCHES):
                add("pp_row:" + name, qij[start - 1:end].sum(), qb[:, start - 1:end].sum(axis=(1, 2)))
                for other, lo, hi in PATCHES:
                    cell = qij[start - 1:end, lo - 1:hi].sum()
                    add("pp_cell:" + name + ":" + other, cell)
                    add("pp_complete_pair:" + name + ":" + other, cell * (1 if name == other else 2))
            key = tuple(base[k] for k in KEYS)
            matrices.setdefault(key, []).append(qij.copy())
        curve = pd.read_csv(directory / "ginter.csv")
        for key, value in base.items():
            curve[key] = value
        curves.append(curve)
        if metadata["dna_duplexes"]:
            with np.load(directory / "dna_connectivity.npz") as data:
                for name, array in (("pp_only", "largest_pp"), ("pp_and_dna", "largest_pp_dna")):
                    series = data[array] / metadata["chains"]
                    add(name + ":largest_fraction", series.mean(), block_mean(series))
                    for threshold in (0.4, 0.5, 0.6):
                        occupied = series >= threshold
                        add(name + f":fraction_ge_{threshold:.1f}", occupied.mean(), block_mean(occupied))
            with np.load(directory / "protein_dna.npz") as data:
                for kind in ("any", "backbone", "base"):
                    profile = data["residue_frequency_" + kind]
                    fb = data["residue_frequency_blocks_" + kind]
                    add("dna_" + kind + ":occupancy_percent", profile.mean() * 100, fb.mean(axis=1) * 100)
                    for name, start, end in PATCHES:
                        add("dna_" + kind + ":count:" + name, profile[start - 1:end].sum(),
                            fb[:, start - 1:end].sum(axis=1))
                    dna_profiles.extend(dict(base, bead_class=kind, residue=index + 1, frequency=float(value))
                                        for index, value in enumerate(profile))

    def stats(table, keys, value, nested=False):
        result = table.groupby(keys)[value].agg(["mean", "std", "count"]).reset_index()
        independent = table.groupby(keys).replicate.nunique()
        if not independent.eq(2).all() or (not nested and not result["count"].eq(2).all()):
            raise ValueError("Every aggregated observable requires two distinct independent trajectories.")
        result["independent_n"] = 2
        return result

    table = pd.DataFrame(values)
    table.to_csv(output / "trajectory_observables.csv", index=False)
    condition = stats(table, KEYS + ["metric"], "value")
    condition["sd_definition"] = "sample_SD_of_two_independent_trajectory_means"
    condition.to_csv(output / "condition_observables.csv", index=False)
    if blocks:
        block_table = pd.DataFrame(blocks)
        block_table.to_csv(output / "temporal_block_observables.csv", index=False)
        summary = stats(block_table, KEYS + ["metric"], "value", nested=True).rename(columns={"count": "n_temporal_blocks"})
        if require_paper and not summary.n_temporal_blocks.eq(12).all():
            raise ValueError("Each paper statistic requires twelve nested temporal blocks.")
        summary["sd_definition"] = "sample_SD_of_temporal_blocks_nested_within_two_trajectories"
        summary.to_csv(output / "paper_block_observables.csv", index=False)
    curve_table = pd.concat(curves, ignore_index=True)
    curve_table.to_csv(output / "trajectory_ginter.csv", index=False)
    stats(curve_table, KEYS + ["pair", "r_nm"], "g_inter").to_csv(output / "condition_ginter.csv", index=False)
    probability_table = pd.DataFrame(probabilities)
    probability_table.to_csv(output / "trajectory_cluster_distribution.csv", index=False)
    probability_summary = stats(probability_table, KEYS + ["cutoff_nm", "cluster_size"], "probability")
    event_sums = probability_table.groupby(KEYS + ["cutoff_nm", "cluster_size"]).events.sum().reset_index()
    event_sums["pooled_event_probability"] = event_sums.events / event_sums.groupby(KEYS + ["cutoff_nm"]).events.transform("sum")
    probability_summary.merge(event_sums).to_csv(output / "condition_cluster_distribution.csv", index=False)
    if dna_profiles:
        profile_table = pd.DataFrame(dna_profiles)
        profile_table.to_csv(output / "trajectory_dna_profiles.csv", index=False)
        stats(profile_table, KEYS + ["bead_class", "residue"], "frequency").to_csv(output / "condition_dna_profiles.csv", index=False)
    contrasts = []
    for (background, construct, metric), group in condition.groupby(["background", "construct", "metric"]):
        low = group[np.isclose(group.salt_molar, 0.01)]
        high = group[np.isclose(group.salt_molar, 0.3)]
        if len(low) == len(high) == 1:
            contrasts.append(dict(background=background, construct=construct, metric=metric,
                                  contrast="0.30M_minus_0.01M", delta=float(high["mean"].iloc[0] - low["mean"].iloc[0])))
    for (background, salt, metric), group in condition.groupby(["background", "salt_molar", "metric"]):
        wt = group[group.construct.eq("WT")]
        if len(wt) == 1:
            for _, mutant in group[~group.construct.eq("WT")].iterrows():
                contrasts.append(dict(background=background, construct=mutant.construct, salt_molar=salt,
                                      metric=metric, contrast="mutant_minus_WT",
                                      delta=float(mutant["mean"] - wt["mean"].iloc[0])))
    pd.DataFrame(contrasts).to_csv(output / "observable_contrasts.csv", index=False)
    destination = output / "condition_contacts"
    destination.mkdir()
    for (background, construct, salt), arrays in matrices.items():
        np.savez_compressed(destination / f"{background}_{construct}_{salt:g}M.npz", qij=np.mean(arrays, axis=0), independent_n=2)
