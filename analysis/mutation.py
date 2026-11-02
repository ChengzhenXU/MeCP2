"""Rank native IDR-FD contact losses and recurrent top-six residues."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

MUTANTS = ("R133C", "T158M", "P225R", "R306C")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", type=Path, required=True,
                   help="native_idr_fd_profiles.csv produced by summarize.py")
    p.add_argument("--salt", type=float, default=0.1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    import numpy as np
    import pandas as pd
    profiles = pd.read_csv(args.profiles)
    profiles = profiles[profiles.background.eq("protein") &
                        np.isclose(profiles.salt_molar, args.salt)].copy()
    if set(profiles.construct) != {"WT", *MUTANTS}:
        raise ValueError("WT and all four mutants are required at the selected salt.")
    if profiles.duplicated(["construct", "replicate", "residue"]).any():
        raise ValueError("Duplicate residue record within a replicate.")
    for (_, _), group in profiles.groupby(["construct", "replicate"]):
        if set(group.residue) != set(range(91, 163)):
            raise ValueError("Each replicate must cover FD residues 91-162.")
    counts = profiles.groupby(["construct", "residue"]).size()
    if not counts.eq(2).all():
        raise ValueError("Each residue requires exactly two independent replicates.")
    means = profiles.groupby(["construct", "residue"]).contact.mean().unstack("construct")
    rows = []
    selected = {}
    for mutant in MUTANTS:
        delta = means[mutant] - means.WT
        order = sorted(delta.index, key=lambda residue: (delta[residue], residue))
        losses = [residue for residue in order if delta[residue] < 0]
        if len(losses) < 6:
            raise ValueError(f"{mutant} has fewer than six negative contact changes.")
        selected[mutant] = set(losses[:6])
        for rank, residue in enumerate(order, 1):
            rows.append(dict(construct=mutant, residue=residue, rank=rank,
                             wt_mean=means.WT[residue], mutant_mean=means[mutant][residue],
                             delta=delta[residue], is_top6=residue in selected[mutant]))
    occurrences = Counter(residue for chosen in selected.values() for residue in chosen)
    table = pd.DataFrame(rows)
    table["occurrence_count"] = table.residue.map(lambda residue: occurrences[residue])
    table["is_recurrent"] = table.occurrence_count.ge(2)
    table["is_marked"] = table.is_top6 & table.is_recurrent
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / "residue_rankings.csv", index=False)
    table[table.is_top6].to_csv(args.output / "top6.csv", index=False)
    table[table.is_recurrent].to_csv(args.output / "recurrent_membership.csv", index=False)


if __name__ == "__main__":
    main()
