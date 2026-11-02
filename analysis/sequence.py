"""Calculate regional composition and centered sequence-charge windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.core.ginter import PATCHES


def charge(residue):
    return 1 if residue in "KR" else -1 if residue in "DE" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct", choices=("WT", "R133C", "T158M", "P225R", "R306C"), default="WT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sequence = "".join(line.strip() for line in
                       (ROOT / "inputs/protein" / args.construct / "MeCP2.fasta").read_text().splitlines()
                       if not line.startswith(">"))
    if len(sequence) != 486:
        raise ValueError("Expected a 486-residue sequence.")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "regional_charge.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["region", "start", "end", "length", "positive_KR", "negative_DE", "net_charge"])
        for name, start, end in PATCHES:
            segment = sequence[start - 1:end]
            positive = sum(aa in "KR" for aa in segment)
            negative = sum(aa in "DE" for aa in segment)
            writer.writerow([name, start, end, len(segment), positive, negative, positive - negative])
    with (args.output / "local_charge.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["residue", "aa", "half_window", "start", "end", "actual_length", "mean_charge"])
        for half_window in (3, 10):
            for index, aa in enumerate(sequence, 1):
                start, end = max(1, index - half_window), min(len(sequence), index + half_window)
                values = [charge(residue) for residue in sequence[start - 1:end]]
                writer.writerow([index, aa, half_window, start, end, len(values), sum(values) / len(values)])


if __name__ == "__main__":
    main()
