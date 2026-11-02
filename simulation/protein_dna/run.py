"""Run WT MeCP2 with three 376-bp two-bead DNA duplexes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.common.workflow import main

if __name__ == "__main__":
    main("protein_dna")
