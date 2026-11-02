# WT protein

150 full-length WT MeCP2 chains without DNA. Run two independent replicates
at each salt concentration: 0.01, 0.10, and 0.30 M.

```bash
python simulation/protein/run.py --salt 0.01 --seed 101 --output runs/protein_10mM_n1
python simulation/protein/run.py --salt 0.01 --seed 102 --output runs/protein_10mM_n2
```

The default production length is 5 microseconds and the write interval is
0.5 ns. Initial structures and force-field inputs are under `inputs/protein/WT`.
See the root README for environment setup and analysis commands.
