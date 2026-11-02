# Protein mutation series

The main mutation comparison uses 150 protein chains without DNA at 0.10 M
salt. The constructs are WT, R133C, T158M, P225R, and R306C. Each construct
uses its own initial structure and sequence under `inputs/protein/`.

```bash
python simulation/mutation/run.py --construct R133C --salt 0.10 --seed 301 --output runs/R133C_100mM_n1
python simulation/mutation/run.py --construct R133C --salt 0.10 --seed 302 --output runs/R133C_100mM_n2
```

Repeat for the remaining mutants and WT using distinct seeds and output
directories. Production is 5 microseconds with a 0.5 ns write interval.
Name the two independent replicates n1 and n2 for every construct.
R306C n2 uses adaptive compression:

```bash
python simulation/mutation/run.py --construct R306C --salt 0.10 --seed 401 --output runs/R306C_100mM_n1
python simulation/mutation/run.py --construct R306C --salt 0.10 --seed 402 --protein-protocol adaptive --output runs/R306C_100mM_n2
```

See `docs/replicates.md` for the preparation protocol of each replicate.
Pass the same `--construct` value to `analysis/run.py`. Use
`analysis/summarize.py` for replicate statistics and `analysis/mutation.py`
for native IDR-FD loss rankings and recurrent top-six residues.
