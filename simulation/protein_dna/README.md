# WT protein with DNA

150 WT MeCP2 chains and three identical 376-bp double-stranded DNA molecules.
Run two independent replicates at 0.01, 0.10, and 0.30 M salt.

```bash
python simulation/protein_dna/run.py --salt 0.10 --seed 201 --output runs/protein_dna_100mM_n1
python simulation/protein_dna/run.py --salt 0.10 --seed 202 --output runs/protein_dna_100mM_n2
```

The default production length is 5 microseconds and the write interval is
1 ns. Protein and DNA parameters are assembled from the bundled inputs.
Analyze this system with `--dna-duplexes 3`.
