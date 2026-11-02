# Numerical analysis

`run.py` analyzes one topology and a DCD or `trajectories.json` segment manifest. It calculates protein clusters,
continuous residue contacts, region contacts, and interchain radial
distributions. `--dna-duplexes 3` also enables DNA occupancy and mixed
protein-DNA connectivity. No figures are generated.

Use explicit frame indices. For 0.5 ns output, the standard 3,000-frame
selection is `--start-frame 4000 --stop-frame 10000 --stride 2`. For 1 ns
output it is `--start-frame 2000 --stop-frame 5000 --stride 1`. These settings
apply only to trajectories of the corresponding length and write interval.
Use `--paper-window` to require the final 3,000 samples at 1 ns intervals and
six 500 ns blocks. DCD intervals are read from their headers. `frames.csv` records source
indices and trajectory times. Segment manifests check gaps, overlaps, atom
counts, header intervals and checkpoint-bounded frame counts without copying
or concatenating DCD files. Pass `--settings runs/example/settings.json` to
bind salt and seed metadata to an analysis.
Protein-only contacts default to full-pair reference calculations; these can
be expensive for 150 chains. `--contact-engine sparse` selects the validated
1e-8-tail approximation explicitly. DNA analysis defaults to that sparse
kernel. The reference retains the original algebra with float64 arithmetic;
the original Torch calculation used float32. Neither mode promises byte-identical
historical output matrices.

`summarize.py` requires a user-created CSV manifest with these columns:

```text
directory,construct,salt_molar,background,replicate
```

Each row names one independent trajectory analysis directory. Relative paths
are resolved from the manifest's directory. Background is `protein` or
`protein_dna`; use exactly two distinct replicates per condition. Do not list
different time blocks of a single trajectory as separate replicates.

```bash
python analysis/summarize.py --manifest analysis_manifest.csv --paper-statistics --output outputs/summary
python analysis/mutation.py --profiles outputs/summary/native_idr_fd_profiles.csv \
  --salt 0.10 --output outputs/mutation
```

The mutation ranking selects the six most negative native mutant-minus-WT
mean contact changes per mutant. A residue is recurrent if it occurs in at
least two mutant top-six sets. Ties are resolved by residue index. The native
IDR-FD profile is `2 * qij[0:90, 90:162].sum(axis=0)`; it retains each
construct's own residue diameters. This procedure is descriptive and does
not turn four cross-replicate differences into four independent replicates.

`trajectory_metrics.csv` and `condition_metrics.csv` provide trajectory means
and the sample SD across two independent trajectories. `idr_fd_contact` is
the one-direction block; `idr_fd_complete_pair_contact` is twice that value.
`paper_block_observables.csv` provides the paper's separate mean and sample SD
over twelve nested 500 ns blocks, while retaining `independent_n=2`.
`temporal_block_observables.csv` preserves each block's value and trajectory ID.
Incomplete final blocks are recorded in analysis metadata and are rejected by
`--paper-statistics`.

Additional numerical outputs include per-trajectory and condition-level
observables, all cutoff cluster distributions, regional contact row sums and
cells, full-pair contacts, DNA residue profiles and regional residue counts,
mixed connectivity, whole-protein and 36 regional RDF curves, condition contact
matrices, and signed high-minus-low-salt and mutant-minus-WT contrasts.
Cluster distribution tables distinguish the equal-trajectory mean probability
from the pooled event probability. No error bars are calculated from the four
cross-replicate mutant-WT differences.

Sequence-only numerical values for the model description are generated with:

```bash
python analysis/sequence.py --construct WT --output outputs/sequence
```
