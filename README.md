# MeCP2 simulations and analysis

Coarse-grained molecular dynamics of full-length human MeCP2 using CALVADOS3,
with a Kapoor two-bead DNA model for the protein-DNA system.

## Layout

```text
simulation/
  protein/       WT protein-only simulations at 10, 100, and 300 mM salt
  protein_dna/   WT protein with three 376-bp DNA duplexes
  mutation/      WT, R133C, T158M, P225R, and R306C protein-only comparisons
  common/        Shared model construction and preparation code
  vendor/        Required CALVADOS simulation modules
analysis/        Clusters, contacts, DNA connectivity, radial distributions
inputs/          Initial protein PDBs, sequences, domain definitions, parameters
tests/           Numerical and workflow checks
docs/            Methods, input conventions, and replicate definitions
```

The folder contains source code and initial model inputs. Trajectories,
checkpoints, previously calculated observables, and figure code are excluded.
The CSV files under `inputs/` are residue force-field parameters.

## Environment

Use Python 3.10-3.12 and a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

OpenMM selects an available platform automatically. The optional `--platform`
argument selects an installed OpenMM platform by name. Platforms exposing a
precision setting use mixed precision. The default production length is
5 microseconds.

## Simulation

Run these commands from this directory, choosing a new output directory for
each independent replicate:

```bash
python simulation/protein/run.py --salt 0.10 --seed 101 --output runs/protein_100mM_n1
python simulation/protein_dna/run.py --salt 0.10 --seed 201 --output runs/protein_dna_100mM_n1
python simulation/mutation/run.py --construct R133C --salt 0.10 --seed 301 --output runs/R133C_100mM_n1
```

Repeat the WT systems at 0.01, 0.10, and 0.30 M. Use two independently seeded
replicates per condition. The main mutation comparison uses protein-only
systems at 0.10 M, including WT as the reference. `--build-only` constructs
the initial system without running minimization or dynamics. Fresh output
directories must not already exist. Use `--protein-protocol adaptive` for the
R306C n2; both replicates are named n1 and n2 for every construct. See
[replicate definitions](docs/replicates.md).

Preparation and production publish atomic checkpoint generations. Continue a
run with the matching entry point and its existing directory:

```bash
python simulation/protein/run.py --resume runs/protein_100mM_n1
```

The saved settings, System, integrator and checkpoint are validated before
loading. Resume uses the recorded platform and requires a compatible OpenMM
environment. `--steps` specifies the total production target, including steps
already completed. Each continuation writes a new DCD segment;
`trajectories.json` records safe frame counts and production-relative times.

## Analysis

Frame selection is explicit and uses zero-based indices with an exclusive stop.
For a protein trajectory containing 10,000 frames at 0.5 ns intervals:

```bash
python analysis/run.py \
  --topology runs/protein_100mM_n1/topology.pdb \
  --trajectory runs/protein_100mM_n1/trajectories.json \
  --settings runs/protein_100mM_n1/settings.json --paper-window \
  --start-frame 4000 --stop-frame 10000 --stride 2 \
  --output outputs/protein_100mM_n1
```

For a protein-DNA trajectory containing 5,000 frames at 1 ns intervals:

```bash
python analysis/run.py \
  --topology runs/protein_dna_100mM_n1/topology.pdb \
  --trajectory runs/protein_dna_100mM_n1/trajectories.json \
  --settings runs/protein_dna_100mM_n1/settings.json --paper-window \
  --dna-duplexes 3 \
  --start-frame 2000 --stop-frame 5000 --stride 1 \
  --output outputs/protein_dna_100mM_n1
```

For mutants, also supply `--construct R133C` (or the corresponding variant).
The analysis writes numerical arrays and tables. See [methods](docs/methods.md)
for definitions and [input conventions](docs/inputs.md) for particle ordering.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover force-field kernels, preparation geometry, numerical analysis, and
checkpoint continuation using small synthetic systems. They do not validate
full production trajectories. New stochastic runs are not expected to reproduce
historical trajectories byte for byte.

See [third-party code and model references](THIRD_PARTY_NOTICES.md).
