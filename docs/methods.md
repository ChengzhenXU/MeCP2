# Model and analysis definitions

## Simulation

Each system contains 150 chains of 486 residues. The protein force field is
CALVADOS3, with a harmonic elastic network over residues 91-162. Network
parameters are 700 kJ mol^-1 nm^-2 and a 0.9 nm reference-distance cutoff.
Backbone connectivity, terminal corrections, and nonbonded exclusions use the
bundled CALVADOS implementation and the construct-specific PDB.

Protein-only systems start in a 50 x 50 x 150 nm box, undergo minimization,
10 million pre-equilibration steps, z compression, and final minimization.
The actual protein-only compression pressure is 1 bar along z, with attempts
every 25 steps. Historical default-pressure updates did not change an existing
OpenMM Context. The default protocol checks z after each 1,000 steps; the
R306C n2 uses 1,000/250/50/25-step batches above
100/70/52/50 nm. Final all-bead z scaling is followed by minimization with a
100 kJ mol^-1 nm^-1 tolerance. Initial protein velocities are zero. Placement
uses the NumPy MT19937 generator and the PDB coordinate round trip.
Protein-DNA systems start in a 50 x 50 x 500 nm box. Their preparation uses
the reference staged warmup (50, 100, 200, and 310 K), pre-equilibration,
z compression, six molecule-preserving box corrections, post-relaxation, and
1 ns of NVT equilibration. The production box is 50 x 50 x 50 nm.
The compression and clean NVT stages each create a fresh Context. Only the
box, coordinates and velocities are transferred; integrator random-number
state and the preceding Context clock are not copied. Production starts
after the final clock reset.
DNA compression applies an actual Context pressure of 1, 2, ..., 1,000 bar
before successive 1,000-step blocks. The final 0.5 ns mean temperature must
lie within 305-315 K before production is allowed.
DNA acceptance checks also retain the original maximum bond lengths
(1.0 nm for protein-chain bonds; 0.85 nm for DNA backbone and backbone-base
bonds), force counts, finite state values, fixed-box geometry, and the
10 kJ mol^-1 nm^-1 final minimization RMS-force limit.

Production uses a LangevinMiddle integrator at 310 K, a 0.01 ps step, and
0.01 ps^-1 friction. The protein-only barostat is disabled before production;
the DNA preparation barostat is removed. The default
length is 500 million steps (5 microseconds). The default DCD interval is
0.5 ns without DNA and 1 ns with DNA. Reporters start after preparation.
DNA and adaptive-protocol protein runs reset the production clock to zero.
The legacy protein protocol retains its preparation step/time offset; the
segment manifest records the production origin separately. The target length
always counts production steps only.
State logs use 1,000,000-step intervals for protein-only runs and 10,000-step
intervals with DNA. Checkpoints are published at each DCD interval; this is
more frequent than the legacy protein runner's 50,000,000-step checkpoints
and does not change the integration or analysis sampling interval.

The DNA builder retains the Kapoor two-bead masses, bond and angle terms,
WCA repulsion, stacking, complementary-base interactions, and protein-DNA
coupling. DD and PD electrostatics use the CALVADOS temperature-dependent
screening parameters. The selected salt concentration is in mol/L.

The runner exposes an explicit seed: placement uses that seed; DNA translation,
orientation and placement use seed+1,+2,+3; preparation integrator, DNA velocity
initialization and barostat use seed+4,+5,+6; the clean DNA NVT integrator uses
seed+7. Historical seeds were separate registry entries, and some legacy
protein seeds were unset. These are fresh stochastic reproductions rather
than replays of historical random-number streams. Checkpoints retain the
integrator and random-number state for continuation on a compatible platform.

## Analysis

The standard analysis window contains the last 3 microseconds, sampled every
1 ns. Select frames using the actual write interval of the supplied trajectory.
Do not infer the interval from a file name. The stop frame is exclusive.

Direct protein clusters are connected components of protein-chain nodes. An
edge exists if any interchain residue pair lies within the selected cutoff.
Cutoffs of 0.50, 0.55, and 0.60 nm are evaluated; 0.55 nm is the primary value.
Cluster threshold occupancies use 60, 75, and 90 proteins out of 150.

For protein-DNA connectivity, each whole duplex adds one node. Protein-DNA
edges use `r <= 1.5 * (sigma_protein + 0.55) / 2`. All direct PP edges are
retained. Component size counts only protein nodes.

Continuous PP contact weights are
`w(r) = 1 / (1 + exp(50 * (r - 1.5 * sigma_ij)))`, with distances in nm and
`sigma_ij = (sigma_i + sigma_j) / 2`. The symmetric residue matrix averages
over sampled frames and unordered chain pairs. Protein-only analysis defaults
to the original NumPy full-pair reference algebra, with no distance-tail
truncation. The original contact calculation used float32 Torch arithmetic;
the portable reference uses float64, so bitwise-identical matrices are not
claimed. DNA analysis retains the original sparse kernel with a numerical
tail tolerance of 1e-8. `--contact-engine sparse` also allows an explicitly
selected faster approximation for protein-only data. Region matrices sum
residue entries. These values describe geometric contact weights.

The eight inclusive sequence intervals are IDR (1-90), FD (91-162), P1
(163-205), M1 (206-241), P2 (242-313), M2 (314-375), N1 (376-409), and CTR
(410-486). CTR replaces the historical display name M3 without changing its
residue interval.

Interchain radial distributions use 60 bins of width 0.05 nm over 0-3 nm,
midpoint shell volumes `4*pi*r_mid^2*dr`, and the sum of inverse frame volumes.
The opportunity count is `choose(N, 2)*L_A^2` for equal regions and
`2*choose(N, 2)*L_A*L_B` for distinct regions. All curves include only
interchain protein residue pairs, including in DNA-containing systems.

The independent trajectory count is two per condition. Each final 3 microsecond
window is divided into six contiguous 500 ns blocks. Paper bar heights and
error bars use the mean and sample SD (ddof=1) of the twelve nested block
statistics. This SD describes temporal-block variability; it is not SEM,
a confidence interval, or the SD of two independent trajectory means.
The latter SD is exported separately. Curves and contact matrices average
the two trajectories equally. Contrasts subtract condition means without
treating cross-replicate differences as additional independent observations.

The scalar IDR-FD contact is the one-direction matrix block
`qij[0:90,90:162].sum()`. Native FD profiles use
`2*qij[0:90,90:162].sum(axis=0)` and retain construct-specific diameters.
Regional DNA occupancy counts are sums of residue frequencies, without
division by region length. Whole-protein DNA occupancy is the sum over all
486 positions divided by 486 and multiplied by 100. High-minus-low salt
contrasts use 0.30 M minus 0.01 M. Repeated contacts to the same residue
within a frame count once.

Sequence-charge descriptions assign +1 to K/R, -1 to D/E, and zero otherwise.
Regional counts and centered half-windows of 3 and 10 residues use the actual
truncated window length at sequence ends. These descriptive charges are
separate from simulation charges and terminal corrections.
