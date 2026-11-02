# Independent trajectories

The manuscript mutation comparison uses two independent protein-only
trajectories per construct at 0.10 M salt. Replicates are named n1 and n2
throughout this package:

| Construct | First replicate | Second replicate | Compression |
|---|---|---|---|
| WT | n1 | n2 | legacy for both |
| R133C | n1 | n2 | legacy for both |
| T158M | n1 | n2 | legacy for both |
| P225R | n1 | n2 | legacy for both |
| R306C | n1 | n2 | legacy, adaptive |

Use `--protein-protocol adaptive` for R306C n2. All other entries use the
default `legacy` protocol. These preparation settings follow the two
trajectories selected for the manuscript. For R306C, public n2 corresponds to
the historical n3 trajectory.

Use distinct seeds and retain the replicate name and preparation protocol
with each new run. Use n1 and n2 in analysis manifests.
Independent trajectory means, temporal blocks and cross-replicate differences
have different roles and are not interchangeable replication units.
