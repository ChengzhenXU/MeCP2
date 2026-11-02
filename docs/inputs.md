# Input conventions

`inputs/protein/` contains five independently supplied full-length structures:
WT, R133C, T158M, P225R, and R306C. Each directory contains:

- `MeCP2.pdb`: the initial all-atom protein reference structure.
- `MeCP2.fasta`: its 486-residue sequence.
- `domains.yaml`: the folded interval, residues 91-162 (inclusive).
- `residues_CALVADOS3.csv`: residue masses, diameters, charges, interaction
  parameters, and backbone bond lengths.

Protein coordinates are mapped to residue centers of mass using the bundled
CALVADOS code. The folded interval is restrained using its own construct's
reference structure. The shared file name `MeCP2.pdb` is required by the
component name; the enclosing directory identifies the construct.

`inputs/dna/sequence.txt` is the 376-nucleotide forward strand. The other strand
is its reverse complement. Each duplex contains 1,504 beads. Three identical
duplexes are included in the protein-DNA simulation.

Analysis expects 150 contiguous protein chains of 486 beads each, followed by
DNA when present. Each protein residue has one bead named `CA`. DNA is ordered
by duplex, strand A, strand B, and then `[BB, base]` for each nucleotide. The
base names are `A`, `T`, `C`, and `G`. Both DNA strands are stored in 5'-to-3'
sequence order. Each duplex forms a single connectivity node.

Only orthorhombic periodic boxes are supported. Coordinates read from DCD
files are converted from angstroms to nanometers. No trajectory or numerical
result is distributed with the source.
