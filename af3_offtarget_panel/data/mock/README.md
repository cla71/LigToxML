# Mock dataset for local practice

This directory contains a minimal, synthetic dataset for running the AF3/PDB off-target workflow end to end without downloading real FoldBench data.

## Contents
- `foldbench/`: CSVs that mirror FoldBench target lists and interfaces for three mock targets.
- `structures/pdb_assemblies/MOCK1.cif`: a tiny mmCIF with a ligand to enable pocket building.
- `inputs/compounds.smi`: a small SMILES batch for screening.

## Intended limitations
- Only `MOCK1` has a structure file; `MOCK2` will be skipped during pocket building.
- The docking scorer is a deterministic mock, so results are illustrative only.
