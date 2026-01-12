from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


@dataclass
class PocketDefinition:
    pdb_assembly_id: str
    protein_chain_id: str
    ligand_id: str
    center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    residues: list[str]
    receptor_path: str
    qc_flags: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


PARSER = MMCIFParser(QUIET=True)


def _load_structure(structure_path: Path):
    return PARSER.get_structure(structure_path.stem, structure_path)


def _extract_ligand_fields(row: pd.Series) -> dict:
    mapping = {
        "pdb_assembly_id": ["pdb_assembly_id", "pdb_id", "assembly_id"],
        "protein_chain_id": ["protein_chain", "protein_chain_id", "protein_asym_id"],
        "ligand_id": ["ligand_id", "ligand_comp_id"],
        "ligand_chain_id": ["ligand_chain_id", "ligand_asym_id", "ligand_chain"],
        "ligand_resseq": ["ligand_residue_number", "ligand_resseq"],
    }
    result = {}
    for key, candidates in mapping.items():
        for col in candidates:
            if col in row:
                result[key] = row[col]
                break
    required = ["pdb_assembly_id", "protein_chain_id", "ligand_id"]
    for key in required:
        if key not in result:
            raise KeyError(f"Missing required column for interfaces: {key}")
    return result


def _find_chain(structure, chain_id: str) -> Optional[Chain]:
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return chain
    return None


def _find_ligand(chain: Chain, ligand_chain_id: Optional[str], ligand_id: str, ligand_resseq: Optional[int]) -> Optional[Residue]:
    target_chain_id = ligand_chain_id or chain.id

    for residue in chain:
        if residue.id[0].strip() and residue.id[0] != "W":
            same_chain = (ligand_chain_id is None) or (chain.id == target_chain_id)
            matches_resname = residue.resname.strip() == ligand_id
            matches_resseq = True if ligand_resseq is None else residue.id[1] == ligand_resseq
            if same_chain and matches_resname and matches_resseq:
                return residue

    if ligand_chain_id and ligand_chain_id != chain.id:
        chain_obj = _find_chain(chain.get_parent(), ligand_chain_id)
        if chain_obj:
            for residue in chain_obj:
                if residue.id[0].strip() and residue.resname.strip() == ligand_id:
                    matches_resseq = True if ligand_resseq is None else residue.id[1] == ligand_resseq
                    if matches_resseq:
                        return residue
    return None


def _centroid(coords: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    coord_list = list(coords)
    count = len(coord_list)
    if count == 0:
        return 0.0, 0.0, 0.0
    xs, ys, zs = zip(*coord_list)
    return sum(xs) / count, sum(ys) / count, sum(zs) / count


def _bounding_box(coords: Iterable[tuple[float, float, float]], padding: float = 6.0) -> tuple[float, float, float]:
    xs, ys, zs = zip(*coords)
    span = (max(xs) - min(xs) + padding, max(ys) - min(ys) + padding, max(zs) - min(zs) + padding)
    return span


def _residues_within(chain: Chain, ligand_residue: Residue, cutoff: float) -> List[str]:
    ligand_atoms = list(ligand_residue.get_atoms())
    residues: List[str] = []
    for residue in chain:
        if residue.id == ligand_residue.id:
            continue
        for atom in residue.get_atoms():
            for lig_atom in ligand_atoms:
                if atom - lig_atom <= cutoff:
                    residues.append(f"{chain.id}:{residue.resname}{residue.id[1]}")
                    break
            else:
                continue
            break
    return sorted(set(residues))


def build_pocket_from_interface(row: pd.Series, pdb_dir: Path, cutoff: float = 4.5) -> Optional[PocketDefinition]:
    fields = _extract_ligand_fields(row)
    pdb_path = pdb_dir / f"{fields['pdb_assembly_id']}.cif"
    if not pdb_path.exists():
        return None

    structure = _load_structure(pdb_path)
    chain = _find_chain(structure[0], str(fields["protein_chain_id"]))
    qc_flags: list[str] = []
    if chain is None:
        qc_flags.append("missing_protein_chain")
        return None

    ligand_resseq = None
    if "ligand_resseq" in fields and not pd.isna(fields["ligand_resseq"]):
        try:
            ligand_resseq = int(fields["ligand_resseq"])
        except ValueError:
            ligand_resseq = None

    ligand = _find_ligand(chain, fields.get("ligand_chain_id"), str(fields["ligand_id"]), ligand_resseq)
    if ligand is None:
        qc_flags.append("missing_ligand")
        return None

    ligand_coords = [tuple(atom.coord) for atom in ligand.get_atoms()]
    center = _centroid(ligand_coords)
    box_size = _bounding_box(ligand_coords)
    residues = _residues_within(chain, ligand, cutoff=cutoff)

    return PocketDefinition(
        pdb_assembly_id=str(fields["pdb_assembly_id"]),
        protein_chain_id=str(fields["protein_chain_id"]),
        ligand_id=str(fields["ligand_id"]),
        center=center,
        box_size=box_size,
        residues=residues,
        receptor_path=str(pdb_path),
        qc_flags=qc_flags,
    )


def build_pockets(interfaces_csv: Path, pdb_dir: Path, outdir: Path, cutoff: float = 4.5) -> list[Path]:
    interfaces = pd.read_csv(interfaces_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for _, row in interfaces.iterrows():
        pocket = build_pocket_from_interface(row, pdb_dir, cutoff=cutoff)
        if pocket is None:
            continue
        outfile = outdir / f"{pocket.pdb_assembly_id}__{pocket.protein_chain_id}__{pocket.ligand_id}.json"
        outfile.write_text(pocket.to_json())
        outputs.append(outfile)
    return outputs

