from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - rdkit is optional
    Chem = None
    AllChem = None


@dataclass
class Pocket:
    pdb_assembly_id: str
    protein_chain_id: str
    ligand_id: str
    receptor_path: str
    center: tuple[float, float, float]
    qc_flags: list[str]


@dataclass
class DockingResult:
    compound_id: str
    smiles: str
    pocket: Pocket
    score: float
    receptor_mode: str
    confidence: str


CONFIDENCE_THRESHOLDS = {
    "high": -8.0,
    "medium": -6.0,
}


def _load_pocket(path: Path) -> Pocket:
    payload = json.loads(path.read_text())
    return Pocket(
        pdb_assembly_id=payload["pdb_assembly_id"],
        protein_chain_id=payload["protein_chain_id"],
        ligand_id=payload["ligand_id"],
        receptor_path=payload["receptor_path"],
        center=tuple(payload["center"]),
        qc_flags=payload.get("qc_flags", []),
    )


def _parse_smiles(smiles_path: Path) -> List[tuple[str, str]]:
    compounds: List[tuple[str, str]] = []
    with open(smiles_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            parts = line.strip().split()
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else f"cmpd_{idx+1}"
            compounds.append((name, smi))
    return compounds


def _parse_sdf(sdf_path: Path) -> List[tuple[str, str]]:
    if Chem is None:
        raise ImportError("RDKit is required to parse SDF files")
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    compounds: List[tuple[str, str]] = []
    for idx, mol in enumerate(suppl):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"cmpd_{idx+1}"
        compounds.append((name, smi))
    return compounds


def load_compounds(path: Path) -> List[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".smi":
        return _parse_smiles(path)
    if suffix == ".sdf":
        return _parse_sdf(path)
    raise ValueError("Compounds must be provided as .smi or .sdf")


def _mock_score(smiles: str) -> float:
    if Chem is None:
        # simple heuristic: penalise longer SMILES so ranking is deterministic-ish
        return -len(smiles) / 5.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    num_atoms = mol.GetNumAtoms()
    rings = Chem.GetSSSR(mol)
    return -float(num_atoms) / 4.0 - float(rings) / 2.0


def _confidence_from_score(score: float) -> str:
    if score <= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if score <= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def dock_compounds(compounds: List[tuple[str, str]], pockets: Iterable[Pocket], receptor_mode: str) -> List[DockingResult]:
    results: List[DockingResult] = []
    for name, smiles in compounds:
        for pocket in pockets:
            score = _mock_score(smiles) + random.uniform(-0.25, 0.25)
            confidence = _confidence_from_score(score)
            results.append(
                DockingResult(
                    compound_id=name,
                    smiles=smiles,
                    pocket=pocket,
                    score=score,
                    receptor_mode=receptor_mode,
                    confidence=confidence,
                )
            )
    return results


def summarise_results(results: List[DockingResult], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    hits_rows = []
    detail_dir = outdir / "per_target_details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        hits_rows.append(
            {
                "compound_id": res.compound_id,
                "smiles": res.smiles,
                "pdb_assembly_id": res.pocket.pdb_assembly_id,
                "protein_chain_id": res.pocket.protein_chain_id,
                "ligand_id": res.pocket.ligand_id,
                "score": res.score,
                "confidence": res.confidence,
                "receptor_mode": res.receptor_mode,
                "qc_flags": ";".join(res.pocket.qc_flags),
            }
        )
        detail_path = detail_dir / f"{res.compound_id}__{res.pocket.pdb_assembly_id}.json"
        detail_payload = {
            "compound_id": res.compound_id,
            "smiles": res.smiles,
            "score": res.score,
            "confidence": res.confidence,
            "pocket": res.pocket.__dict__,
        }
        detail_path.write_text(json.dumps(detail_payload, indent=2))

    hits_df = pd.DataFrame(hits_rows)
    hits_df.sort_values(["compound_id", "score"], ascending=[True, True], inplace=True)
    hits_df.to_csv(outdir / "off_target_hits.csv", index=False)


def screen(compound_path: Path, pockets_dir: Path, receptor_mode: str, outdir: Path) -> None:
    pockets = [_load_pocket(path) for path in pockets_dir.glob("*.json")]
    compounds = load_compounds(compound_path)
    results = dock_compounds(compounds, pockets, receptor_mode=receptor_mode)
    summarise_results(results, outdir=outdir)

