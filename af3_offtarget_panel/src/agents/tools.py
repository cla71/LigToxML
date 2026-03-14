"""Tool definitions that wrap existing LigToxML pipeline steps for agentic use.

Each tool is defined as:
  1. An Ollama-compatible JSON schema (for the LLM to call)
  2. A Python executor function (to actually run the step)
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..panel.panel_builder import build_panel
from ..structures.downloader import fetch_structures, download_structure
from ..pockets.builder import build_pockets, build_pocket_from_interface
from ..docking.screen import load_compounds, screen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema definitions (Ollama tool-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_uniprot",
            "description": (
                "Search UniProt for a protein by name or identifier. Returns "
                "UniProt accession, gene name, organism, and PDB cross-references."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Protein name, gene name, or UniProt accession",
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism filter (e.g., 'Homo sapiens')",
                        "default": "Homo sapiens",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_pdb_ids",
            "description": (
                "Given a list of protein names or UniProt IDs, resolve them to "
                "PDB structure identifiers suitable for docking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proteins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of protein names or UniProt accessions",
                    },
                },
                "required": ["proteins"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_protein_structure",
            "description": (
                "Download a protein structure (mmCIF) from RCSB PDB given a PDB ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pdb_id": {
                        "type": "string",
                        "description": "4-character PDB identifier",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the structure file",
                    },
                },
                "required": ["pdb_id", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_structure_alphafold",
            "description": (
                "Predict a protein structure using AlphaFold (local ColabFold). "
                "Use this for proteins without experimental PDB structures, or "
                "when higher-quality models are needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Amino acid sequence (one-letter code)",
                    },
                    "protein_name": {
                        "type": "string",
                        "description": "Name for the prediction output",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for AlphaFold output",
                    },
                },
                "required": ["sequence", "protein_name", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_ligand",
            "description": (
                "Validate and prepare a ligand from SMILES string. Generates 3D "
                "coordinates and writes PDBQT file for docking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the ligand",
                    },
                    "name": {
                        "type": "string",
                        "description": "Identifier for the ligand",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to write ligand files",
                    },
                },
                "required": ["smiles", "name", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_binding_pocket",
            "description": (
                "Detect binding pockets in a protein structure using native "
                "ligand coordinates or fpocket cavity detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "structure_path": {
                        "type": "string",
                        "description": "Path to the protein structure (CIF/PDB)",
                    },
                    "chain_id": {
                        "type": "string",
                        "description": "Protein chain identifier",
                    },
                    "ligand_id": {
                        "type": "string",
                        "description": "Ligand residue name (if known)",
                        "default": "",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for pocket definitions",
                    },
                },
                "required": ["structure_path", "chain_id", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_docking_vina",
            "description": (
                "Run AutoDock Vina molecular docking of a ligand against a "
                "protein pocket. Returns binding affinity score in kcal/mol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "receptor_path": {
                        "type": "string",
                        "description": "Path to receptor PDBQT file",
                    },
                    "ligand_path": {
                        "type": "string",
                        "description": "Path to ligand PDBQT file",
                    },
                    "center_x": {"type": "number", "description": "Pocket center X"},
                    "center_y": {"type": "number", "description": "Pocket center Y"},
                    "center_z": {"type": "number", "description": "Pocket center Z"},
                    "size_x": {"type": "number", "description": "Box size X (Angstroms)", "default": 20},
                    "size_y": {"type": "number", "description": "Box size Y (Angstroms)", "default": 20},
                    "size_z": {"type": "number", "description": "Box size Z (Angstroms)", "default": 20},
                    "exhaustiveness": {"type": "integer", "description": "Search exhaustiveness", "default": 8},
                },
                "required": ["receptor_path", "ligand_path", "center_x", "center_y", "center_z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_docking_diffdock",
            "description": (
                "Run DiffDock deep-learning docking for a protein-ligand pair. "
                "More accurate than Vina for flexible docking but slower."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "receptor_path": {
                        "type": "string",
                        "description": "Path to receptor PDB/CIF file",
                    },
                    "ligand_smiles": {
                        "type": "string",
                        "description": "SMILES string of the ligand",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for docking poses",
                    },
                    "num_poses": {
                        "type": "integer",
                        "description": "Number of poses to generate",
                        "default": 10,
                    },
                },
                "required": ["receptor_path", "ligand_smiles", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_screening_batch",
            "description": (
                "Screen a batch of compounds against multiple protein pockets "
                "using the LigToxML screening pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "compounds_file": {
                        "type": "string",
                        "description": "Path to compounds (.smi or .sdf)",
                    },
                    "pockets_dir": {
                        "type": "string",
                        "description": "Directory containing pocket JSON files",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for results",
                    },
                    "receptor_mode": {
                        "type": "string",
                        "description": "Receptor source: 'pdb', 'af3_or_pdb', or 'vina'",
                        "default": "pdb",
                    },
                },
                "required": ["compounds_file", "pockets_dir", "output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyse_results",
            "description": (
                "Analyse docking results and generate a ranked summary of "
                "off-target binding predictions with confidence scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results_dir": {
                        "type": "string",
                        "description": "Directory containing screening results",
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": "Score threshold for flagging hits (kcal/mol)",
                        "default": -6.0,
                    },
                },
                "required": ["results_dir"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor functions
# ---------------------------------------------------------------------------


def execute_search_uniprot(query: str, organism: str = "Homo sapiens") -> dict:
    """Search UniProt REST API for protein info and PDB cross-references."""
    import requests as req

    search_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({query}) AND (organism_name:{organism})",
        "format": "json",
        "size": 5,
        "fields": "accession,gene_names,protein_name,organism_name,xref_pdb",
    }
    try:
        resp = req.get(search_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for entry in data.get("results", []):
            pdb_refs = []
            for xref in entry.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "PDB":
                    pdb_refs.append(xref.get("id", ""))
            results.append({
                "accession": entry.get("primaryAccession", ""),
                "protein_name": entry.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", query),
                "gene_names": [g.get("geneName", {}).get("value", "")
                               for g in entry.get("genes", [])],
                "organism": entry.get("organism", {}).get("scientificName", ""),
                "pdb_ids": pdb_refs,
            })
        return {"status": "success", "results": results}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def execute_resolve_pdb_ids(proteins: list[str]) -> dict:
    """Resolve a batch of protein names to PDB IDs via UniProt."""
    resolved = {}
    for protein in proteins:
        result = execute_search_uniprot(protein)
        pdb_ids = []
        if result["status"] == "success":
            for r in result["results"]:
                pdb_ids.extend(r.get("pdb_ids", []))
        resolved[protein] = pdb_ids[:5]  # Top 5 structures per protein
    return {"status": "success", "resolved": resolved}


def execute_fetch_protein_structure(pdb_id: str, output_dir: str) -> dict:
    """Download a PDB structure using existing downloader."""
    outdir = Path(output_dir)
    result = download_structure(pdb_id, outdir)
    if result:
        return {"status": "success", "path": str(result)}
    return {"status": "error", "message": f"Failed to download {pdb_id}"}


def execute_predict_structure_alphafold(
    sequence: str, protein_name: str, output_dir: str
) -> dict:
    """Run local AlphaFold/ColabFold structure prediction."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write FASTA input
    fasta_path = outdir / f"{protein_name}.fasta"
    fasta_path.write_text(f">{protein_name}\n{sequence}\n")

    # Try ColabFold (localcolabfold) first, then fall back to AlphaFold
    for cmd_name, cmd_args in [
        ("colabfold_batch", ["colabfold_batch", str(fasta_path), str(outdir)]),
        ("alphafold", [
            "python", "-m", "alphafold.run_alphafold",
            f"--fasta_paths={fasta_path}",
            f"--output_dir={outdir}",
            "--model_preset=monomer",
            "--db_preset=reduced_dbs",
        ]),
    ]:
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if result.returncode == 0:
                # Find the output model
                for ext in ("_relaxed_rank_001*.pdb", "ranked_0.pdb", "*.pdb"):
                    models = list(outdir.glob(ext))
                    if models:
                        return {
                            "status": "success",
                            "path": str(models[0]),
                            "method": cmd_name,
                        }
                return {
                    "status": "success",
                    "path": str(outdir),
                    "method": cmd_name,
                    "note": "Model generated but output file not found in expected location",
                }
            logger.warning("%s failed: %s", cmd_name, result.stderr[:500])
        except FileNotFoundError:
            logger.info("%s not found, trying next option...", cmd_name)
            continue
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"{cmd_name} timed out (2h limit)"}

    return {
        "status": "error",
        "message": (
            "No AlphaFold runtime found. Install ColabFold: "
            "pip install colabfold[alphafold] "
            "or set up AlphaFold locally."
        ),
        "fasta_path": str(fasta_path),
    }


def execute_prepare_ligand(smiles: str, name: str, output_dir: str) -> dict:
    """Prepare a ligand: validate SMILES, generate 3D coords, write files."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        # Fallback: write SMILES file only
        smi_path = outdir / f"{name}.smi"
        smi_path.write_text(f"{smiles} {name}\n")
        return {"status": "success", "smiles_file": str(smi_path), "note": "RDKit not available; wrote SMILES only"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "error", "message": f"Invalid SMILES: {smiles}"}

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # Write SDF
    sdf_path = outdir / f"{name}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    # Write PDBQT via obabel if available
    pdbqt_path = outdir / f"{name}.pdbqt"
    try:
        subprocess.run(
            ["obabel", str(sdf_path), "-O", str(pdbqt_path), "--gen3d"],
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        pdbqt_path = None

    info = {
        "status": "success",
        "sdf_path": str(sdf_path),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "num_atoms": mol.GetNumHeavyAtoms(),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    }
    if pdbqt_path and pdbqt_path.exists():
        info["pdbqt_path"] = str(pdbqt_path)
    return info


def execute_detect_binding_pocket(
    structure_path: str,
    chain_id: str,
    output_dir: str,
    ligand_id: str = "",
) -> dict:
    """Detect binding pocket from structure, optionally guided by a ligand."""
    struct_path = Path(structure_path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not struct_path.exists():
        return {"status": "error", "message": f"Structure not found: {structure_path}"}

    # If ligand_id is given, use existing pocket builder
    if ligand_id:
        row = pd.Series({
            "pdb_assembly_id": struct_path.stem,
            "protein_chain_id": chain_id,
            "ligand_id": ligand_id,
        })
        pocket = build_pocket_from_interface(row, struct_path.parent)
        if pocket:
            outfile = outdir / f"{struct_path.stem}__{chain_id}__{ligand_id}.json"
            outfile.write_text(pocket.to_json())
            return {
                "status": "success",
                "pocket_file": str(outfile),
                "center": list(pocket.center),
                "box_size": list(pocket.box_size),
                "num_residues": len(pocket.residues),
            }

    # Fallback: try fpocket for blind pocket detection
    try:
        result = subprocess.run(
            ["fpocket", "-f", str(struct_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            fpocket_dir = struct_path.parent / f"{struct_path.stem}_out"
            pockets_file = fpocket_dir / f"{struct_path.stem}_out.pdb"
            if pockets_file.exists():
                return {
                    "status": "success",
                    "method": "fpocket",
                    "pocket_dir": str(fpocket_dir),
                }
    except FileNotFoundError:
        pass

    # Last resort: geometric center of chain
    from Bio.PDB import MMCIFParser, PDBParser
    if struct_path.suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(struct_path.stem, struct_path)
    coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    for atom in residue:
                        if atom.name == "CA":
                            coords.append(tuple(atom.coord))
    if not coords:
        return {"status": "error", "message": f"Chain {chain_id} not found"}

    xs, ys, zs = zip(*coords)
    center = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
    pocket_def = {
        "pdb_assembly_id": struct_path.stem,
        "protein_chain_id": chain_id,
        "ligand_id": "UNKNOWN",
        "center": center,
        "box_size": [30.0, 30.0, 30.0],
        "residues": [],
        "receptor_path": str(struct_path),
        "qc_flags": ["blind_pocket_detection"],
    }
    outfile = outdir / f"{struct_path.stem}__{chain_id}__blind.json"
    outfile.write_text(json.dumps(pocket_def, indent=2))
    return {
        "status": "success",
        "method": "geometric_center",
        "pocket_file": str(outfile),
        "center": center,
        "note": "Blind pocket; consider using fpocket or providing a ligand",
    }


def execute_run_docking_vina(
    receptor_path: str,
    ligand_path: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 20.0,
    size_y: float = 20.0,
    size_z: float = 20.0,
    exhaustiveness: int = 8,
) -> dict:
    """Run AutoDock Vina docking."""
    receptor = Path(receptor_path)
    ligand = Path(ligand_path)

    if not receptor.exists():
        return {"status": "error", "message": f"Receptor not found: {receptor_path}"}
    if not ligand.exists():
        return {"status": "error", "message": f"Ligand not found: {ligand_path}"}

    # Prepare receptor PDBQT if needed
    receptor_pdbqt = receptor.with_suffix(".pdbqt")
    if not receptor_pdbqt.exists():
        try:
            subprocess.run(
                ["obabel", str(receptor), "-O", str(receptor_pdbqt), "-xr"],
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError:
            return {
                "status": "error",
                "message": "Open Babel not found. Install: conda install -c conda-forge openbabel",
            }

    out_path = ligand.parent / f"{ligand.stem}_docked.pdbqt"
    log_path = ligand.parent / f"{ligand.stem}_vina.log"

    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand),
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--exhaustiveness", str(exhaustiveness),
        "--out", str(out_path),
        "--log", str(log_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"status": "error", "message": result.stderr[:500]}

        # Parse best score from log
        scores = []
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    mode = int(parts[0])
                    score = float(parts[1])
                    scores.append(score)
                except ValueError:
                    continue

        return {
            "status": "success",
            "best_score": scores[0] if scores else None,
            "all_scores": scores[:9],
            "output_path": str(out_path),
            "log_path": str(log_path),
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "AutoDock Vina not found. Install: conda install -c conda-forge autodock-vina",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Vina timed out (10 min limit)"}


def execute_run_docking_diffdock(
    receptor_path: str,
    ligand_smiles: str,
    output_dir: str,
    num_poses: int = 10,
) -> dict:
    """Run DiffDock deep-learning docking."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write input CSV for DiffDock
    input_csv = outdir / "input.csv"
    input_csv.write_text(
        f"complex_name,protein_path,ligand_description,protein_sequence\n"
        f"complex_0,{receptor_path},{ligand_smiles},\n"
    )

    try:
        result = subprocess.run(
            [
                "python", "-m", "inference",
                "--config", "default_inference_args.yaml",
                "--protein_ligand_csv", str(input_csv),
                "--out_dir", str(outdir),
                "--samples_per_complex", str(num_poses),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode == 0:
            poses = list(outdir.glob("**/rank*_confidence*.sdf"))
            return {
                "status": "success",
                "num_poses": len(poses),
                "output_dir": str(outdir),
                "poses": [str(p) for p in poses[:5]],
            }
        return {"status": "error", "message": result.stderr[:500]}
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "DiffDock not found. Clone and install from: https://github.com/gcorso/DiffDock",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "DiffDock timed out (30 min limit)"}


def execute_run_screening_batch(
    compounds_file: str,
    pockets_dir: str,
    output_dir: str,
    receptor_mode: str = "pdb",
) -> dict:
    """Run batch screening using the LigToxML pipeline."""
    try:
        screen(
            Path(compounds_file),
            Path(pockets_dir),
            receptor_mode=receptor_mode,
            outdir=Path(output_dir),
        )
        hits_csv = Path(output_dir) / "off_target_hits.csv"
        if hits_csv.exists():
            df = pd.read_csv(hits_csv)
            return {
                "status": "success",
                "num_hits": len(df),
                "results_file": str(hits_csv),
                "top_hits": df.head(10).to_dict(orient="records"),
            }
        return {"status": "success", "output_dir": output_dir}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def execute_analyse_results(results_dir: str, score_threshold: float = -6.0) -> dict:
    """Analyse screening results and provide summary statistics."""
    rdir = Path(results_dir)
    hits_csv = rdir / "off_target_hits.csv"
    if not hits_csv.exists():
        return {"status": "error", "message": f"No results file at {hits_csv}"}

    df = pd.read_csv(hits_csv)
    strong_hits = df[df["score"] <= score_threshold]
    summary = {
        "status": "success",
        "total_compound_target_pairs": len(df),
        "unique_compounds": df["compound_id"].nunique(),
        "unique_targets": df["pdb_assembly_id"].nunique(),
        "hits_below_threshold": len(strong_hits),
        "score_threshold": score_threshold,
        "best_score": float(df["score"].min()) if len(df) > 0 else None,
        "worst_score": float(df["score"].max()) if len(df) > 0 else None,
        "mean_score": float(df["score"].mean()) if len(df) > 0 else None,
        "confidence_distribution": df["confidence"].value_counts().to_dict() if "confidence" in df.columns else {},
        "top_hits": strong_hits.nsmallest(10, "score").to_dict(orient="records") if len(strong_hits) > 0 else [],
    }
    return summary


# ---------------------------------------------------------------------------
# Dispatcher: route tool name → executor
# ---------------------------------------------------------------------------

TOOL_EXECUTORS = {
    "search_uniprot": execute_search_uniprot,
    "resolve_pdb_ids": execute_resolve_pdb_ids,
    "fetch_protein_structure": execute_fetch_protein_structure,
    "predict_structure_alphafold": execute_predict_structure_alphafold,
    "prepare_ligand": execute_prepare_ligand,
    "detect_binding_pocket": execute_detect_binding_pocket,
    "run_docking_vina": execute_run_docking_vina,
    "run_docking_diffdock": execute_run_docking_diffdock,
    "run_screening_batch": execute_run_screening_batch,
    "analyse_results": execute_analyse_results,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Dispatch a tool call to its executor function."""
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return {"status": "error", "message": f"Unknown tool: {name}"}
    try:
        return executor(**arguments)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"status": "error", "message": f"{name} failed: {exc}"}
