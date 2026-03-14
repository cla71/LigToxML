"""AlphaFold local structure prediction integration.

Supports multiple backends:
  1. ColabFold (localcolabfold) — fastest for single sequences
  2. AlphaFold2/3 local installation
  3. ESMFold (via transformers) — no MSA needed, GPU only
  4. AlphaFold DB download — pre-computed structures from EMBL-EBI
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

ALPHAFOLD_DB_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
)


@dataclass
class PredictionResult:
    """Result of a structure prediction."""
    method: str
    model_path: str
    confidence_score: Optional[float] = None
    plddt_mean: Optional[float] = None


def download_alphafold_db(
    uniprot_id: str, output_dir: Path
) -> Optional[PredictionResult]:
    """Download a pre-computed AlphaFold structure from the EBI database.

    This is the fastest option — no computation required.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    url = ALPHAFOLD_DB_URL.format(uniprot_id=uniprot_id)
    outfile = output_dir / f"AF-{uniprot_id}.pdb"

    if outfile.exists():
        logger.info("AlphaFold DB model already exists: %s", outfile)
        return PredictionResult(method="alphafold_db", model_path=str(outfile))

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            outfile.write_bytes(resp.content)
            logger.info("Downloaded AlphaFold DB model for %s", uniprot_id)
            return PredictionResult(method="alphafold_db", model_path=str(outfile))
        logger.warning(
            "AlphaFold DB download failed for %s (HTTP %s)", uniprot_id, resp.status_code
        )
    except Exception as exc:
        logger.warning("AlphaFold DB download error: %s", exc)

    return None


def predict_colabfold(
    sequence: str,
    name: str,
    output_dir: Path,
    num_models: int = 1,
    use_amber: bool = False,
) -> Optional[PredictionResult]:
    """Run local ColabFold structure prediction.

    Requires: pip install colabfold[alphafold]
    """
    if not shutil.which("colabfold_batch"):
        logger.info("ColabFold not found")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{name}.fasta"
    fasta_path.write_text(f">{name}\n{sequence}\n")

    cmd = [
        "colabfold_batch",
        str(fasta_path),
        str(output_dir),
        "--num-models", str(num_models),
    ]
    if use_amber:
        cmd.append("--amber")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200
        )
        if result.returncode == 0:
            # Find best ranked model
            for pattern in [
                f"{name}_relaxed_rank_001*.pdb",
                f"{name}_unrelaxed_rank_001*.pdb",
                f"{name}*.pdb",
            ]:
                models = sorted(output_dir.glob(pattern))
                if models:
                    plddt = _extract_plddt(models[0])
                    return PredictionResult(
                        method="colabfold",
                        model_path=str(models[0]),
                        plddt_mean=plddt,
                    )
        logger.warning("ColabFold failed: %s", result.stderr[:300])
    except subprocess.TimeoutExpired:
        logger.error("ColabFold timed out after 2 hours")
    except Exception as exc:
        logger.error("ColabFold error: %s", exc)

    return None


def predict_esmfold(
    sequence: str, name: str, output_dir: Path
) -> Optional[PredictionResult]:
    """Run ESMFold structure prediction (no MSA required, GPU recommended).

    Requires: pip install transformers torch
    """
    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError:
        logger.info("ESMFold dependencies not available (transformers, torch)")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{name}_esmfold.pdb"

    if outfile.exists():
        return PredictionResult(method="esmfold", model_path=str(outfile))

    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pdb_string = model.output_to_pdb(outputs)[0]
        outfile.write_text(pdb_string)

        # Extract mean pLDDT
        plddt = float(outputs["plddt"].mean().cpu().item()) if hasattr(outputs, "plddt") else None

        return PredictionResult(
            method="esmfold",
            model_path=str(outfile),
            plddt_mean=plddt,
        )
    except Exception as exc:
        logger.error("ESMFold prediction failed: %s", exc)
        return None


def predict_structure(
    sequence: str,
    name: str,
    output_dir: Path,
    uniprot_id: Optional[str] = None,
    prefer_method: Optional[str] = None,
) -> Optional[PredictionResult]:
    """Predict a protein structure using the best available method.

    Priority order (unless prefer_method is set):
      1. AlphaFold DB (if UniProt ID provided) — instant download
      2. ColabFold — local GPU prediction with MSA
      3. ESMFold — single-sequence, no MSA needed
    """
    methods = []

    if prefer_method:
        method_map = {
            "alphafold_db": lambda: download_alphafold_db(uniprot_id, output_dir) if uniprot_id else None,
            "colabfold": lambda: predict_colabfold(sequence, name, output_dir),
            "esmfold": lambda: predict_esmfold(sequence, name, output_dir),
        }
        if prefer_method in method_map:
            result = method_map[prefer_method]()
            if result:
                return result
            logger.warning("Preferred method %s failed, trying others...", prefer_method)

    # Try AlphaFold DB first (fastest)
    if uniprot_id:
        result = download_alphafold_db(uniprot_id, output_dir)
        if result:
            return result

    # Try ColabFold
    result = predict_colabfold(sequence, name, output_dir)
    if result:
        return result

    # Try ESMFold
    result = predict_esmfold(sequence, name, output_dir)
    if result:
        return result

    logger.error(
        "No structure prediction method available. Install one of:\n"
        "  pip install colabfold[alphafold]  (recommended)\n"
        "  pip install transformers torch  (ESMFold)\n"
    )
    return None


def _extract_plddt(pdb_path: Path) -> Optional[float]:
    """Extract mean pLDDT from B-factor column of an AlphaFold PDB."""
    try:
        plddts = []
        for line in pdb_path.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                bfactor = float(line[60:66].strip())
                plddts.append(bfactor)
        if plddts:
            return sum(plddts) / len(plddts)
    except Exception:
        pass
    return None
