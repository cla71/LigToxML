from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def _guess_pdb_column(panel: pd.DataFrame) -> str:
    for candidate in ["pdb_assembly_id", "pdb_id", "assembly_id"]:
        if candidate in panel.columns:
            return candidate
    raise KeyError("Panel CSV missing a PDB identifier column")


def iter_target_ids(panel_csv: Path) -> Iterable[str]:
    panel = pd.read_csv(panel_csv)
    pdb_col = _guess_pdb_column(panel)
    for raw in panel[pdb_col].unique():
        yield str(raw)


def _strip_assembly_suffix(identifier: str) -> str:
    # FoldBench assemblies can be e.g. 4XYZ-1. We only need the PDB code for download.
    return identifier.split("-")[0]


def download_structure(pdb_assembly_id: str, outdir: Path, url_template: str = DEFAULT_RCSB_URL) -> Optional[Path]:
    pdb_id = _strip_assembly_suffix(pdb_assembly_id)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{pdb_assembly_id}.cif"

    if outfile.exists():
        logger.info("Skipping %s (already exists)", outfile)
        return outfile

    url = url_template.format(pdb_id=pdb_id)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        logger.warning("Failed to download %s from %s (status %s)", pdb_assembly_id, url, response.status_code)
        return None

    outfile.write_bytes(response.content)
    return outfile


def fetch_structures(panel_csv: Path, outdir: Path, url_template: str = DEFAULT_RCSB_URL) -> list[Path]:
    downloaded: list[Path] = []
    ids = list(iter_target_ids(panel_csv))
    for pdb_assembly_id in tqdm(ids, desc="Downloading PDB assemblies"):
        file_path = download_structure(pdb_assembly_id, outdir, url_template)
        if file_path:
            downloaded.append(file_path)
    return downloaded

