from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


INTERFACE_TASK_KEY = "interface_protein_ligand"


def _normalise_pdb_column(df: pd.DataFrame) -> str:
    for candidate in ["pdb_assembly_id", "pdb_id", "assembly_id"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("Cannot find a column containing PDB assembly identifiers.")


def load_panel_candidates(target_list: Path) -> pd.DataFrame:
    df = pd.read_csv(target_list)
    if "tasks" not in df.columns:
        raise KeyError("FoldBench target list must include a 'tasks' column")
    return df


def filter_protein_ligand_targets(targets: pd.DataFrame) -> pd.DataFrame:
    def has_protein_ligand(task_cell: Iterable[str]) -> bool:
        if pd.isna(task_cell):
            return False
        return INTERFACE_TASK_KEY in str(task_cell).split(";")

    mask = targets["tasks"].apply(has_protein_ligand)
    return targets.loc[mask].copy()


def filter_targets_with_interfaces(targets: pd.DataFrame, interfaces_csv: Path) -> pd.DataFrame:
    interfaces = pd.read_csv(interfaces_csv)
    pdb_col_targets = _normalise_pdb_column(targets)
    pdb_col_interfaces = _normalise_pdb_column(interfaces)

    valid_ids = set(interfaces[pdb_col_interfaces].astype(str).unique())
    return targets.loc[targets[pdb_col_targets].astype(str).isin(valid_ids)].copy()


def build_panel(target_list: Path, interfaces_csv: Path, output: Path) -> pd.DataFrame:
    targets = load_panel_candidates(target_list)
    protein_ligand_targets = filter_protein_ligand_targets(targets)
    filtered = filter_targets_with_interfaces(protein_ligand_targets, interfaces_csv)

    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output, index=False)
    return filtered

