"""AutoDock Vina integration for real molecular docking.

Supports both the classic vina binary and the vina Python package.
Falls back to the mock scorer if neither is available.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VinaConfig:
    """Configuration for a Vina docking run."""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    cpu: int = 0  # 0 = auto-detect


@dataclass
class VinaResult:
    """Result of a Vina docking run."""
    score: float
    all_scores: list[float]
    output_path: Optional[str] = None
    log_path: Optional[str] = None


def check_vina_available() -> str:
    """Check which Vina runtime is available.

    Returns: 'binary', 'python', or 'none'.
    """
    if shutil.which("vina"):
        return "binary"
    try:
        from vina import Vina
        return "python"
    except ImportError:
        pass
    return "none"


def prepare_receptor_pdbqt(
    receptor_path: Path, output_path: Optional[Path] = None
) -> Path:
    """Convert a receptor structure to PDBQT format using Open Babel.

    If the input is already PDBQT, return it directly.
    """
    if receptor_path.suffix == ".pdbqt":
        return receptor_path

    out = output_path or receptor_path.with_suffix(".pdbqt")
    if out.exists():
        return out

    # Try obabel
    try:
        result = subprocess.run(
            ["obabel", str(receptor_path), "-O", str(out), "-xr"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and out.exists():
            return out
        logger.warning("obabel conversion failed: %s", result.stderr[:200])
    except FileNotFoundError:
        pass

    # Try ADFR prepare_receptor
    try:
        result = subprocess.run(
            ["prepare_receptor", "-r", str(receptor_path), "-o", str(out)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and out.exists():
            return out
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Cannot convert {receptor_path} to PDBQT. "
        "Install Open Babel (conda install -c conda-forge openbabel) or "
        "ADFR Suite (prepare_receptor)."
    )


def prepare_ligand_pdbqt(
    ligand_path: Path, output_path: Optional[Path] = None
) -> Path:
    """Convert a ligand file to PDBQT format."""
    if ligand_path.suffix == ".pdbqt":
        return ligand_path

    out = output_path or ligand_path.with_suffix(".pdbqt")
    if out.exists():
        return out

    try:
        result = subprocess.run(
            ["obabel", str(ligand_path), "-O", str(out), "--gen3d"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and out.exists():
            return out
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Cannot convert {ligand_path} to PDBQT. "
        "Install Open Babel: conda install -c conda-forge openbabel"
    )


def dock_vina_binary(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    config: VinaConfig,
    output_dir: Path,
) -> VinaResult:
    """Run docking using the vina command-line binary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ligand_pdbqt.stem}_docked.pdbqt"
    log_path = output_dir / f"{ligand_pdbqt.stem}_vina.log"

    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(config.center_x),
        "--center_y", str(config.center_y),
        "--center_z", str(config.center_z),
        "--size_x", str(config.size_x),
        "--size_y", str(config.size_y),
        "--size_z", str(config.size_z),
        "--exhaustiveness", str(config.exhaustiveness),
        "--num_modes", str(config.num_modes),
        "--energy_range", str(config.energy_range),
        "--out", str(out_path),
        "--log", str(log_path),
    ]
    if config.cpu > 0:
        cmd.extend(["--cpu", str(config.cpu)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Vina failed: {result.stderr[:500]}")

    scores = _parse_vina_output(result.stdout)
    return VinaResult(
        score=scores[0] if scores else 0.0,
        all_scores=scores,
        output_path=str(out_path),
        log_path=str(log_path),
    )


def dock_vina_python(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    config: VinaConfig,
    output_dir: Path,
) -> VinaResult:
    """Run docking using the vina Python package."""
    from vina import Vina

    v = Vina(sf_name="vina")
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(
        center=[config.center_x, config.center_y, config.center_z],
        box_size=[config.size_x, config.size_y, config.size_z],
    )
    v.dock(exhaustiveness=config.exhaustiveness, n_poses=config.num_modes)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ligand_pdbqt.stem}_docked.pdbqt"
    v.write_poses(str(out_path), n_poses=config.num_modes, overwrite=True)

    energies = v.energies(n_poses=config.num_modes)
    scores = [float(e[0]) for e in energies] if energies is not None else []

    return VinaResult(
        score=scores[0] if scores else 0.0,
        all_scores=scores,
        output_path=str(out_path),
    )


def dock(
    receptor_path: Path,
    ligand_path: Path,
    config: VinaConfig,
    output_dir: Path,
) -> VinaResult:
    """Run Vina docking using the best available runtime.

    Handles receptor/ligand format conversion automatically.
    """
    runtime = check_vina_available()
    if runtime == "none":
        raise RuntimeError(
            "AutoDock Vina not found. Install via:\n"
            "  conda install -c conda-forge autodock-vina  (binary)\n"
            "  pip install vina  (Python bindings)"
        )

    receptor_pdbqt = prepare_receptor_pdbqt(receptor_path)
    ligand_pdbqt = prepare_ligand_pdbqt(ligand_path)

    if runtime == "binary":
        return dock_vina_binary(receptor_pdbqt, ligand_pdbqt, config, output_dir)
    return dock_vina_python(receptor_pdbqt, ligand_pdbqt, config, output_dir)


def _parse_vina_output(stdout: str) -> list[float]:
    """Parse binding affinities from Vina stdout."""
    scores = []
    in_results = False
    for line in stdout.splitlines():
        if "-----+------------" in line:
            in_results = True
            continue
        if in_results:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    int(parts[0])  # mode number
                    scores.append(float(parts[1]))
                except ValueError:
                    in_results = False
    return scores
