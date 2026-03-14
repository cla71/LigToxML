from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .panel.panel_builder import build_panel
from .pockets.builder import build_pockets
from .structures.downloader import fetch_structures
from .docking.screen import screen
from .agents.cli import app as agent_app

app = typer.Typer(help="AF3/PDB off-target screening pipeline")
app.add_typer(agent_app, name="agent", help="AI-driven agentic docking system")
console = Console()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@app.command("build-panel")
def cli_build_panel(
    foldbench_dir: Path = typer.Option(..., exists=True, help="Directory containing FoldBench CSVs"),
    out: Path = typer.Option(..., help="Path to write the filtered panel CSV"),
):
    target_list = foldbench_dir / "alphafold3_foldbench_target_list.csv"
    interfaces = foldbench_dir / "alphafold3_foldbench_protein_ligand_interfaces.csv"
    panel = build_panel(target_list, interfaces, out)
    console.print(f"Wrote {len(panel)} panel entries to {out}")


@app.command("fetch-structures")
def cli_fetch_structures(
    panel: Path = typer.Option(..., exists=True, help="Panel CSV produced by build-panel"),
    outdir: Path = typer.Option(..., help="Directory to save PDB assemblies"),
    url_template: str = typer.Option(
        "https://files.rcsb.org/download/{pdb_id}.cif", help="URL template for PDB downloads"
    ),
):
    downloaded = fetch_structures(panel, outdir, url_template=url_template)
    console.print(f"Downloaded {len(downloaded)} assemblies to {outdir}")


@app.command("build-pockets")
def cli_build_pockets(
    interfaces: Path = typer.Option(..., exists=True, help="FoldBench protein-ligand interface CSV"),
    pdb_dir: Path = typer.Option(..., exists=True, help="Directory containing PDB assemblies"),
    outdir: Path = typer.Option(..., help="Output directory for pocket JSON files"),
    cutoff: float = typer.Option(4.5, help="Distance cutoff in Angstroms for pocket residues"),
):
    pockets = build_pockets(interfaces, pdb_dir, outdir, cutoff=cutoff)
    console.print(f"Generated {len(pockets)} pocket definitions in {outdir}")


@app.command("screen")
def cli_screen(
    compounds: Path = typer.Option(..., exists=True, help="Compounds file (.smi or .sdf)"),
    pockets: Path = typer.Option(..., exists=True, help="Directory containing pocket JSON files"),
    receptor_mode: str = typer.Option(
        "pdb", help="Receptor mode (pdb or af3_or_pdb)",
    ),
    out: Path = typer.Option(..., help="Output directory for results"),
):
    screen(compounds, pockets, receptor_mode=receptor_mode, outdir=out)
    console.print(f"Results written to {out}")


@app.command("summarise")
def cli_summarise(
    pockets: Path = typer.Option(..., exists=True, help="Pocket directory for inspection"),
):
    table = Table(title="Pocket definitions")
    table.add_column("Pocket file")
    table.add_column("Receptor")
    table.add_column("QC flags")
    for pocket_file in pockets.glob("*.json"):
        table.add_row(pocket_file.name, pocket_file.as_posix(), "")
    console.print(table)


if __name__ == "__main__":
    app()
