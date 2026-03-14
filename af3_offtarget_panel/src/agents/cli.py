"""CLI entry points for the agentic docking system."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Agentic AI-driven molecular docking system")
console = Console()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@app.command("dock")
def cli_dock(
    proteins: str = typer.Argument(
        ..., help="Comma-separated list of protein names, UniProt IDs, or PDB IDs"
    ),
    ligand: str = typer.Argument(
        ..., help="Ligand SMILES string"
    ),
    ligand_name: str = typer.Option("ligand", help="Name for the ligand"),
    work_dir: Optional[Path] = typer.Option(
        None, help="Working directory for outputs (auto-generated if not set)"
    ),
    alphafold: bool = typer.Option(
        False, "--alphafold/--no-alphafold",
        help="Use AlphaFold for proteins without PDB structures",
    ),
    vina: bool = typer.Option(
        True, "--vina/--no-vina",
        help="Use AutoDock Vina for docking",
    ),
    diffdock: bool = typer.Option(
        False, "--diffdock/--no-diffdock",
        help="Use DiffDock for deep-learning docking",
    ),
    exhaustiveness: int = typer.Option(
        8, help="Vina search exhaustiveness (higher = slower but more thorough)"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress agent reasoning output"),
):
    """Run an AI-orchestrated docking workflow.

    The agent (powered by Qwen 3.5 via Ollama) will:
      1. Resolve protein names to PDB structures
      2. Download or predict structures
      3. Prepare the ligand
      4. Detect binding pockets
      5. Run molecular docking
      6. Analyse and report results

    Example:
        af3-agent dock "EGFR,BRAF,CDK2" "CC(=O)Nc1ccc(O)cc1" --ligand-name acetaminophen
    """
    from .orchestrator import DockingOrchestrator

    protein_list = [p.strip() for p in proteins.split(",") if p.strip()]
    if not protein_list:
        console.print("[red]Error: No proteins specified[/red]")
        raise typer.Exit(1)

    orchestrator = DockingOrchestrator(
        proteins=protein_list,
        ligand_smiles=ligand,
        ligand_name=ligand_name,
        work_dir=work_dir,
        use_alphafold=alphafold,
        use_vina=vina,
        use_diffdock=diffdock,
        exhaustiveness=exhaustiveness,
        verbose=not quiet,
    )

    state = orchestrator.run()
    if not state.completed:
        raise typer.Exit(1)


@app.command("interpret")
def cli_interpret(
    results_dir: Path = typer.Argument(
        ..., exists=True, help="Directory containing docking results"
    ),
    question: str = typer.Option(
        "", help="Specific question about the results"
    ),
):
    """Use AI to interpret docking results in natural language.

    Uses the smaller Qwen 3.5 4B model for fast interpretation.

    Example:
        af3-agent interpret results/agent_1234567890 --question "Which protein binds strongest?"
    """
    from .orchestrator import ResultAnalyser

    analyser = ResultAnalyser(results_dir)
    interpretation = analyser.interpret(question)
    console.print(interpretation)


@app.command("check")
def cli_check():
    """Check system readiness: Ollama server, models, and optional tools."""
    from .models import check_ollama_health, list_local_models, ORCHESTRATOR_MODEL, WORKER_MODEL
    import shutil

    console.print("[bold]System Readiness Check[/bold]\n")

    # Ollama
    ollama_ok = check_ollama_health()
    status = "[green]OK[/green]" if ollama_ok else "[red]NOT RUNNING[/red]"
    console.print(f"  Ollama server:  {status}")

    if ollama_ok:
        models = list_local_models()
        for model_name in [ORCHESTRATOR_MODEL, WORKER_MODEL]:
            found = any(m.startswith(model_name.split(":")[0]) for m in models)
            status = "[green]available[/green]" if found else "[yellow]not pulled[/yellow]"
            console.print(f"  {model_name:20s} {status}")
    else:
        console.print("  [dim]Start Ollama with: ollama serve[/dim]")

    # Optional tools
    console.print()
    tools = {
        "vina": "AutoDock Vina",
        "obabel": "Open Babel",
        "fpocket": "fpocket",
        "colabfold_batch": "ColabFold",
    }
    for binary, label in tools.items():
        found = shutil.which(binary) is not None
        status = "[green]found[/green]" if found else "[dim]not installed[/dim]"
        console.print(f"  {label:20s} {status}")

    # Python packages
    console.print()
    packages = {
        "rdkit": "RDKit",
        "vina": "Vina (Python)",
        "transformers": "Transformers (ESMFold)",
        "torch": "PyTorch",
    }
    for pkg, label in packages.items():
        try:
            __import__(pkg)
            status = "[green]installed[/green]"
        except ImportError:
            status = "[dim]not installed[/dim]"
        console.print(f"  {label:20s} {status}")


@app.command("pull-models")
def cli_pull_models():
    """Pull the required Qwen models into Ollama."""
    from .models import ensure_model, ORCHESTRATOR_MODEL, WORKER_MODEL

    for model in [ORCHESTRATOR_MODEL, WORKER_MODEL]:
        console.print(f"Pulling {model}...")
        if ensure_model(model):
            console.print(f"  [green]{model} ready[/green]")
        else:
            console.print(f"  [red]Failed to pull {model}[/red]")


if __name__ == "__main__":
    app()
