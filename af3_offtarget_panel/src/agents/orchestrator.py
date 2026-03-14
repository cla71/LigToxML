"""Orchestrator agent: plans and executes the full docking workflow.

Uses Qwen 3.5 9B (via Ollama) for high-level reasoning and planning,
and Qwen 3.5 4B for lightweight sub-tasks like result parsing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.markdown import Markdown

from .models import (
    ORCHESTRATOR_MODEL,
    WORKER_MODEL,
    ConversationContext,
    check_ollama_health,
    ensure_model,
    chat_completion,
    extract_response_text,
    extract_tool_calls,
)
from .tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)
console = Console()

MAX_AGENT_STEPS = 30
STEP_PAUSE = 0.5  # seconds between steps to avoid hammering Ollama

SYSTEM_PROMPT = """\
You are LigToxML Agent, an expert computational chemistry AI assistant \
specializing in molecular docking and off-target screening.

Your job is to help researchers screen compounds (ligands) against a panel \
of protein targets to identify potential off-target binding.

## Capabilities

You have access to the following tools:
- **search_uniprot**: Look up proteins by name/ID, get PDB cross-references
- **resolve_pdb_ids**: Batch-resolve protein names to PDB structures
- **fetch_protein_structure**: Download PDB structures from RCSB
- **predict_structure_alphafold**: Run local AlphaFold for proteins without PDB structures
- **prepare_ligand**: Validate SMILES, generate 3D coordinates
- **detect_binding_pocket**: Find binding pockets (native ligand or blind detection)
- **run_docking_vina**: AutoDock Vina molecular docking
- **run_docking_diffdock**: DiffDock deep-learning docking (more accurate, slower)
- **run_screening_batch**: Batch screening using the LigToxML pipeline
- **analyse_results**: Statistical analysis of docking results

## Workflow

For a typical docking request, follow this workflow:
1. **Resolve proteins**: Look up each protein to find PDB IDs
2. **Fetch structures**: Download PDB structures (or predict with AlphaFold)
3. **Prepare ligand**: Validate and generate 3D coordinates
4. **Detect pockets**: Find binding sites on each protein
5. **Run docking**: Dock the ligand against each pocket
6. **Analyse results**: Summarize binding affinities and flag hits

## Important Rules

- Always validate inputs before proceeding
- If a PDB structure is unavailable, suggest AlphaFold prediction
- Report scores in kcal/mol; lower (more negative) = stronger binding
- Flag any off-target hits with score ≤ -6.0 kcal/mol as potentially significant
- If tools fail, explain what happened and suggest alternatives
- Think step-by-step and explain your reasoning

Always use /think tags for internal reasoning before making decisions.
"""


@dataclass
class AgentState:
    """Tracks the state of an agentic workflow run."""
    proteins: list[str] = field(default_factory=list)
    ligand_smiles: str = ""
    ligand_name: str = ""
    work_dir: Path = field(default_factory=lambda: Path("results/agent_run"))
    resolved_structures: dict[str, list[str]] = field(default_factory=dict)
    pocket_files: list[str] = field(default_factory=list)
    docking_results: list[dict] = field(default_factory=list)
    step_count: int = 0
    completed: bool = False
    errors: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [
            f"Proteins: {', '.join(self.proteins)}",
            f"Ligand: {self.ligand_name} ({self.ligand_smiles})",
            f"Structures resolved: {sum(len(v) for v in self.resolved_structures.values())}",
            f"Pockets detected: {len(self.pocket_files)}",
            f"Docking results: {len(self.docking_results)}",
            f"Steps taken: {self.step_count}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class DockingOrchestrator:
    """Main orchestrator that drives the agentic docking workflow."""

    def __init__(
        self,
        proteins: list[str],
        ligand_smiles: str,
        ligand_name: str = "ligand",
        work_dir: Optional[Path] = None,
        use_alphafold: bool = False,
        use_vina: bool = True,
        use_diffdock: bool = False,
        exhaustiveness: int = 8,
        verbose: bool = True,
    ):
        self.state = AgentState(
            proteins=proteins,
            ligand_smiles=ligand_smiles,
            ligand_name=ligand_name,
            work_dir=work_dir or Path(f"results/agent_{int(time.time())}"),
        )
        self.use_alphafold = use_alphafold
        self.use_vina = use_vina
        self.use_diffdock = use_diffdock
        self.exhaustiveness = exhaustiveness
        self.verbose = verbose

        self.context = ConversationContext(model=ORCHESTRATOR_MODEL)
        self.worker_context = ConversationContext(model=WORKER_MODEL)

    def _log(self, msg: str, style: str = "bold cyan") -> None:
        if self.verbose:
            console.print(f"[{style}][Agent][/{style}] {msg}")

    def _log_tool(self, name: str, result: dict) -> None:
        status = result.get("status", "unknown")
        style = "green" if status == "success" else "red"
        if self.verbose:
            console.print(f"  [{style}]→ {name}: {status}[/{style}]")

    def setup(self) -> bool:
        """Verify Ollama is running and models are available."""
        self._log("Checking Ollama server...")
        if not check_ollama_health():
            console.print(
                "[bold red]Error:[/bold red] Ollama is not running. "
                "Start it with: [bold]ollama serve[/bold]"
            )
            return False

        self._log(f"Ensuring orchestrator model: {ORCHESTRATOR_MODEL}")
        if not ensure_model(ORCHESTRATOR_MODEL):
            console.print(f"[bold red]Failed to pull {ORCHESTRATOR_MODEL}[/bold red]")
            return False

        self._log(f"Ensuring worker model: {WORKER_MODEL}")
        if not ensure_model(WORKER_MODEL):
            console.print(f"[bold red]Failed to pull {WORKER_MODEL}[/bold red]")
            return False

        self.state.work_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Work directory: {self.state.work_dir}")
        return True

    def _build_initial_prompt(self) -> str:
        """Build the first user message describing the task."""
        docking_methods = []
        if self.use_vina:
            docking_methods.append("AutoDock Vina")
        if self.use_diffdock:
            docking_methods.append("DiffDock")
        if not docking_methods:
            docking_methods.append("LigToxML batch screening")

        methods_str = " and ".join(docking_methods)
        af_note = ""
        if self.use_alphafold:
            af_note = (
                "\nFor any proteins without PDB structures, use AlphaFold "
                "to predict their structures locally."
            )

        return (
            f"I need to dock a ligand against the following protein targets and "
            f"identify potential binding interactions.\n\n"
            f"**Proteins:** {', '.join(self.state.proteins)}\n"
            f"**Ligand SMILES:** {self.state.ligand_smiles}\n"
            f"**Ligand name:** {self.state.ligand_name}\n"
            f"**Docking method(s):** {methods_str}\n"
            f"**Work directory:** {self.state.work_dir}\n"
            f"{af_note}\n\n"
            f"Please proceed step by step:\n"
            f"1. Resolve each protein to PDB structure IDs\n"
            f"2. Download the structures\n"
            f"3. Prepare the ligand\n"
            f"4. Detect binding pockets\n"
            f"5. Run docking for each protein-ligand pair\n"
            f"6. Analyse and summarize the results\n"
        )

    def _agent_step(self) -> bool:
        """Execute one step of the agent loop. Returns False when done."""
        self.state.step_count += 1
        if self.state.step_count > MAX_AGENT_STEPS:
            self._log("Max steps reached, stopping.", "bold yellow")
            return False

        messages = self.context.to_ollama_messages()
        response = chat_completion(
            model=self.context.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.3,
        )

        text = extract_response_text(response)
        tool_calls = extract_tool_calls(response)

        # Show the agent's reasoning
        if text and self.verbose:
            console.print(Panel(
                Markdown(text),
                title=f"[bold]Agent Step {self.state.step_count}[/bold]",
                border_style="blue",
            ))

        # Record assistant message
        self.context.add("assistant", text, tool_calls=tool_calls if tool_calls else None)

        if not tool_calls:
            # No more tool calls — agent is done reasoning
            self._log("Agent finished (no more tool calls).", "bold green")
            self.state.completed = True
            return False

        # Execute each tool call
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            arguments = func.get("arguments", {})

            self._log(f"Calling tool: {tool_name}({json.dumps(arguments, default=str)[:200]})")
            result = execute_tool(tool_name, arguments)
            self._log_tool(tool_name, result)

            # Track state
            if tool_name == "resolve_pdb_ids" and result.get("status") == "success":
                self.state.resolved_structures.update(result.get("resolved", {}))
            elif tool_name == "detect_binding_pocket" and result.get("status") == "success":
                pf = result.get("pocket_file", "")
                if pf:
                    self.state.pocket_files.append(pf)
            elif tool_name in ("run_docking_vina", "run_docking_diffdock") and result.get("status") == "success":
                self.state.docking_results.append(result)
            elif result.get("status") == "error":
                self.state.errors.append(f"{tool_name}: {result.get('message', 'unknown error')}")

            # Feed tool result back to the agent
            self.context.add(
                "tool",
                json.dumps(result, default=str),
                tool_call_id=tool_name,
            )

        time.sleep(STEP_PAUSE)
        return True

    def run(self) -> AgentState:
        """Execute the full agentic workflow."""
        if not self.setup():
            return self.state

        # Initialize conversation
        self.context.add("system", SYSTEM_PROMPT)
        self.context.add("user", self._build_initial_prompt())

        self._log("Starting agentic docking workflow...", "bold magenta")
        console.print()

        # Agent loop
        while self._agent_step():
            pass

        # Print summary
        self._print_summary()
        return self.state

    def _print_summary(self) -> None:
        """Print a final summary of the workflow."""
        console.print()
        table = Table(title="Docking Workflow Summary", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Proteins", ", ".join(self.state.proteins))
        table.add_row("Ligand", f"{self.state.ligand_name} ({self.state.ligand_smiles})")
        table.add_row(
            "Structures found",
            str(sum(len(v) for v in self.state.resolved_structures.values())),
        )
        table.add_row("Pockets detected", str(len(self.state.pocket_files)))
        table.add_row("Docking results", str(len(self.state.docking_results)))
        table.add_row("Total steps", str(self.state.step_count))
        table.add_row("Errors", str(len(self.state.errors)))
        table.add_row("Status", "Completed" if self.state.completed else "Incomplete")

        console.print(table)

        if self.state.errors:
            console.print("\n[bold yellow]Errors encountered:[/bold yellow]")
            for err in self.state.errors:
                console.print(f"  [red]• {err}[/red]")

        if self.state.docking_results:
            console.print("\n[bold green]Docking Scores:[/bold green]")
            for i, res in enumerate(self.state.docking_results, 1):
                score = res.get("best_score", "N/A")
                path = res.get("output_path", "")
                console.print(f"  {i}. Score: {score} kcal/mol — {path}")

        # Save state to JSON
        state_path = self.state.work_dir / "agent_state.json"
        state_path.write_text(json.dumps({
            "proteins": self.state.proteins,
            "ligand_smiles": self.state.ligand_smiles,
            "ligand_name": self.state.ligand_name,
            "resolved_structures": self.state.resolved_structures,
            "pocket_files": self.state.pocket_files,
            "docking_results": self.state.docking_results,
            "step_count": self.state.step_count,
            "completed": self.state.completed,
            "errors": self.state.errors,
        }, indent=2, default=str))
        console.print(f"\n[dim]State saved to {state_path}[/dim]")


class ResultAnalyser:
    """Lightweight agent (4B model) for interpreting docking results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.context = ConversationContext(model=WORKER_MODEL)

    def interpret(self, question: str = "") -> str:
        """Use the smaller model to interpret results in natural language."""
        hits_csv = self.results_dir / "off_target_hits.csv"
        state_json = self.results_dir / "agent_state.json"

        data_context = ""
        if hits_csv.exists():
            import pandas as pd
            df = pd.read_csv(hits_csv)
            data_context += f"Docking results ({len(df)} entries):\n{df.head(20).to_string()}\n\n"
        if state_json.exists():
            state = json.loads(state_json.read_text())
            data_context += f"Agent state:\n{json.dumps(state, indent=2)}\n\n"

        if not data_context:
            return "No results found to analyse."

        prompt = (
            "You are a computational chemistry expert. Analyse these docking results "
            "and provide a concise scientific interpretation.\n\n"
            f"{data_context}"
        )
        if question:
            prompt += f"\nSpecific question: {question}\n"
        prompt += (
            "\nProvide:\n"
            "1. Summary of strongest binding interactions\n"
            "2. Off-target risk assessment\n"
            "3. Recommendations for follow-up experiments\n"
        )

        self.context.add("system", "You are a computational chemistry expert assistant.")
        self.context.add("user", prompt)

        response = chat_completion(
            model=WORKER_MODEL,
            messages=self.context.to_ollama_messages(),
            temperature=0.3,
        )
        return extract_response_text(response)
