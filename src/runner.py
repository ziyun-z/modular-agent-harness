"""
CLI entry point for the agent harness.

Usage:
    python -m src.runner --config configs/baseline.yaml --task <task_id>
    python -m src.runner --config configs/baseline.yaml --num-tasks 5 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Module registries  (config string → implementation class)
# ---------------------------------------------------------------------------
from src.memory.naive import NaiveMemory
from src.compression.none import NoCompression
from src.communication.single_agent import SingleAgentCommunication

MEMORY_REGISTRY: dict[str, type] = {
    "naive": NaiveMemory,
}
COMPRESSION_REGISTRY: dict[str, type] = {
    "none": NoCompression,
}
COMMUNICATION_REGISTRY: dict[str, type] = {
    "single_agent": SingleAgentCommunication,
}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def build_memory_module(cfg: dict):
    memory_type = cfg["memory"]["type"]
    if memory_type not in MEMORY_REGISTRY:
        raise ValueError(
            f"Unknown memory type '{memory_type}'. "
            f"Available: {list(MEMORY_REGISTRY)}"
        )
    params = cfg["memory"].get("params", {})
    return MEMORY_REGISTRY[memory_type](**params)


def build_compression_module(cfg: dict):
    compression_type = cfg["compression"]["type"]
    if compression_type not in COMPRESSION_REGISTRY:
        raise ValueError(
            f"Unknown compression type '{compression_type}'. "
            f"Available: {list(COMPRESSION_REGISTRY)}"
        )
    params = cfg["compression"].get("params", {})
    return COMPRESSION_REGISTRY[compression_type](**params)


def build_communication_module(cfg: dict):
    comm_type = cfg["communication"]["type"]
    if comm_type not in COMMUNICATION_REGISTRY:
        raise ValueError(
            f"Unknown communication type '{comm_type}'. "
            f"Available: {list(COMMUNICATION_REGISTRY)}"
        )
    params = cfg["communication"].get("params", {})
    return COMMUNICATION_REGISTRY[comm_type](**params)


def run_single_task(cfg: dict, task_id: str) -> None:
    from src.orchestrator import Orchestrator, OrchestratorConfig
    from src.llm_client import LLMClient
    from src.tool_executor import ToolExecutor
    from src.logger import TrajectoryLogger
    from src.evaluation.swebench_loader import SWEBenchLoader

    console.print(Panel(f"[bold]Task:[/bold] {task_id}", title="Agent Harness"))

    loader = SWEBenchLoader(cfg["evaluation"].get("dataset", "swebench_lite"))
    task = loader.get_task(task_id)

    memory = build_memory_module(cfg)
    compression = build_compression_module(cfg)
    communication = build_communication_module(cfg)

    orch_cfg = OrchestratorConfig(**cfg.get("orchestrator", {}))
    llm = LLMClient(model=cfg.get("model", "claude-sonnet-4-6"))
    sandbox_cfg = cfg.get("sandbox", {})
    tools = ToolExecutor(
        docker_image=sandbox_cfg.get("docker_image", "swebench-sandbox:latest"),
        timeout_per_task=sandbox_cfg.get("timeout_per_task", 600),
    )
    traj_logger = TrajectoryLogger()

    orchestrator = Orchestrator(
        config=orch_cfg,
        memory=memory,
        compression=compression,
        communication=communication,
        llm_client=llm,
        tool_executor=tools,
        logger=traj_logger,
    )

    result = orchestrator.run_task(task)

    status = "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"
    console.print(f"\nResult: {status} | Steps: {result.steps} | Cost: ${result.estimated_cost_usd:.4f}")
    if result.error:
        console.print(f"[yellow]Error: {result.error}[/yellow]")

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_id}.json"
    traj_logger.save(str(out_path))
    console.print(f"Trajectory saved to {out_path}")


def run_multi_task(cfg: dict, num_tasks: int, seed: int) -> None:
    from src.evaluation.swebench_loader import SWEBenchLoader

    loader = SWEBenchLoader(cfg["evaluation"].get("dataset", "swebench_lite"))
    task_ids = loader.sample_task_ids(num_tasks, seed=seed)

    console.print(f"Running {len(task_ids)} tasks...")
    results = []
    for task_id in task_ids:
        try:
            run_single_task(cfg, task_id)
        except Exception as e:
            console.print(f"[yellow]Task {task_id} crashed: {e}[/yellow]")

    passed = sum(1 for r in results if r.passed)
    console.print(f"\nSolve rate: {passed}/{len(task_ids)} ({100*passed/len(task_ids):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular Agent Harness for SWE-bench ablation studies"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--task", help="Specific SWE-bench task ID to run")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to sample and run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task sampling")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and validate without running any tasks",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    console.print(f"[dim]Loaded config: {args.config}[/dim]")
    console.print(f"[dim]Experiment: {cfg.get('experiment_name', 'unnamed')}[/dim]")

    if args.dry_run:
        console.print("[green]Dry run complete. Config is valid.[/green]")
        return

    if args.task:
        run_single_task(cfg, args.task)
    elif args.num_tasks:
        run_multi_task(cfg, args.num_tasks, args.seed)
    else:
        parser.error("Specify --task <id> or --num-tasks <n>")


if __name__ == "__main__":
    main()
