"""Command line interface for RL-XAR."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from .config import load_config
from .data import load_examples

app = typer.Typer(
    name="rlxar",
    help="Train and validate rubric-guided RL-XAR runs.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _demo_output_default() -> Path:
    """Choose an external default directory, with an environment override."""
    configured = os.environ.get("RL_XAR_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser()
    archive_runs = Path("/mnt/archive/runs")
    if archive_runs.is_dir() and os.access(archive_runs, os.W_OK):
        return archive_runs / "rl-xar"
    data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return data_home / "rl-xar" / "runs"


_DEFAULT_DEMO_OUTPUT = _demo_output_default()


def _report_value_error(exc: ValueError) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2) from exc


@app.command("run")
def run_command(
    config: Path = typer.Option(
        ..., "--config", exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True, help="TOML run configuration."
    ),
) -> None:
    """Run training and evaluation from a TOML configuration."""
    try:
        from .pipeline import run_pipeline

        run_dir = run_pipeline(load_config(config))
    except ValueError as exc:
        _report_value_error(exc)
    except Exception as exc:
        # Provider errors may include request data or credentials; keep them out of CLI output.
        typer.echo(
            f"Run failed ({type(exc).__name__}). Check your configuration and runtime logs.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    typer.echo(f"Run complete: {run_dir}")


@app.command("validate")
def validate_command(
    path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True, help="JSONL dataset path."
    ),
) -> None:
    """Validate a JSONL dataset and report its split sizes."""
    try:
        examples = load_examples(path)
    except ValueError as exc:
        _report_value_error(exc)
    counts = {
        split: sum(example.split == split for example in examples)
        for split in ("train", "validation", "test")
    }
    typer.echo(
        f"Valid dataset: {len(examples)} examples "
        f"(train={counts['train']}, validation={counts['validation']}, test={counts['test']})"
    )


@app.command("demo")
def demo_command(
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model",
        help="Hugging Face model ID."
    ),
    output_dir: Path = typer.Option(
        _DEFAULT_DEMO_OUTPUT, "--output", file_okay=False,
        help="Output directory (defaults to RL_XAR_OUTPUT_DIR or XDG data storage)."
    ),
    dataset_path: Path | None = typer.Option(
        None, "--dataset", exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True, help="Optional JSONL dataset override."
    ),
    steps: int = typer.Option(1, "--steps", min=1, help="Number of demo training steps."),
    use_cpu: bool | None = typer.Option(
        None, "--use-cpu/--no-use-cpu",
        help="Force CPU when enabled; otherwise let the demo choose a device."
    ),
) -> None:
    """Run a small local demonstration without a run configuration."""
    try:
        from .demo import run_demo

        run_dir = run_demo(
            model_id=model_id,
            output_dir=output_dir.expanduser(),
            dataset_path=dataset_path,
            steps=steps,
            use_cpu=use_cpu,
        )
    except ValueError as exc:
        _report_value_error(exc)
    except Exception as exc:
        typer.echo(f"Demo failed ({type(exc).__name__}). Check your runtime logs.", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Demo complete: {run_dir}")
