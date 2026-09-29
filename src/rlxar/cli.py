"""Command line interface for RL-XAR."""

from __future__ import annotations

import os
import sys
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


def _judge_field(candidate: object, name: str, default: str = "unknown") -> str:
    """Read a display-only field without asking a candidate to reveal secrets."""
    if isinstance(candidate, dict):
        value = candidate.get(name, default)
    else:
        value = getattr(candidate, name, default)
    return str(value) if value is not None else default


def _show_judges(candidates: list[object]) -> None:
    typer.echo("Trance found these configured chat interfaces (discovery only):")
    for index, candidate in enumerate(candidates, start=1):
        typer.echo(
            f"  {index}. provider={_judge_field(candidate, 'provider')} "
            f"model={_judge_field(candidate, 'model_name')} "
            f"auth={_judge_field(candidate, 'auth_kind')} "
            f"source={_judge_field(candidate, 'source')}"
        )
    typer.echo("No model calls have been made. Choose one only if you want RL-XAR to use it.")


def _find_judges() -> list[object]:
    from .trance_judge import discover_judges

    try:
        return list(discover_judges())
    except Exception as exc:
        # Discovery may inspect credential stores; never echo exception text.
        typer.echo(
            f"Trance could not scan configured chat interfaces ({type(exc).__name__}).",
            err=True,
        )
        raise typer.Exit(code=1) from None


def _is_interactive() -> bool:
    """Return whether the current invocation can ask an interactive question."""
    return sys.stdin.isatty()


def _select_judge(candidates: list[object], index: int | None, approve: bool) -> object:
    interactive = _is_interactive()
    if index is None:
        if not interactive:
            typer.echo("Noninteractive use requires --judge-index and --approve-judge.", err=True)
            raise typer.Exit(code=2)
        index = typer.prompt("Enter the number of the judge to use", type=int)
    if index < 1 or index > len(candidates):
        typer.echo(f"Judge index must be between 1 and {len(candidates)}.", err=True)
        raise typer.Exit(code=2)
    if not approve:
        if not interactive:
            typer.echo("Noninteractive use requires --approve-judge.", err=True)
            raise typer.Exit(code=2)
        selected = candidates[index - 1]
        approve = typer.confirm(
            f"Use judge {index} ({_judge_field(selected, 'provider')} / "
            f"{_judge_field(selected, 'model_name')}) for this run?",
            default=False,
        )
    if not approve:
        typer.echo("Judge use was not approved; no run was started.")
        raise typer.Exit(code=0)
    return candidates[index - 1]


@app.command("judges")
def judges_command() -> None:
    """List configured chat interfaces found by Trance without making model calls."""
    candidates = _find_judges()
    if not candidates:
        typer.echo("Trance found no configured chat interfaces. No model calls were made.")
        raise typer.Exit(code=0)
    _show_judges(candidates)


@app.command("run")
def run_command(
    config: Path = typer.Option(
        ..., "--config", exists=True, file_okay=True, dir_okay=False,
        readable=True, resolve_path=True, help="TOML run configuration."
    ),
    judge_index: int | None = typer.Option(
        None, "--judge-index", min=1, help="1-based Trance judge selection."
    ),
    approve_judge: bool = typer.Option(
        False, "--approve-judge", help="Consent to spend the selected interface's rate limits."
    ),
) -> None:
    """Run training and evaluation from a TOML configuration."""
    try:
        from .pipeline import run_pipeline
        from .trance_judge import TranceChatClient

        candidates = _find_judges()
        if not candidates:
            typer.echo("Trance found no configured chat interfaces. No run was started.")
            raise typer.Exit(code=1)
        _show_judges(candidates)
        selected = _select_judge(candidates, judge_index, approve_judge)
        judge_client = TranceChatClient(selected)

        run_dir = run_pipeline(load_config(config), judge_client=judge_client)
    except typer.Exit:
        raise
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
