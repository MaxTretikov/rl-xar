"""Small local GRPO smoke pipeline for RL-XAR.

This exercises data loading, a deterministic local reward, and training wiring.
It does not measure whether a learned rubric is useful or high quality.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import create_run_dir, sha256_json, write_json
from .data import load_examples, writer_prompt
from .reward import DemoJudge, make_grpo_reward
from .schema import Criterion, Rubric
from .training import train_grpo

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SOURCE_DEMO_DATASET_PATH = Path(__file__).resolve().parents[2] / "examples" / "demo.jsonl"
PACKAGED_DEMO_DATASET_PATH = Path(__file__).parent / "_data" / "demo.jsonl"

# This fixed rubric is only a schema-compatible input for exercising the
# training path. DemoJudge uses a cheap deterministic lexical score.
_DEMO_RUBRIC = Rubric(
    id="demo-static-v1",
    criteria=[
        Criterion(
            id="relevance",
            instruction="Respond with text related to the supplied context.",
            weight=0.5,
        ),
        Criterion(
            id="substance",
            instruction="Provide a nonempty, informative response.",
            weight=0.5,
        ),
    ],
    provenance="Frozen synthetic demo rubric; not learned or evaluated.",
)


def run_demo(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: Path,
    dataset_path: Path | None = None,
    steps: int = 1,
    use_cpu: bool | None = None,
) -> Path:
    """Run a minimal local GRPO smoke job and return its checkpoint directory.

    ``output_dir`` is used as the external runs root; a timestamped run
    directory and checkpoint are created beneath it. When ``dataset_path`` is
    omitted, the checked-in synthetic fixture is used, with a bundled copy as
    the fallback for installed wheels.
    """
    if not model_id.strip():
        raise ValueError("model_id must be nonempty")
    if isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")

    if dataset_path is not None:
        source = Path(dataset_path)
    elif SOURCE_DEMO_DATASET_PATH.is_file():
        source = SOURCE_DEMO_DATASET_PATH
    else:
        source = PACKAGED_DEMO_DATASET_PATH
    train_examples = [example for example in load_examples(source) if example.split == "train"]
    if not train_examples:
        raise ValueError(f"dataset {source} contains no train examples")

    rubric_json = _DEMO_RUBRIC.model_dump_json()
    records = [
        {
            "prompt": [{"role": "user", "content": writer_prompt(example)}],
            "context": example.context,
            "rubric_json": rubric_json,
        }
        for example in train_examples
    ]
    run_dir = create_run_dir(
        "demo",
        runs_root=Path(output_dir),
        provenance=(
            "Local pipeline/training smoke using synthetic train examples, a frozen static "
            "rubric, and deterministic DemoJudge. This run is not evidence of learned "
            "rubric quality."
        ),
        commands=["run_demo"],
        rubric_hash=sha256_json(_DEMO_RUBRIC),
        config_hash=sha256_json(
            {
                "model_id": model_id,
                "dataset_path": source.resolve(),
                "steps": steps,
                "generations": 2,
                "use_cpu": use_cpu,
            }
        ),
    )
    manifest = run_dir / "manifest.json"
    # Extend the common artifact manifest with the exact local inputs.
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data.update(
        {
            "model_id": model_id,
            "dataset_path": source.resolve(),
            "train_example_ids": [example.id for example in train_examples],
            "steps": steps,
            "generations": 2,
            "use_cpu": use_cpu,
            "rubric": _DEMO_RUBRIC,
        }
    )
    write_json(manifest, manifest_data)

    checkpoint_dir = run_dir / "checkpoint"
    result = Path(
        train_grpo(
            model_id,
            records,
            make_grpo_reward(DemoJudge()),
            checkpoint_dir,
            steps=steps,
            generations=2,
            use_cpu=use_cpu,
        )
    )
    manifest_data["checkpoint_paths"] = [result]
    write_json(manifest, manifest_data)
    return result
