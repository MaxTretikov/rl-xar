"""End-to-end orchestration for an RL-XAR training run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from . import artifacts
from .client import LiteLLMClient
from .config import RunConfig
from .data import load_examples
from .schema import Example


def run_pipeline(
    config: RunConfig,
    *,
    dependencies: Mapping[str, Any] | None = None,
) -> Path:
    """Train successive adapters and evaluate the final policy on held-out data.

    ``dependencies`` may override named callable components (``rollout``,
    ``optimizer``, ``generate_rubric``, ``make_reward``, ``train_grpo``,
    ``evaluate_policy``, ``client_factory``, and ``judge_factory``). This keeps orchestration testable without model or
    API access while normal runs use the package implementations.
    """
    overrides = dependencies or {}
    rubrics = import_module(".rubrics", __package__)

    def component(key: str, module: str, attribute: str) -> Any:
        if key in overrides:
            return overrides[key]
        return getattr(import_module(module, __package__), attribute)

    rollout_fn = component("rollout", ".rollout", "generate_continuations")
    optimize_fn = component("optimizer", ".optimizer", "optimize_meta_prompt")
    rubric_fn = component("generate_rubric", ".rubrics", "generate_rubric")
    reward_fn = component("make_reward", ".reward", "make_grpo_reward")
    train_fn = component("train_grpo", ".training", "train_grpo")
    evaluate_fn = component("evaluate_policy", ".evaluation", "evaluate_policy")
    client_factory = overrides.get("client_factory", LiteLLMClient)
    judge_builder = component("judge_factory", ".scoring", "RubricJudge")

    examples = load_examples(config.dataset_path)
    splits: dict[str, list[Example]] = {
        split: [example for example in examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    for split_name, rows in splits.items():
        if not rows:
            raise ValueError(f"dataset split {split_name!r} is empty")
    if config.generations < 2:
        raise ValueError("generations must be at least 2 for GRPO")

    judge_model = config.judge_model
    if "/" not in judge_model and config.judge_provider != "litellm":
        judge_model = f"{config.judge_provider}/{judge_model}"
    judge_client = client_factory(
        judge_model,
        base_url=config.judge_base_url,
        api_key_env=config.judge_api_key_env,
    )
    judge = judge_builder(judge_client)

    config_data = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    } if hasattr(config, "__dict__") else {
        field: str(getattr(config, field)) if isinstance(getattr(config, field), Path)
        else getattr(config, field)
        for field in config.__dataclass_fields__
    }
    config_hash = artifacts.sha256_json(config_data)
    run_dir = artifacts.create_run_dir(
        "rl-xar",
        runs_root=config.output_dir,
        provenance="RL-XAR outer-round training pipeline",
        commands=["rlxar run"],
        config_hash=config_hash,
    )

    adapter_path: Path | None = None
    meta_prompts: list[str] = []
    round_manifests: list[dict[str, Any]] = []
    for round_index in range(config.outer_rounds):
        seed = config.seed + round_index
        train_outputs = rollout_fn(
            config.writer_model_id, splits["train"], adapter_path=adapter_path, seed=seed
        )
        validation_outputs = rollout_fn(
            config.writer_model_id,
            splits["validation"],
            adapter_path=adapter_path,
            seed=seed,
        )
        train_pairs = _pairs(splits["train"], train_outputs, "train")
        validation_pairs = _pairs(splits["validation"], validation_outputs, "validation")

        initial_meta_prompt = meta_prompts[-1] if meta_prompts else rubrics.DEFAULT_META_PROMPT
        optimization = optimize_fn(
            judge_client,
            judge,
            train_pairs,
            validation_pairs,
            initial_meta_prompt,
            iterations=config.rubric_iterations,
        )
        meta_prompt = _field(optimization, "best_prompt")
        if not isinstance(meta_prompt, str) or not meta_prompt.strip():
            raise ValueError("optimizer returned an empty best_prompt")
        meta_prompts.append(meta_prompt)

        rubric_rows = []
        records = []
        for example in splits["train"]:
            rubric_id = f"round-{round_index + 1}-{example.id}"
            rubric = rubric_fn(judge_client, meta_prompt, example.context, rubric_id=rubric_id)
            rubric_rows.append({"example_id": example.id, "rubric": rubric})
            records.append(
                {
                    "prompt": [{"role": "user", "content": example.context}],
                    "context": example.context,
                    "rubric_json": rubric.model_dump_json(),
                }
            )

        reward = reward_fn(judge, rubric_column="rubric_json")
        round_dir = run_dir / f"round-{round_index + 1:03d}"
        round_dir.mkdir(parents=True, exist_ok=False)
        artifacts.write_json(round_dir / "rubrics.json", rubric_rows)
        artifacts.write_json(round_dir / "meta_prompt.json", {
            "round": round_index + 1,
            "prompt": meta_prompt,
            "best_validation_gap": _field(optimization, "best_validation_gap", None),
            "history": _json_safe(_field(optimization, "history", [])),
        })
        adapter_path = Path(
            train_fn(
                config.writer_model_id,
                records,
                reward,
                round_dir / "adapter",
                steps=config.grpo_steps,
                generations=config.generations,
                seed=seed,
                adapter_path=adapter_path,
            )
        )
        round_manifests.append({
            "round": round_index + 1,
            "adapter_path": str(adapter_path),
            "meta_prompt_path": str(round_dir / "meta_prompt.json"),
            "rubrics_path": str(round_dir / "rubrics.json"),
            "train_count": len(records),
            "validation_count": len(validation_pairs),
        })

    test_outputs = rollout_fn(
        config.writer_model_id,
        splits["test"],
        adapter_path=adapter_path,
        seed=config.seed + config.outer_rounds,
    )
    test_pairs = _pairs(splits["test"], test_outputs, "test")
    evaluation_result = evaluate_fn(judge_client, judge, meta_prompts, splits["test"], test_outputs)
    artifacts.write_json(run_dir / "test_evaluation.json", evaluation_result)
    artifacts.write_json(run_dir / "test_continuations.json", {
        example.id: output for example, output in test_pairs
    })
    artifacts.write_json(run_dir / "run.json", {
        "config": config_data,
        "config_sha256": config_hash,
        "dataset_path": str(config.dataset_path),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "outer_rounds": round_manifests,
        "meta_prompt_versions": meta_prompts,
        "final_adapter_path": str(adapter_path) if adapter_path is not None else None,
        "test_evaluation": "test_evaluation.json",
    })
    return run_dir


def _pairs(
    examples: list[Example], outputs: Mapping[str, str], split_name: str
) -> list[tuple[Example, str]]:
    pairs: list[tuple[Example, str]] = []
    for example in examples:
        output = outputs.get(example.id)
        if not isinstance(output, str) or not output.strip():
            raise ValueError(
                f"rollout omitted a nonempty continuation for {split_name} example {example.id!r}"
            )
        pairs.append((example, output))
    return pairs


def _field(value: Any, name: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise ValueError(f"optimizer result is missing {name!r}")


def _json_safe(value: Any) -> Any:
    """Normalize optimization traces before passing them to artifact storage."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
