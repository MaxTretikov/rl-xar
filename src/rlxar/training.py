"""GRPO fine-tuning for rubric-guided response generation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def train_grpo(
    model_id: str,
    records: list[dict],
    reward_func: Callable[..., list[float]],
    output_dir: Path,
    *,
    steps: int = 1,
    generations: int = 2,
    seed: int = 42,
    adapter_path: Path | None = None,
    use_cpu: bool | None = None,
) -> Path:
    """Train a LoRA adapter with TRL's GRPOTrainer and save it to ``output_dir``.

    Each record must include ``prompt``, ``context``, and ``rubric_json``. The
    latter two columns are retained in the dataset and passed through to the
    reward function as lists, alongside TRL's ``prompts`` and ``completions``.
    ``adapter_path`` loads an existing PEFT adapter as the trainable starting
    point; it is not a Trainer checkpoint.
    """
    if not model_id:
        raise ValueError("model_id must be nonempty")
    if not records:
        raise ValueError("records must contain at least one example")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if generations < 2:
        raise ValueError("generations must be at least 2 for GRPO")
    if not callable(reward_func):
        raise TypeError("reward_func must be callable")

    dataset_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = {"prompt", "context", "rubric_json"} - record.keys()
        if missing:
            raise ValueError(
                f"record {index} is missing required field(s): {', '.join(sorted(missing))}"
            )
        prompt = record["prompt"]
        if not isinstance(prompt, (str, list)):
            raise TypeError(f"record {index} prompt must be text or a conversation")
        if not isinstance(record["context"], str):
            raise TypeError(f"record {index} context must be text")
        dataset_records.append(
            {
                "prompt": prompt,
                "context": record["context"],
                "rubric_json": record["rubric_json"],
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    # GRPO supports chat messages directly. Use that representation for plain
    # prompts when the selected tokenizer supplies a chat template.
    if tokenizer.chat_template:
        for row in dataset_records:
            if isinstance(row["prompt"], str):
                row["prompt"] = [{"role": "user", "content": row["prompt"]}]
    else:
        for index, row in enumerate(dataset_records):
            if isinstance(row["prompt"], list):
                messages = row["prompt"]
                if any(
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                    for message in messages
                ):
                    raise ValueError(
                        f"record {index} conversational prompt requires string role/content messages"
                    )
                row["prompt"] = "\n".join(
                    f"{message['role']}: {message['content']}" for message in messages
                )

    train_dataset = Dataset.from_list(dataset_records)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    # One full generation group per optimizer batch keeps the global batch
    # divisible by generations, including on a single CPU or GPU process.
    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": steps,
        "max_completion_length": 32,
        "num_generations": generations,
        "per_device_train_batch_size": generations,
        "gradient_accumulation_steps": 1,
        "generation_batch_size": generations,
        "seed": seed,
        "optim": "adamw_torch",
        "save_strategy": "no",
        "logging_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "use_vllm": False,
        "gradient_checkpointing": False,
    }
    if use_cpu is not None:
        config_kwargs["use_cpu"] = use_cpu
    args = GRPOConfig(**config_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return output_dir
