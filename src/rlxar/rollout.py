"""Local text generation for RL-XAR examples."""

from __future__ import annotations

from pathlib import Path

from .data import writer_prompt
from .schema import Example


def generate_continuations(
    model_id: str,
    examples: list[Example],
    *,
    adapter_path: Path | None = None,
    max_new_tokens: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, str]:
    """Generate one continuation per example using its context as the prompt.

    The base model, tokenizer, and optional PEFT adapter are loaded once and
    reused for all examples. Generation uses greedy decoding for repeatable
    results, including on CPU.
    """
    if not model_id.strip():
        raise ValueError("model_id must be nonempty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    seen_ids: set[str] = set()
    for example in examples:
        if example.id in seen_ids:
            raise ValueError(f"duplicate example id {example.id!r}")
        seen_ids.add(example.id)
    if not examples:
        return {}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required for local generation."
        ) from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required when adapter_path is provided.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(target_device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    results: dict[str, str] = {}
    for example in examples:
        context = writer_prompt(example)
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = context

        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()
        if not continuation:
            raise RuntimeError(
                f"model produced an empty continuation for example {example.id!r}"
            )
        results[example.id] = continuation

    return results
