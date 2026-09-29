# rlxar

`rlxar` is an implementation of Meta’s [Reinforcement Learning from eXpert-Aligned Rubrics](https://facebookresearch.github.io/RAM/blogs/unslop/).
For each context, it generates a rubric without exposing the expert answer, revises the rubric meta-prompt from training failure cases and the train expert-minus-policy gap, selects among the candidate prompts by validation expert-minus-policy gap, and uses the resulting judge scores as scalar rewards for GRPO updates to a local writer with a LoRA adapter.

<img src="docs/rlxar-architecture.svg" alt="RL-XAR architecture" width="100%">

## Quick start

### Install and validate the example data

Python 3.11 or newer is required. Install the project with [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync
uv run rlxar validate examples/demo.jsonl
```

The example dataset has 8 training, 2 validation, and 2 test examples. Every JSONL row must contain `id`, `group_id`, `context`, `expert`, and one of `train`, `validation`, or `test` as `split`. IDs must be unique, all three splits must be present, and one `group_id` cannot appear in multiple splits.

```sh
uv run rlxar demo --use-cpu --steps 1 --output /tmp/rlxar-demo
```

This downloads the default Hugging Face writer (`Qwen/Qwen2.5-0.5B-Instruct`) if needed, runs one local GRPO step, and writes a timestamped run directory beneath `/tmp/rlxar-demo/`; that directory contains `manifest.json` and a `checkpoint/` subdirectory containing the LoRA adapter. It uses a frozen rubric and deterministic lexical reward, so it is a wiring smoke test rather than evidence of writing quality. Omit `--use-cpu` to let the demo select CUDA when available.

### Run the configured pipeline

The full pipeline uses a local Hugging Face causal language model as the trainable writer and a chat model for rubric generation, rubric scoring, meta-prompt revision, and held-out evaluation. The writer must be loadable locally by Transformers because it is generated from and LoRA-trained by the process. Chat interfaces are discovered through [Trance](https://github.com/MaxTretikov/trance) and used through Pydantic AI.

Judge settings do not belong in the TOML file. At the start of every `run`,
`rlxar` asks Trance to scan the chat interfaces configured on the machine.
Trance recognizes supported local CLI logins, provider accounts, and API-key
environments, then reports each result's canonical provider ID, exact model,
authentication kind, and configuration source.
Discovery does not send an inference request. It may inspect local
account state or invoke an installed CLI as part of checking a supported source.

Copy the example configuration and configure any intended provider interface
before starting discovery. If the interface requires an API key, set it first;
for example:

```sh
cp examples/demo.toml examples/local.toml
export OPENAI_API_KEY='your-api-key'
```

Then start the run:

```sh
uv run rlxar run --config examples/local.toml
```

This command discovers the available judges and, in an interactive terminal,
asks you to choose one and confirm that RL-XAR may use it. The optional
preflight command prints the same discovery results without starting a run:

```sh
uv run rlxar judges
```

Seeing a credential or account does not prove that it is valid, has remaining
quota, or is permitted to make the requested calls. Configure any required
credentials before discovery as shown above.

For automation, `--model-provider PROVIDER` is explicit consent and makes the
run noninteractive:

```sh
uv run rlxar run --config examples/local.toml --model-provider openai
```

RL-XAR matches the value against Trance's canonical provider IDs and uses the
first matching result. It prints all discovered candidates and the exact
selected model and source before making judge requests. The aliases `codex` and
`grok` select `openai-codex` and `grok-consumer`; the xAI API provider ID is
`xai`. A missing provider match fails closed, so no run starts. After consent,
Pydantic AI sends the actual rubric and evaluation requests through the chosen
Trance model; those requests may consume provider quota or rate limits.

A noninteractive run without `--model-provider` also fails closed, because it
cannot obtain the required consent interactively.

Set `output_dir` in the TOML or `RL_XAR_OUTPUT_DIR` in the environment to choose the external run root. Otherwise runs go to `/mnt/archive/runs/rl-xar` when that location is writable, or `$XDG_DATA_HOME/rl-xar/runs`. A completed run contains `manifest.json`, `round-*/adapter/`, `round-*/rubrics.json`, `round-*/meta_prompt.json`, `test_continuations.json`, `test_evaluation.json`, and `run.json`.

The configured defaults are three rubric iterations, three outer rounds, 100 GRPO steps per round, four generations, and seed 42. GRPO requires at least two generations. Reduce `outer_rounds`, `rubric_iterations`, `grpo_steps`, and `generations` for a smaller experiment; the example config already uses one outer round, two rubric iterations, ten steps, and two generations.

## Python API

The main building blocks are available under `rlxar`: data loading and schema validation, local writer rollouts, rubric generation, rubric judging, GRPO reward construction, and pipeline orchestration. The CLI exposes:

```sh
uv run rlxar --help
uv run rlxar validate PATH_TO_DATASET.jsonl
uv run rlxar judges
uv run rlxar demo --help
uv run rlxar run --help
```
