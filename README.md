# RL-XAR

RL-XAR is an independent implementation of the method described in Meta's [“Towards RL for Superhuman Text: Unslopping AI”](https://facebookresearch.github.io/RAM/blogs/unslop/). The public blog describes generating context-grounded rubrics, refining their meta-prompt against expert/model response pairs, and using the resulting rubrics as an RL reward. This repository implements a small configurable version of that loop. It is not Meta's code or a reproduction of Meta's data, models, results, or full method details.

The Python distribution, import package, and console command are named `rlxar`.

The repository contains a small synthetic dataset for checking the local GRPO path and a separate API-backed training pipeline. The demo uses a fixed rubric and a deterministic lexical reward. It does not learn or evaluate rubric quality, and its scores are not evidence of writing quality.

## Requirements and setup

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required. Set `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and `HF_HOME` to writable paths outside the checkout for the virtual environment, package cache, and downloaded model files. Choose paths available on your system; these values vary by machine.

PyTorch is a base dependency. On Linux and Windows, the locked configuration selects the explicit CPU wheel index. On macOS, it resolves PyTorch through PyPI; the available wheel requires Apple Silicon and macOS 14 or newer, and no compatible Intel Mac wheel is locked. GPU-specific PyTorch configuration is not included. For a platform-appropriate GPU setup, see the [official uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

```sh
uv sync --locked
```

Check the package and validate the checked-in dataset:

```sh
uv run python -c 'import rlxar; print(rlxar.__version__)'
uv run rlxar validate examples/demo.jsonl
```

The JSONL dataset has one object per nonblank line, with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Globally unique example identifier |
| `group_id` | string | Group that must remain within one split (for example, a source document) |
| `context` | string | Input shown to the writer and rubric generator |
| `expert` | string | Expert reference used by rubric optimization and evaluation |
| `split` | string | `train`, `validation`, or `test` |
| `metadata` | object, optional | Additional JSON data; defaults to `{}` |

Every dataset must have all three splits. IDs must be unique across the file, and all rows with the same `group_id` must use the same split. Assign groups before splitting so related examples cannot leak across train, validation, and test. Writer prompts and rubric generation receive context only; expert answers are used for pair scoring during meta-prompt optimization and for held-out evaluation.

## Quick local demo

```sh
uv run rlxar demo --use-cpu
```

The demo is configured to run one GRPO step over the synthetic train examples with the default `Qwen/Qwen2.5-0.5B-Instruct` writer, two generations, a frozen demo rubric, and `DemoJudge`. Set `--output DIR` to choose the runs root. By default, output goes to `RL_XAR_OUTPUT_DIR`; otherwise the CLI uses `/mnt/archive/runs/rl-xar` when that archive is writable, then falls back to `$XDG_DATA_HOME/rl-xar/runs` (normally `~/.local/share/rl-xar/runs`). A timestamped run directory and checkpoint are created there.

This is a small pipeline smoke command, not an instant or guaranteed one: a fresh locked sync downloads the Python dependencies, and the first demo downloads the Hugging Face model files. CPU generation and optimization time depends on the machine, and both downloads require network access. A completed demo does not show that learned rubrics are effective.

With a warm cache of the default model, the checked-in dataset, and one CPU step, the end-to-end demo completed in 8.49 seconds on the development host and saved an adapter. Cold dependency or model downloads are excluded from that timing.

## Full configured run

`examples/demo.toml` is a small example configuration. Before running it, replace `judge_model` with a valid LiteLLM model name and set the environment variable named by `judge_api_key_env` to your provider key. The example names `OPENAI_API_KEY`; configure that variable in your shell environment before invoking the command.

The configured run sends example contexts, expert references, generated candidate responses, and rubric/meta-prompt requests to the configured LiteLLM judge provider; the local `demo` makes no judge-provider calls.

```sh
uv run rlxar run --config examples/demo.toml
```

The example uses a small writer model, two rubric refinement iterations, one outer round, ten GRPO steps, and two generations. It writes timestamped run artifacts under the configured output root. Set `output_dir` in the TOML to choose that root explicitly. If it is omitted, the run uses `RL_XAR_OUTPUT_DIR`, then `/mnt/archive/runs/rl-xar` when the archive is writable, then `$XDG_DATA_HOME/rl-xar/runs` (normally `~/.local/share/rl-xar/runs`). Increase the data and training settings only after confirming the model, provider, and hardware can support the run. The configured judge must be able to generate rubrics, score responses, and revise meta-prompts; this can make a full run slow and incur API charges.

Other commands:

```sh
uv run rlxar --help
uv run rlxar validate path/to/dataset.jsonl
uv run rlxar run --config path/to/run.toml
uv run rlxar demo --help
```

See [docs/method.md](docs/method.md) for the implemented algorithm and source map.

## Limitations

- The implementation is independently reconstructed from Meta's public blog description. It is not affiliated with or endorsed by Meta, and it does not claim to reproduce Meta's data, models, results, or full method details.
- The checked-in demo data and deterministic demo reward only exercise the local training path. Tiny demo scores are not quality evidence.
- Full runs depend on the chosen writer, judge and rubric-generation model, provider behavior, data quality, compute, and configuration. A successful command alone does not establish that a rubric measures human quality or that the trained writer improved.
- The implemented test evaluation uses the configured judge and rubrics generated by the run's meta-prompts; it is not an independent human evaluation.

## License

The project is distributed under the [Apache License 2.0](LICENSE).
