# Implemented method

RL-XAR here is a small independent implementation based on the public method description in Meta's [“Towards RL for Superhuman Text: Unslopping AI”](https://facebookresearch.github.io/RAM/blogs/unslop/). It does not use Meta's implementation or reproduce its experiments. The public blog describes context/continuation pairs, context-grounded rubric generation, meta-prompt refinement to increase expert-versus-model score gaps, and reinforcement learning against the resulting rubrics.

## Data and leakage boundaries

Each JSONL example contains an `id`, `group_id`, `context`, `expert`, and `split` (`train`, `validation`, or `test`), with optional `metadata`. `src/rlxar/schema.py` validates the records; `src/rlxar/data.py` reads JSONL and enforces unique IDs, all three splits, and one split per `group_id`. Group IDs should represent the source unit that must not cross splits, such as a document.

The policy rollout and rubric-generation prompts use the context without the expert answer. During rubric meta-prompt optimization, both expert and generated candidate responses are scored against the same generated rubric. Test examples are withheld from optimization and policy training; they are rolled out after the final configured round and passed to held-out evaluation.

## Outer-round loop

`src/rlxar/pipeline.py` coordinates each configured outer round:

1. **Generate continuations.** `src/rlxar/rollout.py` loads the writer model and optional current adapter, then greedily generates one continuation for each training and validation context.
2. **Refine rubric instructions.** `src/rlxar/optimizer.py` evaluates the current meta-prompt and its revisions on train pairs. `src/rlxar/rubrics.py` asks the chat model for a validated rubric using only each example's context. The user-selected Trance chat interface is called through Pydantic AI to score the expert and current policy continuation on each same rubric. It flags a training case when the candidate's weighted total ties or exceeds the expert's total within a `1e-6` tolerance, then sends those cases and the train expert-minus-policy gap to a meta-prompt reviser. It selects the prompt with the largest mean validation expert-minus-policy gap across the initial prompt and configured revisions.
3. **Build rubric rewards and train.** For each training context, the selected meta-prompt creates one context-only rubric. `src/rlxar/reward.py` parses the serialized rubric and returns the judge's weighted score for each generated completion. `src/rlxar/training.py` passes this reward to TRL's GRPO trainer with a LoRA adapter. Later rounds start from the previous round's adapter.
4. **Evaluate on test.** After the final round, the pipeline generates test continuations. `src/rlxar/evaluation.py` evaluates each continuation against the expert response for every round's selected meta-prompt, using a fresh context-only rubric and the same rubric for both responses. It records the expert gap and the policy-to-expert normalized score, including aggregate and worst rubric-set values. A zero expert score makes a positive policy/expert ratio undefined, which is recorded as `null` for that example.

This configured implementation runs the configured number of outer rounds; it has no automatic stopping rule based on a gap threshold. It uses the selected judge for both training-time rubric scoring and test evaluation, so the test metric is not independent of that judge. Before a run, `rlxar judges` asks Trance to discover configured chat interfaces and displays every candidate's canonical provider ID, exact model, authentication kind, and configuration source. Discovery does not make model inference requests or consume provider rate limits, although an interface may require Trance to invoke an installed CLI to inspect authentication status. Without a provider flag, `rlxar run` interactively selects and confirms a candidate. `--model-provider PROVIDER` is explicit consent for noninteractive use: it matches the provider against Trance's canonical IDs and uses the first matching discovery result, displaying all candidates and the exact selected model and source before making judge requests. The aliases `codex` and `grok` resolve to `openai-codex` and `grok-consumer`; the xAI API provider ID is `xai`. A noninteractive run fails closed when the flag is absent or no matching Trance result exists. Approved requests may consume rate limits.

## Main source files

- `src/rlxar/config.py`: strict TOML config parsing and path resolution.
- `src/rlxar/schema.py`, `src/rlxar/data.py`: example, rubric, judgment schemas and JSONL validation.
- `src/rlxar/rollout.py`: local writer generation.
- `src/rlxar/trance_judge.py`, `src/rlxar/client.py`, `src/rlxar/rubrics.py`, `src/rlxar/scoring.py`: Trance discovery and Pydantic AI chat client, context-only rubric creation, and weighted rubric judging.
- `src/rlxar/optimizer.py`: failure-driven meta-prompt revisions and validation-gap selection.
- `src/rlxar/reward.py`, `src/rlxar/training.py`: TRL-compatible rubric reward and LoRA GRPO training.
- `src/rlxar/evaluation.py`: held-out rubric-set scoring.
- `src/rlxar/artifacts.py`: timestamped run directories, JSON artifacts, and provenance manifests.
- `src/rlxar/demo.py`: the separate synthetic smoke path with a frozen rubric and deterministic lexical reward.

## Scope and evidence

The public blog is a high-level description, so implementation details here are reconstruction choices. The synthetic demo does not run the rubric-optimization loop and its deterministic reward is not an LLM quality judgment. An end-to-end one-step CPU demo was verified on this host using the checked-in synthetic dataset and a warm cache of the default `Qwen/Qwen2.5-0.5B-Instruct`; it took 8.49 seconds and saved an adapter. Cold dependency and model downloads were not included in that timing. The pipeline's round control flow was also exercised with real Qwen rollouts and training plus a deterministic fake judge; that 10.27-second check covered a rubric-revision path but does not verify live Pydantic AI judge calls. Neither demo scores nor API-pipeline metrics establish writing quality: they depend on the dataset, rubric generator, meta-prompt editor, writer, and configured judge. Use independent human assessment and appropriate held-out data before drawing quality conclusions.
