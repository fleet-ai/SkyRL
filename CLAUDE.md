# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learnings

The `learnings/` folder contains documented knowledge about SkyRL internals:
- `training-mechanics.md` — step_wise_trajectories, retokenize_chat_history, chat templates, loss reduction modes, thinking token stripping

**Always check learnings/ before asking questions about SkyRL training mechanics.**

## Critical Rules

1. **NEVER push to main** — Always create a branch and open a PR. Do NOT merge PRs; the user will review and merge.
2. **Always push to origin (fleet-ai/SkyRL)** — Never push to upstream (NovaSky-AI/SkyRL). When creating PRs, use `--repo fleet-ai/SkyRL`.
3. **DO NOT use Fleet SDK directly** — Use OpenEnv as the abstraction layer. Use OpenEnv's `FleetTaskEnv` from `envs.fleet_env.task_env` for Fleet task integration.
4. **Always look at API documentation** — Do not guess flags or API parameters. Read actual docs/source first.

## Build & Install

Uses `uv` (>=0.8.10) as the package manager. Python 3.12 required.

```bash
# Install skyrl-train with dev dependencies
cd skyrl-train && uv sync --extra dev

# Install with an inference backend (vllm and sglang are mutually exclusive)
cd skyrl-train && uv sync --extra dev --extra vllm
cd skyrl-train && uv sync --extra dev --extra sglang

# Install skyrl-gym
cd skyrl-gym && uv sync --extra dev

# Install skyrl-tx
cd skyrl-tx && uv sync
```

## Formatting & Linting

Pre-commit hooks: ruff (with --fix), black (line-length 120), gitleaks (secret detection). Ruff excludes `skyrl-agent/`.

```bash
# Run all pre-commit checks (do this before every PR)
pre-commit run --all-files --config .pre-commit-config.yaml

# Or use the format script
bash format.sh
```

If pre-commit reformats files, stage them and commit again.

## Tests

```bash
# skyrl-train CPU tests (what CI runs)
cd skyrl-train && uv run --frozen pytest tests/cpu/ -v

# Single test file
cd skyrl-train && uv run --frozen pytest tests/cpu/test_trainer_utils.py -v

# skyrl-gym tests
cd skyrl-gym && uv run --frozen pytest tests/

# GPU tests (requires GPU)
cd skyrl-train && uv run --frozen pytest tests/gpu/
```

Pytest markers: `vllm`, `sglang`, `integrations`, `megatron`.

CI runs code quality checks first (ruff, black, gitleaks), then CPU tests for skyrl-train and skyrl-gym in parallel.

## Architecture

SkyRL is a monorepo with four independent packages:

**skyrl-train** — RL training framework. The core package.
- `entrypoints/main_base.py` — Hydra-based training entry point
- `trainer.py` — `RayPPOTrainer`: main training loop, advantage computation, weight updates
- `fully_async_trainer.py` — Async variant with in-flight weight updates
- `generators/skyrl_gym_generator.py` — Agent loop for multi-turn rollouts (init env → generate → env.step → repeat)
- `generators/base.py` — `GeneratorInterface` abstract class; `GeneratorInput`/`GeneratorOutput` types
- `inference_engines/` — Backends: `vllm/`, `sglang/`, `ray_wrapped_inference_engine.py`, `remote_inference_engine.py`
- `workers/` — Ray actors for policy/ref/critic training; `fsdp/` and `megatron/` strategy implementations
- `distributed/` — `DistributedStrategy` abstraction with FSDP2 and Megatron backends
- `weight_sync/` — Weight sync between inference and training: NCCL, gloo, CUDA IPC, broadcast
- `dataset/` — `PromptDataset`, preprocessing, replay buffer for async training
- `utils/ppo_utils.py` — PPO/GRPO loss functions, advantage estimators (GRPO, GAE, RLOO, REINFORCE++), KL controllers
- `model_wrapper.py` — `HFModelWrapper`: model loading, LoRA, attention configuration
- `config/ppo_base_config.yaml` — Default Hydra config with all settings

**skyrl-gym** — Gymnasium-style environments for RL tasks.
- `core.py` — Base `Env` class
- `envs/` — Implementations: `gsm8k/`, `aime/`, `search/`, `sql/`, `lcb/` (LiveCodeBench), `searchcode/`
- `tools/` — Tool implementations (search, SQL, code execution)
- `metrics.py` — Environment metrics

**skyrl-agent** — Agent layer for long-horizon tasks.
- `agents/` — Agent implementations
- `tasks/` — Task definitions (SWE-Bench, MemAgent, WebResearch)
- `dispatcher/` — Async dispatching
- `integrations/` — Backend integrations (skyrl-train, verl, tinker)
- `auto.py` — `AutoAgentRunner`

**skyrl-tx** — Tinker-like REST API backend for post-training (JAX-based).
- `tinker/` — Engine, API, database models, JAX backend
- `models/` — Model implementations (Qwen3, Llama3)
- CLI entry point: `tx` command

## Configuration System

Hydra-based. Root config: `skyrl-train/skyrl_train/config/ppo_base_config.yaml`.

Key config sections:
- `data` — train/val data paths
- `trainer` — placement (nodes, GPUs, colocate), strategy (fsdp2/megatron), policy/ref/critic model configs
- `trainer.algorithm` — advantage estimator, loss_reduction mode, KL control
- `generator` — inference backend, sampling params, weight_sync_backend, step_wise_trajectories
- `environment` — environment class, skyrl_gym settings

Override via command line:
```bash
python -m skyrl_train.entrypoints.main_base \
    +generator.engine_init_kwargs.chat_template=/path/to/template.jinja2 \
    generator.step_wise_trajectories=true \
    trainer.algorithm.loss_reduction=token_mean
```

## Training

Example: GSM8K with GRPO on Qwen2.5-1.5B (see `examples/gsm8k/run_gsm8k.sh`):
```bash
cd skyrl-train && bash examples/gsm8k/run_gsm8k.sh
```

## Inference Backend Extras

`vllm`, `sglang`, `mcore` (Megatron), `flashrl` are mutually exclusive uv extras — only one can be installed at a time. The `miniswe` extra conflicts with `flashrl`.

## Fleet Integration Pattern

Tinker integration code (`integrations/fleet/`) must mirror `skyrl_train/generators/skyrl_gym_generator.py`:
- Same parameter names: `max_input_length`, `max_generate_length`, `max_sequence_length`
- Same context length handling: end rollout when `len(input_ids) > max_input_length`
- DAPO overlong filtering: truncate sequences > `max_sequence_length`, zero out loss mask
- Same metrics naming: `pass_at_n`, per-environment metrics

## WandB Project Naming

Use distinct WandB project names per training focus to keep runs organized:

| Training Focus | `trainer.project_name` | WandB URL |
|---|---|---|
| All environments (general) | `fleet-task-grpo` | wandb.ai/thefleet/fleet-task-grpo |
| Wallst only | `fleet-wallst-grpo` | wandb.ai/thefleet/fleet-wallst-grpo |
| Jira + Outlook | `fleet-jira-outlook-grpo` | wandb.ai/thefleet/fleet-jira-outlook-grpo |

Pattern: `fleet-{env_scope}-grpo`. When adding a new env-specific training config, create a new project name following this convention.

## Training Run Lessons Learned

**Failure modes observed (Feb 2026):**

1. **Disk exhaustion at checkpoint save (OSError [Errno 28])** — Run `wallst_tool_use_cc566a69` crashed at `global_step_21`. With `keep_n=2` and S3 enabled, saving a 3rd checkpoint exceeded 200GB disk. **Fix:** `s3_checkpoints.py:219` — set `keep_n=1` when S3 is enabled since checkpoints are backed up to S3. Old checkpoints are cleaned up *before* saving the new one.

2. **GPU unavailability (ResourcesUnavailableError)** — B200:4 and H200:4 instances are scarce. SkyPilot tries all clouds (Lambda, RunPod, Vast, PrimeIntellect, Nebius) but can fail if none have capacity. Runs fail instantly (~2min) with no training. **Mitigation:** Just retry later. Consider off-peak hours.

3. **PrimeIntellect insufficient funds** — `API request failed: Insufficient funds in the wallet. Required: $1.00, Available: $0.00`. PrimeIntellect wallet needs funding. This blocks that cloud as a fallback.

4. **Self-hosted runner disconnection** — Runner loses communication mid-job, causing instant failure unrelated to training code. **Mitigation:** Retry the workflow run.

5. **Nebius not configured on runner** — `ImportError: Failed to import dependencies for Nebius AI Cloud`. Runner needs `pip install "skypilot[nebius]"`. This is non-fatal (logged as warning) but reduces available cloud options.

**Pre-launch checklist:**
- Verify S3 dataset exists: `aws s3 ls s3://fleet-internal-datasets/{DATA_VERSION}/openenv/`
- Check GPU availability is reasonable (evenings/weekends tend to be better)
- Ensure the `workdir.ref` in your task YAML points to the correct branch
- Confirm the self-hosted runner is healthy before triggering

## Eval Trajectory Analysis

When documenting training runs, analyze eval trajectories from S3:

```bash
# Download trajectories
aws s3 cp s3://skyrl-trajectories/evals/{run_name}/global_step_0/{env}.jsonl /tmp/{env}_step0.jsonl

# Get aggregated results
aws s3 cp s3://skyrl-trajectories/evals/{run_name}/global_step_X/aggregated_results.jsonl -
```

Trajectory fields: `input_prompt`, `output_response`, `score` (per-token list), `stop_reason`, `env_extras` (contains `task_key`, `data_source`).

Key metrics per environment: avg turns (count `<|im_start|>assistant` in output_response), avg tool calls (count `"name":` patterns), success rate (from `aggregated_results.jsonl` pass@3).

Categorize failures as: tool argument errors, authorization errors, resource not found, false completions, context overflow. Document results in `fleet-research/threads/tool-use-training/experiments.md`.
