# Fleet Training Runbook

Quick reference for launching training runs on Fleet tasks.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Training Loop                           │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│   │  Sample  │───→│   Env    │───→│  Train   │                │
│   │  (LLM)   │    │  (Fleet) │    │  (GRPO)  │                │
│   └──────────┘    └──────────┘    └──────────┘                │
│        │                                │                      │
│        ▼                                ▼                      │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │ SkyRL:  Local vLLM + FSDP (your GPUs)                   │ │
│   │ Tinker: Remote APIs (hosted GPUs)                       │ │
│   └─────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

| Backend | Infrastructure | Best For |
|---------|---------------|----------|
| **SkyRL** | Self-managed GPU (Lambda/RunPod/Vast) | Full control, longer runs |
| **Tinker** | Hosted GPU (Theseus) | Quick iteration, no GPU setup |

## Data Sources

Tasks are stored in S3 and downloaded at workflow start:

| Dataset | S3 Path | Description |
|---------|---------|-------------|
| Tool Use | `s3://fleet-internal-datasets/v0.2/openenv/all_tool_use.json` | MCP tool-based tasks |
| Computer Use | `s3://fleet-internal-datasets/v0.2/openenv/all_computer_use.json` | Browser/desktop tasks |

Each task JSON contains:
- `task_key`: Unique identifier
- `prompt`: Task instruction
- `env_key`: Environment (e.g., `github`, `linear`, `notion`)
- `env_version`: Environment version
- `verifier_code`: Python verification function

## Launching a Run

### Via GitHub Actions

1. Go to **Actions** → Select workflow:
   - `Fleet Task Training (SkyPilot)` - SkyRL backend
   - `Fleet Task Training (Tinker)` - Tinker backend

2. Click **Run workflow** and configure:

#### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `modality` | `tool_use` or `computer_use` | `tool_use` |
| `env_key` | Filter by environment (empty=all) | `""` |
| `max_tasks` | Limit tasks for testing | `""` (all) |

#### SkyRL-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cloud` | `any`, `lambda`, `runpod`, `vast` | `any` |
| `task_name` | SkyPilot YAML in `skyrl-train/tasks/` | `openenv-fleet-grpo` |

#### Tinker-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model to train | `Qwen/Qwen3-8B` |
| `max_steps` | Training steps | `50` |
| `batch_size` | Batch size | `8` |
| `n_samples_per_prompt` | GRPO samples per prompt | `4` |

## Monitoring

### Slack Notifications

Channel: `#fleet-training-runs`

- **Started**: Run name, backend, modality, config
- **Completed**: Final status, WandB link
- **Failed**: Error logs link

### WandB Dashboards

| Backend | Project |
|---------|---------|
| SkyRL | [fleet-task-grpo](https://wandb.ai/thefleet/fleet-task-grpo) |
| Tinker | [fleet-tinker-grpo](https://wandb.ai/thefleet/fleet-tinker-grpo) |

### Key Metrics

| Metric | Description |
|--------|-------------|
| `reward/avg_pass_at_1` | Primary success metric |
| `reward/avg_pass_at_n` | Pass rate with n samples |
| `reward/avg_raw_reward` | Average reward per episode |
| `reward/{env_key}/pass_at_1` | Per-environment breakdown |

## Quick Test Run

Validate setup before full training:

```
modality: tool_use
env_key: github          # Single environment
max_tasks: 4             # Few tasks
max_steps: 10            # Tinker only
```

## Required Secrets

| Secret | Used By | Purpose |
|--------|---------|---------|
| `FLEET_API_KEY` | Both | Fleet environment access |
| `WANDB_API_KEY_TOOL_USE` | Both | Logging (tool_use) |
| `WANDB_API_KEY_COMPUTER_USE` | Both | Logging (computer_use) |
| `TINKER_API_KEY` | Tinker | Hosted inference/training |
| `LAMBDA_API_KEY` | SkyRL | Lambda Cloud GPUs |
| `RUNPOD_API_KEY` | SkyRL | RunPod GPUs |
| `VAST_API_KEY` | SkyRL | Vast.ai GPUs |
| `AWS_ACCESS_KEY_ID` | Both | S3 dataset access |
| `AWS_SECRET_ACCESS_KEY` | Both | S3 dataset access |
| `SLACK_BOT_TOKEN` | Both | Notifications |
