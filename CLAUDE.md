# Claude Code Instructions for SkyRL

## Project Context

SkyRL is a reinforcement learning training framework. Fleet integration uses OpenEnv's `FleetTaskEnv` as the abstraction layer.

- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- OpenEnv provides: `FleetEnvClient`, `FleetMCPTools`, `FleetTaskEnv`, `make_fleet_task_env`

## Documentation

### Experiment Log
- Location: `~/repos/fleet-training/experiments.md`
- Contains: dataset versions, method, changelog, analysis, future work
- **Always tag experiments with PR numbers** (e.g., `**PR:** [#122](https://github.com/fleet-ai/SkyRL/pull/122)`)

### Versioning Scheme
`v{dataset}.{iteration}` - dataset version precedes iteration number
- v0.2.x = experiments on v0.2 dataset (~3.4K tasks)
- v0.3.x = experiments on v0.3 dataset (10K+ tasks)
- Example: v0.3.2 = second iteration on v0.3 dataset

### Learnings
The `learnings/` folder contains documented knowledge about SkyRL internals:
- `training-mechanics.md` - step_wise_trajectories, retokenize_chat_history, chat templates, loss reduction

**Always check learnings/ before asking questions about SkyRL training mechanics.**

## Critical Rules

1. **NEVER push to main** - Always create a branch and open a PR. Do NOT merge PRs - user will review and merge.

2. **Always push to origin (fleet-ai/SkyRL)** - Never push to upstream (NovaSky-AI/SkyRL). Use `--repo fleet-ai/SkyRL` for PRs.

3. **Always run black before commits**:
   ```bash
   black skyrl-train/integrations/fleet/
   ```
   Or full pre-commit:
   ```bash
   pre-commit run --all-files --config .pre-commit-config.yaml
   ```

4. **DO NOT use Fleet SDK directly** - Use OpenEnv as the abstraction layer.
   - Fleet SDK repo: `/Users/deniz/repos/fleet-sdk`
   - OpenEnv repo: `/Users/deniz/repos/OpenEnv`

5. **Always look at API documentation** - Do not guess flags or parameters.

6. **Tinker integration must mimic skyrl_gym_generator** - Match patterns from `skyrl_train/generators/skyrl_gym_generator.py`:
   - Same parameter names: `max_input_length`, `max_generate_length`, `max_sequence_length`
   - Same context length handling: `if len(input_ids) > max_input_length` to end rollout
   - DAPO overlong filtering: truncate sequences > `max_sequence_length` and zero out loss mask
   - Same metrics naming: `pass_at_n`, per-environment metrics

## Fleet Task Integration Pattern

```python
from envs.fleet_env import FleetTaskEnv, make_fleet_task_env

task_config = {
    "task_key": "...",
    "prompt": "...",
    "env_key": "...",
    "env_version": "...",
    "verifier_code": "...",
    "task_modality": "tool_use",
}
env = FleetTaskEnv(task_config, api_key=...)

# Gymnasium-style interface
obs = env.reset()
obs, reward, done, info = env.step(action)
env.close()
```

## Credentials

**WandB:** `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`

## Trajectory Analysis Reference

For each iteration, analyze trajectory metrics like this:

| Environment | Avg Tools | Tools/Turn | Avg Turns | Episodes |
|-------------|-----------|------------|-----------|----------|
| reddit | 26.5 | 0.98 | 27.0 | 4 |
| google-maps | 10.2 | 0.89 | 11.5 | 4 |
| github | 8.8 | 0.88 | 10.0 | 12 |
| booking | 7.4 | 0.88 | 8.4 | 16 |
| hubspot | 7.2 | 0.88 | 8.3 | 4 |
| fira | 7.0 | 0.88 | 8.0 | 4 |
| zillow | 7.0 | 0.88 | 8.0 | 4 |
| outlook | 6.5 | 1.00 | 6.5 | 4 |
| ticketmaster | 2.2 | 0.69 | 3.3 | 4 |

**Sequence Length Metrics:**

| Metric | Value |
|--------|-------|
| max_num_tokens | 22,345 |
| avg_num_tokens | 10,509 |
| std_num_tokens | 4,818 |
| min_num_tokens | 4,714 |
