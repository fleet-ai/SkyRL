# SkyRL Fleet Training: Tool-Use and Browser-Use Agent Development

## 1. Overview

Training an 8B parameter model (Qwen3-8B) on Fleet tool-use tasks using SkyRL's GRPO algorithm. Evaluation on held-out Fleet environments to measure cross-environment generalization.

## 2. Dataset

### v0.2 (Current)

**Summary**: 3,485 tasks (3,409 train / 76 eval)
- 95% with env_variables (`CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME`)
- Held-out environment: `outlook`
- Fixes: YaRN rope_scaling (65K context), system prompt improvements (see [changelog.md](changelog.md))

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| booking | 1,070 | 19 | 1,089 |
| dropbox | 1 | 0 | 1 |
| fira | 51 | 0 | 51 |
| github | 1,827 | 38 | 1,865 |
| google-maps | 7 | 0 | 7 |
| hubspot | 10 | 0 | 10 |
| outlook | 0 | 19 | 19 |
| reddit | 188 | 0 | 188 |
| ticketmaster | 222 | 0 | 222 |
| zillow | 33 | 0 | 33 |
| **TOTAL** | **3,409** | **76** | **3,485** |

**S3 path**: `s3://fleet-internal-datasets/v0.2/openenv/all_tool_use.json`

### Data Preparation Pipeline

**Task Validation Process**:
1. **Tool call replay**: Each task's tool calls are replayed against the Fleet environment to ensure they execute successfully
2. **Teacher model verification**: Tasks are validated by running a teacher model (e.g., Claude) to confirm solvability
3. **Filtering**: Tasks that fail replay or teacher verification are excluded from the training set

This ensures the training data contains only valid, solvable tasks with working tool chains.

## 3. Results

### Latest: Run `fleet_tool_use_a7e85045` (Step 80)

**Trajectories**: `s3://skyrl-trajectories/evals/fleet_tool_use_a7e85045/`

| Environment | Step 0 | Step 80 | Delta | Status |
|-------------|--------|---------|-------|--------|
| booking | 53.8% | 57.7% | +3.8% | ➖ FLAT |
| fira | 40.0% | 40.0% | +0.0% | ➖ FLAT |
| github | 46.7% | **70.0%** | **+23.3%** | 📈 IMPROVED |
| hubspot | 100.0% | 100.0% | +0.0% | ✅ SATURATED |
| outlook | 100.0% | 100.0% | +0.0% | ✅ SATURATED |
| reddit | 66.7% | 66.7% | +0.0% | ➖ FLAT |
| ticketmaster | 3.6% | 7.1% | +3.6% | ❌ FAILING |
| zillow | 50.0% | 50.0% | +0.0% | ➖ FLAT |
| **OVERALL** | 43.1% | **50.9%** | **+7.8%** | |

**Key Issues:**
- ticketmaster: Only 1 unique eval task (dataset bug), complex 5+ step workflow
- hubspot/outlook: Saturated at step 0, no learning signal
- github: Best improvement, training effective

---

## 4. TODO

- [ ] **Hillclimbing on tool-use eval set**: Iterate on model/training to improve eval performance
- [ ] **Scaling dataset**: Increase training samples to demonstrate performance improvement with data scale
- [ ] **Trace analysis**: Compare trajectories before and after training to measure behavioral changes (`s3://skyrl-trajectories/evals/`)
- [ ] **Kimi K2.5 training**: Train on Kimi K2.5 model
- [ ] **OpenEnv browseComp integration**: Integrate with OpenEnv browseComp for browser-use tasks
- [ ] **Browser use (computer_use modality)**: Train on browser-based tasks with visual grounding
- [ ] **Combined training**: Joint training on `tool_use` + `computer_use` for generalist agents

## 5. References

- SkyRL repo: https://github.com/fleet-ai/SkyRL
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- Changelog: [changelog.md](changelog.md)
- Training config: `skyrl-train/tasks/openenv-fleet-grpo-qwen3-8b.yaml`
- Trajectories: `s3://skyrl-trajectories/evals/`
