# SkyRL Fleet Training

## Abstract

Training an 8B parameter model (Qwen3-8B) on Fleet tool-use tasks using GRPO. We iteratively identify failure modes, fix them through prompt engineering and hyperparameter tuning, and measure improvement on held-out evaluation tasks. Starting from 2.9% success rate, we achieve 50.9% pass@1 after three iterations.

## 1. Introduction

Tool-use agents require multi-step reasoning: understanding the task, selecting appropriate tools, executing them in sequence, and handling errors. We train on Fleet environments (booking, github, outlook, etc.) which provide realistic API interactions with automatic verification.

**Goal:** Train a generalist tool-use agent that generalizes across environments.

## 2. Dataset

### 2.1 Task Collection

Tasks are collected from Fleet environments with automatic verification:
1. **Tool call replay**: Each task's tool calls are replayed against the Fleet environment to ensure they execute successfully
2. **Teacher model verification**: Tasks are validated by running a teacher model (Claude) to confirm solvability
3. **Filtering**: Tasks that fail replay or teacher verification are excluded

### 2.2 Versions

| Version | Tasks | env_variables | Notes |
|---------|-------|---------------|-------|
| v0.1 | 3,603 | 0% | Missing runtime context (dates, user info) |
| v0.2 | 3,485 | 95% | Added `CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME` |
| v0.3 | 10,175 | 95%+ | 3x scale: added google-maps, fostgres, amazon, carlisle, wallst |

### 2.3 Train/Eval Split

**v0.3** (10% eval ratio, capped at ~30 samples per environment)

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| amazon | 16 | 2 | 18 |
| booking | 1,313 | 23 | 1,336 |
| carlisle | 36 | 2 | 38 |
| dropbox | 3 | 0 | 3 |
| fira | 60 | 6 | 66 |
| fostgres | 923 | 26 | 949 |
| github | 2,048 | 33 | 2,081 |
| google-maps | 4,695 | 32 | 4,727 |
| hubspot | 16 | 3 | 19 |
| outlook | 36 | 3 | 39 |
| reddit | 432 | 23 | 455 |
| ticketmaster | 229 | 28 | 257 |
| wallst | 65 | 4 | 69 |
| zillow | 109 | 9 | 118 |
| **TOTAL** | **9,981** | **194** | **10,175** |

<details>
<summary>v0.2 split (previous)</summary>

| Environment | Train | Eval | Total |
|-------------|-------|------|-------|
| booking | 1,059 | 30 | 1,089 |
| fira | 46 | 5 | 51 |
| github | 1,835 | 30 | 1,865 |
| hubspot | 9 | 1 | 10 |
| outlook | 17 | 2 | 19 |
| reddit | 169 | 19 | 188 |
| ticketmaster | 200 | 22 | 222 |
| zillow | 30 | 3 | 33 |
| **TOTAL** | **3,365** | **112** | **3,477** |

</details>

## 3. Method

**Algorithm:** GRPO (Group Relative Policy Optimization)
- Samples multiple rollouts per prompt
- Computes advantages relative to group mean
- Updates policy to increase probability of high-reward trajectories

**Training Setup:**
| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B |
| Context | 65K (YaRN rope_scaling) |
| max_input_length | 48K |
| max_generate_length | 8K |
| n_samples_per_prompt | 4 (train), 1 (eval) |

**Reward:** Binary (1.0 if task completed correctly, 0.0 otherwise)

## 4. Experiments (Changelog)

---

### v0.2.4 - Run `larger data`
<!-- ### v0.2.4

**Hypothesis:** Learning rate too conservative + need more data.

**Changes:**
| Change | v0.2.3 | v0.2.4 | Rationale |
|--------|--------|--------|-----------|
| Dataset | v0.2 (~3.4K tasks) | v0.3 (larger) | Scale up training data |
| `learning_rate` | 1e-6 | 5e-6 | 5x increase - no PPO clip hit suggests updates too small |
| `kl_coef` | 0.01 | 0.01 | Keep (isolate LR effect first) |

**Metrics to watch:**
- `reward/variance_per_prompt` - Should remain > 0 (learning signal exists)
- PPO clip ratio hit rate - If now clipping, LR is effective
- Per-env pass@n - Especially flat envs (booking, fira, reddit, zillow) -->

---

### v0.2.3 - Run `fleet_tool_use_a7e85045`

**Changes:**
- Dataset split: eval_ratio 2% → 10%, MAX_EVAL_SAMPLES=30
- outlook no longer held-out (split like other envs)

**Results (Step 80):**

| Environment | Step 0 | Step 80 | Delta | Status |
|-------------|--------|---------|-------|--------|
| booking | 53.8% | 57.7% | +3.8% | FLAT |
| fira | 40.0% | 40.0% | +0.0% | FLAT |
| github | 46.7% | **70.0%** | **+23.3%** | IMPROVED |
| hubspot | 100.0% | 100.0% | +0.0% | SATURATED |
| outlook | 100.0% | 100.0% | +0.0% | SATURATED |
| reddit | 66.7% | 66.7% | +0.0% | FLAT |
| ticketmaster | 3.6% | 7.1% | +3.6% | FAILING |
| zillow | 50.0% | 50.0% | +0.0% | FLAT |
| **OVERALL** | **43.1%** | **50.9%** | **+7.8%** | |

**Trajectories:** `s3://skyrl-trajectories/evals/fleet_tool_use_a7e85045/`

**Issues identified:**
- **ticketmaster**: Only 1 unique eval task - complex 5+ step e-commerce workflow
- **hubspot/outlook**: Saturated at 100% from step 0 - no learning signal
- **Most envs FLAT**: No PPO clipping observed, suggesting LR too low

---

### v0.2.2 - Run `mk6nr5ij`

**Changes:**
- First training run with v0.2 dataset (env_variables fix)
- YaRN rope_scaling enabled (65K context)
- System prompt improvements for task completion

**Results:**

| Environment | Baseline | pass@3 | Delta |
|-------------|----------|--------|-------|
| github | 0% | 42.1% | **+42.1%** |
| booking | ~5% | 52.6% | **+47.6%** |
| outlook | ~5% | 36.8% | **+31.8%** |
| **OVERALL** | **2.9%** | **43.4%** | **+40.5%** |

**Issues fixed (from baseline):**

| Issue | Baseline | After | Status |
|-------|----------|-------|--------|
| GitHub no env response | 88% no response, 1.1 tool calls | 14.3 avg tool calls | FIXED |
| Excessive thinking | 81% token budget on thinking | 6K tokens (success) vs 11K (failure) | IMPROVED |
| Cannot signal done | 40% booking stuck | max_turns=6, tasks completing | FIXED |
| Hitting max turns | 90% hit 50 turns | booking max=6, reddit max=5 | IMPROVED |
| Tool repetition | 56% booking, 20x repeats | - | NEEDS ANALYSIS |

---

### v0.2.1

**Changes:** Infrastructure fixes for rope_scaling compatibility.

| Fix | Issue | Change |
|-----|-------|--------|
| vLLM rope_scaling | vLLM 0.13.0 removed direct `rope_scaling` param | Pass via `hf_overrides["rope_parameters"]` |
| HuggingFace rope_scaling | HF models reject `rope_scaling` as kwarg | Set `model_config.rope_scaling` instead |

**PR:** [NovaSky-AI/SkyRL#976](https://github.com/NovaSky-AI/SkyRL/pull/976)

---

### v0.2

**Changes:** Dataset and prompt fixes based on baseline analysis.

| Fix | Issue Addressed | Change |
|-----|-----------------|--------|
| Context extension | Excessive thinking | YaRN rope_scaling (65K), MAX_INPUT_LENGTH 24K→48K, MAX_GENERATE_LENGTH 4K→8K |
| Done signal prompt | Cannot signal completion | System prompt requires `<tool_call>` OR `<done>` every turn |
| Error handling prompt | Tool repetition | System prompt: "Do NOT repeat the same call with identical arguments" |
| env_variables fix | Date/user errors | Training data includes `CURRENT_DATE`, `LOGGED_IN_USER`, `LOGGED_IN_NAME` |

**Commits:**
- `de6b4b4` - Enable YaRN for 65K context, increase generation length
- `6a9b813` - Strengthen system prompt for task completion signaling

---

### Baseline Analysis

**Run:** `fleet_tool_use_92d02656`
**Overall success rate:** 2.9% (7/243 trajectories)

| Issue | Severity | Description | Impact |
|-------|----------|-------------|--------|
| **0. No env response** | CRITICAL | 88% GitHub tool calls get no response | 0% GitHub success |
| **1. Excessive thinking** | CRITICAL | 81% token budget on `<think>` blocks | Only 1.1 tool calls on GitHub |
| **2. Cannot signal done** | CRITICAL | Env rejects non-tool-call responses | Infinite loops |
| **3. Tool repetition** | HIGH | Same call repeated up to 20x | Wasted turns |
| **4. Max turns hit** | HIGH | 90% hit 50 turn limit | Tasks never complete |
| **5. Thought repetition** | MEDIUM | Same reasoning repeated 14x | Token waste |
| **6. Hallucinated params** | MEDIUM | Made-up IDs in tool calls | Failed API calls |
| **7. Premature done** | LOW | 18% Outlook say done with 0 score | False completion |

---

## 5. Analysis

**What works:**
- github shows strong improvement (+23.3% in v0.2.3, +42.1% in v0.2.2)
- Context extension and done signaling were critical fixes
- Training does improve performance when learning signal exists

**What doesn't:**
- Most environments flat despite training (booking, fira, reddit, zillow)
- No PPO clipping suggests learning rate too low
- Saturated environments (hubspot, outlook) provide no learning signal
- ticketmaster has insufficient eval data (1 unique task)

**Key metric:** `variance_per_prompt` - if 0, no GRPO learning signal exists.

## 6. Future Work

- [ ] **v0.2.4**: Increase LR to enable actual policy updates
- [ ] **v0.3 dataset**: Scale up training data
- [ ] **Per-env curriculum**: Weight environments by learning potential
- [ ] **Browser-use**: Extend to `computer_use` modality
- [ ] **Combined training**: Joint `tool_use` + `computer_use`

## References

- SkyRL repo: https://github.com/fleet-ai/SkyRL
- OpenEnv repo: https://github.com/fleet-ai/OpenEnv
- Training config: `skyrl-train/tasks/openenv-fleet-grpo-qwen3-8b.yaml`
- Trajectories: `s3://skyrl-trajectories/evals/`
