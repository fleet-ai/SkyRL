# Fleet Integration Architecture

This document describes how Fleet environments integrate with SkyRL and Tinker training loops.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Training Loops                                  │
├───────────────────────────────────┬─────────────────────────────────────────┤
│      SkyRL Training Loop          │         Tinker Training Loop            │
│   (skyrl_gym_generator.py)        │      (main_fleet_tinker.py)             │
│                                   │                                         │
│  ┌─────────────────────────┐      │      ┌─────────────────────────┐        │
│  │ generate_batched()      │      │      │ main()                  │        │
│  │                         │      │      │                         │        │
│  │ - env.init(prompt)      │      │      │ - collect_batch_rollouts│        │
│  │ - env.step(action)      │      │      │ - prepare_training_data │        │
│  │ - env.close()           │      │      │ - forward_backward      │        │
│  │                         │      │      │                         │        │
│  │ Uses: SYNC methods      │      │      │ Uses: ASYNC methods     │        │
│  └───────────┬─────────────┘      │      └───────────┬─────────────┘        │
│              │                    │                  │                      │
└──────────────┼────────────────────┼──────────────────┼──────────────────────┘
               │                    │                  │
               │   asyncio.run()    │                  │  await
               ▼                    │                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SkyRL FleetTaskEnv Wrapper                               │
│                  (integrations/fleet/env.py)                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FleetTaskEnv                                 │   │
│  │                      (extends BaseTextEnv)                           │   │
│  │                                                                      │   │
│  │  Sync Methods (wrappers)      │    Async Methods (actual logic)     │   │
│  │  ─────────────────────────    │    ────────────────────────────     │   │
│  │  init(prompt)                 │    init_async(prompt)               │   │
│  │    └─ asyncio.run(init_async) │      └─ await openenv.reset_async() │   │
│  │                               │      └─ build system prompt         │   │
│  │  step(action)                 │    step_async(action)               │   │
│  │    └─ asyncio.run(step_async) │      └─ parse tool calls            │   │
│  │                               │      └─ await openenv.step_async()  │   │
│  │  close()                      │      └─ build observation           │   │
│  │    └─ openenv.close()         │                                     │   │
│  │                               │                                     │   │
│  │  Manages: chat_history, tools, turns, tool_calls                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ await
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenEnv FleetTaskEnv                                │
│                    (envs/fleet_env/task_env.py)                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  reset_async()     → Creates Fleet env, fetches tools               │   │
│  │  step_async(action)→ Executes tool call, returns (obs, reward, done)│   │
│  │  close()           → Cleanup Fleet environment                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/MCP
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Fleet Platform                                    │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Outlook    │  │    GitHub    │  │   Booking    │  │    Reddit    │    │
│  │ Environment  │  │ Environment  │  │ Environment  │  │ Environment  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Abstractions

| Layer | Component | Purpose |
|-------|-----------|---------|
| Training | `main_fleet_tinker.py` | Tinker-based training with async rollout collection |
| Training | `skyrl_gym_generator.py` | SkyRL's standard training with sync interface |
| Wrapper | `FleetTaskEnv` (SkyRL) | Unified sync/async interface, chat history management |
| Env | `FleetTaskEnv` (OpenEnv) | Low-level Fleet environment interaction |
| Platform | Fleet | Hosted task environments (Outlook, GitHub, etc.) |

## Communication Flow

### Tinker Training Loop (Async)
```python
# 1. Create environment
env = FleetTaskEnv(env_config, extras={"task_key": task_key, ...})

# 2. Initialize (async)
chat_history, metadata = await env.init_async([])

# 3. Rollout loop
while not done:
    input_ids = tokenize(env.chat_history)
    output = tinker_sample(input_ids)          # Tinker inference
    step_output = await env.step_async(output)  # Fleet environment
    done = step_output.done

# 4. Cleanup
env.close()
```

### SkyRL Training Loop (Sync)
```python
# 1. Create environment
env = skyrl_gym.make("fleet_task", env_config, extras)

# 2. Initialize (sync - uses asyncio.run internally)
chat_history, metadata = env.init(prompt)

# 3. Rollout loop
while not done:
    output = inference_engine.generate(chat_history)
    step_output = env.step(output)  # Uses asyncio.run internally
    done = step_output.done

# 4. Cleanup
env.close()
```

## Data Flow

```
┌────────────┐    prompt     ┌────────────┐    action     ┌────────────┐
│  Training  │ ───────────▶  │  FleetTask │ ───────────▶  │   Fleet    │
│    Loop    │               │    Env     │               │  Platform  │
│            │ ◀───────────  │  (SkyRL)   │ ◀───────────  │            │
└────────────┘  chat_history └────────────┘  observation  └────────────┘
                step_output
```

### Key Data Structures

**init_async() returns:**
- `chat_history`: List of messages `[{role, content}, ...]`
- `metadata`: `{task_key, env_key, tools, modality}`

**step_async() returns:**
- `BaseTextEnvStepOutput`:
  - `observations`: New messages to append
  - `reward`: Float (0.0 during rollout, final reward on done)
  - `done`: Boolean
  - `metadata`: `{task_key, turn, tool_call, tool_result, error}`
