"""
Fleet Task Training with Tinker Backend.

This entrypoint uses Tinker (hosted) for training and inference,
combined with Fleet environments via OpenEnv for rollout collection.

Usage:
    python -m integrations.fleet.entrypoints.main_fleet_tinker \
        --model-name Qwen/Qwen2.5-1.5B-Instruct \
        --tasks-file /path/to/tasks.json \
        --dataset-file /path/to/train.parquet \
        --eval-dataset-file /path/to/validation.parquet

Environment Variables:
    TINKER_API_URL: Tinker service URL (optional, uses default if not set)
    FLEET_API_KEY: Fleet API key for environment access
    WANDB_API_KEY: Weights & Biases API key for logging
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import tinker
import torch
import wandb
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from integrations.fleet.env import FleetTaskEnv, load_tasks_from_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have mean 0 and std 1."""
    if not advantages or len(advantages) == 1:
        return advantages
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return [0.0] * len(advantages)
    return [(a - mean) / (std + 1e-8) for a in advantages]


def compute_advantages_grpo(
    rewards: List[float],
    group_size: int = None,
    normalize: bool = True,
) -> List[float]:
    """
    GRPO (Group Relative Policy Optimization) advantage estimation.

    For each group of trajectories from the same prompt, compute advantages
    as deviations from the group mean.
    """
    rewards = np.array(rewards)

    if group_size is None:
        group_size = len(rewards)

    n_groups = len(rewards) // group_size
    advantages = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        advantages.extend(group_advantages.tolist())

    remaining = len(rewards) % group_size
    if remaining > 0:
        # Handle remaining samples as their own group
        remaining_rewards = rewards[-remaining:]
        remaining_mean = remaining_rewards.mean()
        advantages.extend((remaining_rewards - remaining_mean).tolist())

    if normalize:
        advantages = normalize_advantages(advantages)

    return advantages


async def collect_fleet_rollout(
    task_config: Dict[str, Any],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Collect a single trajectory using Fleet environment and Tinker inference.

    Args:
        task_config: Task configuration from dataset
        tasks_file: Path to tasks JSON file
        sampling_client: Tinker sampling client for generation
        tokenizer: Tokenizer for encoding/decoding
        max_turns: Maximum turns per episode
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature

    Returns:
        Dictionary with trajectory data (prompt_ids, response_ids, reward, logprobs, etc.)
    """
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY environment variable must be set")

    task_key = task_config.get("task_key") or task_config.get("key")
    env_config = {"tasks_file": tasks_file, "api_key": api_key}

    # Create Fleet environment
    env = FleetTaskEnv(
        env_config=env_config,
        extras={"task_key": task_key, "max_turns": max_turns},
    )

    try:
        # Initialize environment
        prompt_messages = task_config.get("prompt", [])
        if isinstance(prompt_messages, str):
            prompt_messages = [{"role": "user", "content": prompt_messages}]

        chat_history, metadata = env.init(prompt_messages)

        # Tokenize initial prompt
        prompt_ids = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=True)

        all_response_ids = []
        all_logprobs = []
        loss_mask = []
        done = False
        total_reward = 0.0

        while not done:
            # Prepare input for Tinker
            input_ids = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=True)

            # Generate with Tinker
            sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )

            result = sampling_client.sample(
                [types.ModelInput.from_ints(tokens=input_ids)],
                sampling_params=sampling_params,
            ).result()

            if not result.sequences or len(result.sequences) == 0:
                logger.warning("No sequences returned from Tinker")
                break

            sequence = result.sequences[0]
            output_ids = sequence.token_ids
            output_logprobs = sequence.logprobs if hasattr(sequence, "logprobs") else []

            # Decode output
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            # Step environment
            step_output = env.step(output_text)

            # Collect trajectory data
            all_response_ids.extend(output_ids)
            if output_logprobs:
                all_logprobs.extend(output_logprobs)
            else:
                all_logprobs.extend([0.0] * len(output_ids))

            # Loss mask: 1 for response tokens
            loss_mask.extend([1] * len(output_ids))

            # Update chat history
            chat_history.append({"role": "assistant", "content": output_text})
            if step_output.observations:
                for obs in step_output.observations:
                    chat_history.append(obs)
                    # Add observation tokens to response (masked out for loss)
                    obs_ids = tokenizer.encode(obs["content"], add_special_tokens=False)
                    all_response_ids.extend(obs_ids)
                    all_logprobs.extend([0.0] * len(obs_ids))
                    loss_mask.extend([0] * len(obs_ids))  # Don't train on observations

            done = step_output.done
            total_reward = step_output.reward

        return {
            "prompt_ids": prompt_ids,
            "response_ids": all_response_ids,
            "logprobs": all_logprobs,
            "loss_mask": loss_mask,
            "reward": total_reward,
            "task_key": task_key,
            "env_key": metadata.get("env_key"),
            "turns": env.turns,
            "tool_calls": env.tool_calls,
        }

    finally:
        env.close()


async def collect_batch_rollouts(
    batch: List[Dict[str, Any]],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    n_samples_per_prompt: int = 1,
) -> List[Dict[str, Any]]:
    """Collect rollouts for a batch of tasks."""
    rollouts = []

    for task_config in batch:
        for _ in range(n_samples_per_prompt):
            try:
                rollout = await collect_fleet_rollout(
                    task_config=task_config,
                    tasks_file=tasks_file,
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    max_turns=max_turns,
                )
                rollouts.append(rollout)
            except Exception as e:
                logger.error(f"Failed to collect rollout for {task_config.get('task_key')}: {e}")
                # Add dummy failed rollout
                rollouts.append(
                    {
                        "prompt_ids": [],
                        "response_ids": [],
                        "logprobs": [],
                        "loss_mask": [],
                        "reward": 0.0,
                        "task_key": task_config.get("task_key"),
                        "env_key": task_config.get("env_key"),
                        "turns": 0,
                        "tool_calls": 0,
                        "error": str(e),
                    }
                )

    return rollouts


def collate_fn(batch):
    """Return batch as-is without tensor collation."""
    return batch


async def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    save_path: str,
    step: int,
) -> Dict[str, str]:
    """Save training checkpoint."""
    state_result = await training_client.save_state_async(name)
    state_path = (await state_result.result_async()).path

    sampler_result = await training_client.save_weights_for_sampler_async(name)
    sampler_path = (await sampler_result.result_async()).path

    checkpoint_info = {
        "name": name,
        "step": step,
        "state_path": state_path,
        "sampler_path": sampler_path,
    }

    checkpoint_file = os.path.join(save_path, "checkpoints.jsonl")
    with open(checkpoint_file, "a") as f:
        f.write(json.dumps(checkpoint_info) + "\n")

    logger.info(f"Saved checkpoint {name}: state={state_path}, sampler={sampler_path}")
    return checkpoint_info


async def main(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    tasks_file: str = None,
    dataset_file: str = None,
    eval_dataset_file: str = None,
    batch_size: int = 8,
    eval_batch_size: int = 32,
    learning_rate: float = 4e-5,
    lora_rank: int = 16,
    max_steps: int = 200,
    max_turns: int = 50,
    n_samples_per_prompt: int = 4,
    save_every: int = 10,
    eval_every: int = 20,
    seed: int = 42,
    wandb_project: str = "fleet-tinker-grpo",
    wandb_name: str = None,
    resume_from: str = None,
):
    """
    Main training loop using Tinker for training/inference and Fleet for environments.
    """
    set_seed(seed)

    # Setup paths
    if wandb_name is None:
        wandb_name = f"{model_name.split('/')[-1]}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = os.path.join("./tinker_fleet_output", wandb_name)
    os.makedirs(save_path, exist_ok=True)

    # Check for resume
    resume_from_step = 0
    load_state_path = None
    checkpoint_file = os.path.join(save_path, "checkpoints.jsonl")
    if resume_from or os.path.exists(checkpoint_file):
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoints = [json.loads(line) for line in f]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x["step"])
                resume_from_step = latest["step"]
                load_state_path = latest["state_path"]
                logger.info(f"Resuming from step {resume_from_step}")

    # Initialize WandB
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_turns": max_turns,
            "n_samples_per_prompt": n_samples_per_prompt,
        },
    )

    # Load datasets
    train_dataset = load_dataset("parquet", data_files=dataset_file)["train"]
    eval_dataset = load_dataset("parquet", data_files=eval_dataset_file)["train"] if eval_dataset_file else None

    logger.info(f"Loaded {len(train_dataset)} training samples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} eval samples")

    # Setup Tinker
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=lora_rank)

    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataloader
    def create_dataloader(epoch: int):
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + epoch),
        )

    steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
    current_epoch = resume_from_step // steps_per_epoch
    train_dataloader = create_dataloader(current_epoch)
    train_iterator = iter(train_dataloader)

    # Skip to resume position
    batch_offset = resume_from_step % steps_per_epoch
    for _ in range(batch_offset):
        try:
            next(train_iterator)
        except StopIteration:
            break

    # Training loop
    for step in range(resume_from_step, max_steps):
        step_start = time.time()
        metrics = {"step": step, "epoch": step // steps_per_epoch}

        # Save checkpoint
        if save_every > 0 and step > 0 and step % save_every == 0:
            await save_checkpoint(training_client, f"step_{step:06d}", save_path, step)

        # Get sampling weights
        sampling_path = training_client.save_weights_for_sampler(name=f"step_{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Get batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            current_epoch += 1
            train_dataloader = create_dataloader(current_epoch)
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Collect rollouts
        logger.info(f"Step {step}: Collecting rollouts for {len(batch)} tasks...")
        rollout_start = time.time()

        rollouts = await collect_batch_rollouts(
            batch=batch,
            tasks_file=tasks_file,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            max_turns=max_turns,
            n_samples_per_prompt=n_samples_per_prompt,
        )

        metrics["time/rollout"] = time.time() - rollout_start

        # Filter valid rollouts
        valid_rollouts = [r for r in rollouts if r.get("response_ids") and not r.get("error")]
        if not valid_rollouts:
            logger.warning(f"Step {step}: No valid rollouts, skipping")
            continue

        # Compute GRPO advantages
        rewards = [r["reward"] for r in valid_rollouts]
        advantages = compute_advantages_grpo(rewards, group_size=n_samples_per_prompt, normalize=True)

        metrics["reward/mean"] = np.mean(rewards)
        metrics["reward/max"] = np.max(rewards)
        metrics["reward/min"] = np.min(rewards)
        metrics["rollouts/valid"] = len(valid_rollouts)
        metrics["rollouts/total"] = len(rollouts)

        # Prepare training data
        training_datums = []
        for idx, rollout in enumerate(valid_rollouts):
            prompt_ids = rollout["prompt_ids"]
            response_ids = rollout["response_ids"]
            logprobs = rollout["logprobs"]
            loss_mask = rollout["loss_mask"]

            full_sequence = prompt_ids + response_ids
            prompt_len = len(prompt_ids)

            # Target tokens (shifted by 1)
            target_tokens = full_sequence[1:]

            # Logprobs (0 for prompt, actual for response)
            full_logprobs = [0.0] * prompt_len + logprobs
            full_logprobs = full_logprobs[1:]

            # Loss mask (0 for prompt, actual for response)
            full_mask = [0] * prompt_len + loss_mask
            full_mask = full_mask[1:]

            # Advantages
            advantage_value = advantages[idx]
            full_advantages = torch.zeros(len(full_sequence))
            for i in range(prompt_len, len(full_sequence)):
                if i - 1 < len(full_mask) and full_mask[i - 1] > 0:
                    full_advantages[i] = advantage_value
            full_advantages = full_advantages[1:]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(full_logprobs)),
                    "advantages": TensorData.from_torch(full_advantages),
                },
            )
            training_datums.append(datum)

        # Training step
        logger.info(f"Step {step}: Training on {len(training_datums)} sequences...")
        train_start = time.time()

        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn="ppo")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_future.result()
        optim_step_future.result()

        metrics["time/train"] = time.time() - train_start
        metrics["time/total"] = time.time() - step_start

        # Log metrics
        wandb.log(metrics, step=step)
        logger.info(f"Step {step}: reward={metrics['reward/mean']:.3f}, time={metrics['time/total']:.1f}s")

        # Evaluation
        if eval_every > 0 and eval_dataset and step % eval_every == 0:
            logger.info(f"Step {step}: Running evaluation...")
            eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)

            eval_rewards = []
            for eval_batch in eval_dataloader:
                eval_rollouts = await collect_batch_rollouts(
                    batch=eval_batch,
                    tasks_file=tasks_file,
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    max_turns=max_turns,
                    n_samples_per_prompt=1,
                )
                for r in eval_rollouts:
                    if not r.get("error"):
                        eval_rewards.append(r["reward"])

            if eval_rewards:
                wandb.log(
                    {
                        "eval/reward_mean": np.mean(eval_rewards),
                        "eval/reward_std": np.std(eval_rewards),
                        "eval/num_samples": len(eval_rewards),
                    },
                    step=step,
                )

    # Save final checkpoint
    await save_checkpoint(training_client, "final", save_path, max_steps)

    wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Task Training with Tinker")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tasks-file", type=str, required=True, help="Path to tasks JSON file")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--eval-dataset-file", type=str, default=None, help="Path to eval parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="fleet-tinker-grpo")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    args = parser.parse_args()

    asyncio.run(
        main(
            model_name=args.model_name,
            tasks_file=args.tasks_file,
            dataset_file=args.dataset_file,
            eval_dataset_file=args.eval_dataset_file,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_steps=args.max_steps,
            max_turns=args.max_turns,
            n_samples_per_prompt=args.n_samples_per_prompt,
            save_every=args.save_every,
            eval_every=args.eval_every,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            resume_from=args.resume_from,
        )
    )
