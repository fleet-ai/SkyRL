"""
Fleet Task Environment for SkyRL.

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
It uses the Fleet SDK directly to create environments and run verifiers.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

# Import SkyRL base classes
try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
except ImportError as e:
    raise ImportError("skyrl_gym is required. Make sure you're running within the SkyRL environment.") from e

# Import Fleet SDK
try:
    from fleet import Fleet
except ImportError as e:
    raise ImportError("Fleet SDK is required. Install with: pip install fleet-python") from e


# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, List[Any]] = {}
_FLEET_CLIENT: Optional[Fleet] = None


def get_fleet_client(api_key: Optional[str] = None) -> Fleet:
    """Get or create Fleet client singleton."""
    global _FLEET_CLIENT
    if _FLEET_CLIENT is None:
        key = api_key or os.environ.get("FLEET_API_KEY")
        if not key:
            raise ValueError("FLEET_API_KEY environment variable not set")
        _FLEET_CLIENT = Fleet(api_key=key)
    return _FLEET_CLIENT


def load_tasks_from_json(tasks_file: str) -> Dict[str, Any]:
    """Load tasks from JSON file with caching.

    Returns a dict mapping task_key -> task_config dict.
    """
    if tasks_file not in _TASK_CACHE:
        expanded_path = os.path.expanduser(tasks_file)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Tasks file not found: {expanded_path}")

        with open(expanded_path, "r") as f:
            data = json.load(f)

        # Handle both formats: array or {"tasks": [...]}
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Invalid JSON format in {tasks_file}: expected array or object with 'tasks' key")

        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")

        # Index by task_key
        _TASK_CACHE[tasks_file] = {t.get("key") or t.get("task_key"): t for t in tasks}

    return _TASK_CACHE[tasks_file]


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool call from LLM response.

    Supports tag-based formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <function_call>{"name": "...", "arguments": {...}}</function_call>

    Returns dict with "name" and "arguments" keys, or None if not found.
    """
    # Try common tag formats
    for tag in ["tool_call", "function_call"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                # Normalize keys
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params", {})
                if name:
                    return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                pass

    return None


class FleetTaskEnv(BaseTextEnv):
    """
    SkyRL environment for Fleet-hosted tasks.

    Uses Fleet SDK directly to create environments and run verifiers.
    No dependency on OpenEnv.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)

        # Task configuration from extras (set by dataset)
        self.task_key = extras.get("task_key")
        self.tasks_file = env_config.get("tasks_file") or extras.get("tasks_file")

        if not self.task_key:
            raise ValueError("task_key must be provided in extras (from dataset)")
        if not self.tasks_file:
            raise ValueError("tasks_file must be provided in env_config or extras")

        # Expand path
        self.tasks_file = os.path.expanduser(self.tasks_file)

        # Load task config from JSON
        tasks = load_tasks_from_json(self.tasks_file)
        self.task_config = tasks.get(self.task_key)
        if not self.task_config:
            available_keys = list(tasks.keys())[:5]
            raise ValueError(
                f"Task '{self.task_key}' not found in {self.tasks_file}. " f"Available keys (first 5): {available_keys}"
            )

        # API key
        self.api_key = env_config.get("api_key") or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError("FLEET_API_KEY must be set in env_config or environment")

        # Environment state (initialized on init())
        self.fleet_env = None
        self.fleet_task = None
        self.chat_history: ConversationType = []
        self.turns = 0
        self.tools: List[Dict[str, Any]] = []

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation.

        Creates Fleet environment via task.make() and returns the task prompt.
        """
        # Get Fleet client and load task
        fleet = get_fleet_client(self.api_key)

        # Load task from Fleet API using task key
        try:
            tasks = fleet.load_tasks()
            self.fleet_task = None
            for task in tasks:
                if task.key == self.task_key:
                    self.fleet_task = task
                    break

            if self.fleet_task is None:
                raise ValueError(f"Task '{self.task_key}' not found in Fleet API")

        except Exception as e:
            raise RuntimeError(f"Failed to load task from Fleet API: {e}") from e

        # Create Fleet environment
        try:
            self.fleet_env = self.fleet_task.make()
        except Exception as e:
            raise RuntimeError(f"Failed to create Fleet environment: {e}") from e

        # Reset state
        self.turns = 0

        # Get tools from Fleet environment
        try:
            self.tools = self.fleet_env.mcp.list_tools() if hasattr(self.fleet_env, "mcp") else []
        except Exception:
            self.tools = []

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Add tool information if available
        if self.tools:
            tool_names = [t.get("name", "unknown") for t in self.tools]
            tools_info = f"\n\nAvailable tools: {', '.join(tool_names)}\n"
            tools_info += (
                'To use a tool, respond with: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>\n'
            )
            tools_info += "When you have completed the task, include <done> in your response."
        else:
            tools_info = ""

        # Build conversation
        initial_message = {"role": "user", "content": task_prompt + tools_info}
        self.chat_history = [initial_message]

        metadata = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment.

        Parses the action for tool calls, executes via Fleet SDK,
        and returns observation. Reward is computed by the verifier on completion.
        """
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        max_turns_reached = self.turns >= self.max_turns

        # Check if agent signals completion
        agent_done = "<done>" in action.lower() or "[done]" in action.lower()

        # Parse tool call
        tool_call = parse_tool_call(action)

        tool_result = None
        error = None
        reward = 0.0

        # Execute tool call if present
        if tool_call and self.fleet_env and hasattr(self.fleet_env, "mcp"):
            try:
                result = self.fleet_env.mcp.call_tool(tool_call["name"], tool_call.get("arguments", {}))
                tool_result = result
            except Exception as e:
                error = str(e)

        # Check if episode is done
        episode_done = agent_done or max_turns_reached

        # Run verifier if done
        if episode_done and self.fleet_task and self.fleet_env:
            try:
                reward = self.fleet_task.verify(self.fleet_env)
                if reward is None:
                    reward = 0.0
            except Exception as e:
                print(f"Warning: Verifier failed: {e}")
                reward = 0.0

        # Build observation message
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"done_reason": "max_turns", "task_key": self.task_key},
            )

        # Build response observation
        if error:
            obs_content = f"Error: {error}"
        elif tool_result:
            if isinstance(tool_result, dict):
                obs_content = f"Tool result:\n{json.dumps(tool_result, indent=2)}"
            else:
                obs_content = f"Tool result:\n{tool_result}"
        elif agent_done:
            obs_content = "Task marked as complete."
        elif not tool_call:
            obs_content = 'No tool call found. Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> format.'
        else:
            obs_content = "Action executed."

        new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)

        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
        }

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.fleet_env:
            try:
                if hasattr(self.fleet_env, "close"):
                    self.fleet_env.close()
            except Exception as e:
                print(f"Warning: Failed to close Fleet environment: {e}")
            self.fleet_env = None
        self.fleet_task = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics for this episode."""
        return {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "turns": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        total_turns = sum(m.get("turns", 0) for m in metrics)

        # Group by env_key
        env_counts: Dict[str, int] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        return {
            "avg_turns": total_turns / len(metrics),
            "total_episodes": len(metrics),
            "env_distribution": env_counts,
        }


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE, _FLEET_CLIENT
    _TASK_CACHE = {}
    _FLEET_CLIENT = None
