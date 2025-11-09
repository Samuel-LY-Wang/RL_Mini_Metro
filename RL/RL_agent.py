'''
Create an RL agent for the Mini Metro env
'''

import gymnasium as gym
from gymnasium import spaces
from RL_env import MiniMetroEnv
import numpy as np

class RLAgent:
    def __init__(self, env: MiniMetroEnv):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def select_action(self, state):
        # Example: Random action selection
        return self.action_space.sample()
    def train(self, num_episodes: int, plot: bool = False, plot_path: str = 'random_baseline_rewards.png', csv_path: str = 'random_baseline_rewards.csv', verbose: bool = False):
        """Run a simple training loop using this agent's policy (default: random).

        Records per-episode returns and optionally saves a plot and CSV.
        """
        episode_rewards = []
        for episode in range(num_episodes):
            res = self.env.reset()
            # support old gym (obs) and new gym/gymnasium (obs, info)
            if isinstance(res, tuple) and len(res) == 2:
                state, info = res
            else:
                state = res
                info = {}
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action = self.select_action(state)
                step_res = self.env.step(action)
                # handle both modern (5-tuple) and classic (4-tuple)
                if isinstance(step_res, tuple) and len(step_res) == 5:
                    next_state, reward, terminated, truncated, info = step_res
                elif isinstance(step_res, tuple) and len(step_res) == 4:
                    next_state, reward, done, info = step_res
                    terminated = bool(done)
                    truncated = False
                else:
                    raise RuntimeError("env.step returned unexpected result shape")
                episode_reward += float(reward)
                state = next_state
            episode_rewards.append(float(episode_reward))
            if verbose:
                print(f"Episode {episode} finished, total reward: {episode_reward}")

        # optionally save plot and CSV
        if plot:
            try:
                # use Agg backend so savefig works without a display
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o')
                plt.xlabel('Episode')
                plt.ylabel('Episode return')
                plt.title('Random baseline: episode returns')
                plt.grid(True)
                plt.tight_layout()
                try:
                    import os
                    abs_path = os.path.abspath(plot_path)
                    if verbose:
                        print(f"Attempting to save plot to: {abs_path}")
                        print(f"Current working dir: {os.getcwd()}")
                    plt.savefig(plot_path)
                    # confirm file exists
                    exists = os.path.exists(abs_path)
                    if verbose:
                        print(f"savefig reported no exception; file exists: {exists} -> {abs_path}")
                    if not exists and verbose:
                        print("Warning: save completed but file not found at expected path")
                except Exception as e:
                    if verbose:
                        print(f"Failed to save random baseline plot to {plot_path}: {e}")
            except Exception:
                if verbose:
                    print("matplotlib not available; install matplotlib to enable plotting: pip install matplotlib")
            try:
                import csv
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'return'])
                    for i, r in enumerate(episode_rewards, start=1):
                        writer.writerow([i, r])
                if verbose:
                    print(f"Saved random baseline CSV to {csv_path}")
            except Exception:
                pass

    def evaluate(self, num_episodes):
        total_reward = 0.0
        for episode in range(num_episodes):
            res = self.env.reset()
            if isinstance(res, tuple) and len(res) == 2:
                state, info = res
            else:
                state = res
                info = {}
            terminated = False
            truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action = self.select_action(state)
                step_res = self.env.step(action)
                if isinstance(step_res, tuple) and len(step_res) == 5:
                    next_state, reward, terminated, truncated, info = step_res
                elif isinstance(step_res, tuple) and len(step_res) == 4:
                    next_state, reward, done, info = step_res
                    terminated = bool(done)
                    truncated = False
                else:
                    raise RuntimeError("env.step returned unexpected result shape")
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        average_reward = total_reward / max(1, num_episodes)
        return average_reward

    # Utility: build an atomic action from a list of station indices
    def build_atomic_action(self, station_indices: list[int]):
        """Return a dict action suitable for the env.create_atomic action.

        The returned dict contains action_type=6, atomic_len and atomic_stations
        (padded to env.max_atomic_len).
        """
        L = min(len(station_indices), self.env.max_atomic_len)
        padded = [0] * self.env.max_atomic_len
        for i in range(L):
            padded[i] = int(station_indices[i])
        return {
            "action_type": 6,
            "atomic_len": int(L),
            "atomic_stations": padded,
            "station": 0,
            "slot": 0,
        }

    # Small helper to flatten observation dict into a 1D numpy array (deterministic)
    def flatten_observation(self, obs: dict) -> np.ndarray:
        parts = []
        parts.append(np.asarray(obs["stations_passenger_counts"]).ravel().astype(np.float32))
        parts.append(np.asarray(obs["station_shape_ids"]).ravel().astype(np.float32))
        parts.append(np.asarray(obs["paths_station_mask"]).ravel().astype(np.float32))
        parts.append(np.asarray(obs["metro_on_station"]).ravel().astype(np.float32))
        parts.append(np.asarray(obs["metro_progress"]).ravel().astype(np.float32))
        parts.append(np.asarray(obs["score"]).ravel().astype(np.float32))
        parts.append(np.asarray([obs["game_over"]]).ravel().astype(np.float32))
        return np.concatenate(parts)