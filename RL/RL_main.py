"""Train a PPO agent on the MiniMetroEnv.

This is a compact, self-contained PPO trainer (single-process, single-worker)
provided as a starting point. It demonstrates how to:
- flatten the environment's Dict observation
- create a multi-head policy that matches the env's multi-discrete actions
- collect rollouts, compute GAE advantages, and run PPO updates

Notes:
- This trainer is intentionally simple for clarity. For faster training use
  vectorized environments or Ray RLlib / Stable Baselines3.
- You need PyTorch installed (torch). Install with: pip3 install torch
"""

from __future__ import annotations

import time
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from RL_env import MiniMetroEnv
from RL_agent import RLAgent
import argparse


def flatten_obs(agent: RLAgent, obs: dict) -> np.ndarray:
	return agent.flatten_observation(obs)


class PPOPolicy(nn.Module):
	def __init__(self, obs_dim: int, env: MiniMetroEnv, hidden_size: int = 256):
		super().__init__()
		self.env = env
		self.hidden = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

		# heads
		self.action_type_head = nn.Linear(hidden_size, 7)  # action types
		self.station_head = nn.Linear(hidden_size, env.observation_space['station_shape_ids'].n if hasattr(env.observation_space['station_shape_ids'], 'n') else env.observation_space['station_shape_ids'].shape[0])
		self.slot_head = nn.Linear(hidden_size, env.action_space['slot'].n)
		self.atomic_len_head = nn.Linear(hidden_size, env.max_atomic_len + 1)
		self.atomic_station_heads = nn.ModuleList([nn.Linear(hidden_size, env.observation_space['station_shape_ids'].n if hasattr(env.observation_space['station_shape_ids'], 'n') else env.observation_space['station_shape_ids'].shape[0]) for _ in range(env.max_atomic_len)])

		self.value_head = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor):
		h = self.hidden(x)
		return {
			'action_type': self.action_type_head(h),
			'station': self.station_head(h),
			'slot': self.slot_head(h),
			'atomic_len': self.atomic_len_head(h),
			'atomic_stations': [head(h) for head in self.atomic_station_heads],
			'value': self.value_head(h).squeeze(-1),
		}


def sample_action_from_logits(logits: dict) -> Tuple[dict, torch.Tensor, torch.Tensor]:
	# logits are torch tensors (batch dim 1)
	atype_logits = logits['action_type']
	station_logits = logits['station']
	slot_logits = logits['slot']
	alen_logits = logits['atomic_len']
	atomic_logits_list = logits['atomic_stations']

	atype_dist = torch.distributions.Categorical(logits=atype_logits)
	station_dist = torch.distributions.Categorical(logits=station_logits)
	slot_dist = torch.distributions.Categorical(logits=slot_logits)
	alen_dist = torch.distributions.Categorical(logits=alen_logits)

	atype = atype_dist.sample()
	station = station_dist.sample()
	slot = slot_dist.sample()
	alen = alen_dist.sample()

	atomic_stations = []
	atomic_logprob = 0.0
	for i in range(alen.item() if isinstance(alen, torch.Tensor) else int(alen)):
		logits_i = atomic_logits_list[i]
		dist_i = torch.distributions.Categorical(logits=logits_i)
		si = dist_i.sample()
		atomic_stations.append(si.item())
		atomic_logprob = atomic_logprob + dist_i.log_prob(si)

	# compute joint logprob (sum of heads)
	logprob = atype_dist.log_prob(atype) + station_dist.log_prob(station) + slot_dist.log_prob(slot) + alen_dist.log_prob(alen) + atomic_logprob

	action = {
		'action_type': int(atype.item()),
		'station': int(station.item()),
		'slot': int(slot.item()),
		'atomic_len': int(alen.item()),
		'atomic_stations': atomic_stations,
	}
	return action, logprob, logits['value']


def build_gym_action_from_sampled(action: dict, env: MiniMetroEnv) -> dict:
	padded = [0] * env.max_atomic_len
	for i, s in enumerate(action['atomic_stations']):
		if i >= env.max_atomic_len:
			break
		padded[i] = int(s)
	return {
		'action_type': int(action['action_type']),
		'station': int(action['station']),
		'slot': int(action['slot']),
		'atomic_len': int(action['atomic_len']),
		'atomic_stations': padded,
	}


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
	advs = np.zeros_like(rewards)
	lastgaelam = 0
	for t in reversed(range(len(rewards))):
		if t == len(rewards) - 1:
			nextnonterminal = 1.0 - dones[-1]
			nextvalues = last_value
		else:
			nextnonterminal = 1.0 - dones[t+1]
			nextvalues = values[t+1]
		delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
		lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
		advs[t] = lastgaelam
	returns = advs + values
	return advs, returns


def train(
    	total_updates: int = 200,
    	steps_per_update: int = 1024,
    	minibatch_size: int = 64,
    	ppo_epochs: int = 4,
    	lr: float = 3e-4,
    	plot: bool = False,
    	plot_path: str = "training_rewards.png",
    	csv_path: str = "training_rewards.csv",
    	verbose: bool = False,
    	# when True, collect rollouts as complete episodes (each episode runs until terminated=True)
    	collect_full_episodes: bool = False,
    # checkpointing
    	checkpoint_path: Optional[str] = None,
    	load_checkpoint: Optional[str] = None,
    	save_every: int = 1,
):
	env = MiniMetroEnv()
	agent = RLAgent(env)

	# get observation dim
	obs0, _ = env.reset()
	obs_vec0 = flatten_obs(agent, obs0)
	obs_dim = int(obs_vec0.size)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	policy = PPOPolicy(obs_dim, env).to(device)
	optimizer = optim.Adam(policy.parameters(), lr=lr)

	# optionally load checkpoint to resume training
	start_update = 0
	if load_checkpoint:
		if os.path.exists(load_checkpoint):
			if verbose:
				print(f"Loading checkpoint from {load_checkpoint}")
			ckpt = torch.load(load_checkpoint, map_location=device)
			policy.load_state_dict(ckpt.get("policy_state", {}))
			opt_state = ckpt.get("optimizer_state", None)
			if opt_state is not None:
				optimizer.load_state_dict(opt_state)
			start_update = int(ckpt.get("update", 0)) + 1
			if verbose:
				print(f"Resuming from update {start_update}")
		else:
			print(f"Warning: load_checkpoint path not found: {load_checkpoint}")

	# tracking
	mean_rewards_per_update = []

	for update in range(start_update, total_updates):
		obs_buf = []
		actions_buf = []
		logp_buf = []
		rewards_buf = []
		dones_buf = []
		values_buf = []

		# diagnostics counters for this update
		action_effect_count = 0
		nonzero_reward_count = 0

		# reset may return obs or (obs, info)
		res = env.reset()
		if isinstance(res, tuple) and len(res) == 2:
			obs, _ = res
		else:
			obs = res
		step = 0
		# collect steps until we reach steps_per_update. If collect_full_episodes is True,
		# we ensure that episodes are collected to completion (terminated==True) before
		# starting the next episode; otherwise we behave as before and reset mid-update.
		while step < steps_per_update:
			x = torch.from_numpy(flatten_obs(agent, obs)).float().to(device).unsqueeze(0)
			with torch.no_grad():
				out = policy(x)
			action, logp, value = sample_action_from_logits(out)
			gym_action = build_gym_action_from_sampled(action, env)

			step_res = env.step(gym_action)
			# support both gym and gymnasium signatures
			if isinstance(step_res, tuple) and len(step_res) == 5:
				obs2, reward, terminated, truncated, info = step_res
			elif isinstance(step_res, tuple) and len(step_res) == 4:
				obs2, reward, done, info = step_res
				terminated = bool(done)
				truncated = False
			else:
				raise RuntimeError("env.step returned unexpected result shape")

			# store step data
			obs_buf.append(flatten_obs(agent, obs))
			actions_buf.append(action)
			logp_buf.append(logp.cpu().item() if isinstance(logp, torch.Tensor) else float(logp))
			rewards_buf.append(float(reward))
			# diagnostic counters
			if isinstance(info, dict) and info.get("action_effect", False):
				action_effect_count += 1
			if float(reward) != 0.0:
				nonzero_reward_count += 1
			dones_buf.append(bool(terminated or truncated))
			values_buf.append(float(value.cpu().item() if isinstance(value, torch.Tensor) else value))

			step += 1
			obs = obs2

			# If an episode ended (terminated) reset before starting the next episode
			if terminated:
				res = env.reset()
				if isinstance(res, tuple) and len(res) == 2:
					obs, _ = res
				else:
					obs = res
			elif truncated:
				# truncated due to max_steps; reset and continue collecting
				res = env.reset()
				if isinstance(res, tuple) and len(res) == 2:
					obs, _ = res
				else:
					obs = res

		# bootstrap last value
		x = torch.from_numpy(flatten_obs(agent, obs)).float().to(device).unsqueeze(0)
		with torch.no_grad():
			last_value = policy(x)['value'].cpu().item()

		rewards = np.array(rewards_buf, dtype=np.float32)
		values = np.array(values_buf, dtype=np.float32)
		dones = np.array(dones_buf, dtype=np.float32)

		advs, returns = compute_gae(rewards, values, dones, last_value)

		# convert to tensors for optimization
		obs_tensor = torch.tensor(np.array(obs_buf, dtype=np.float32)).to(device)
		returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
		advs_tensor = torch.tensor(advs, dtype=torch.float32).to(device)

		# policy update (simplified: recompute logits for batches)
		dataset_size = len(obs_buf)
		inds = np.arange(dataset_size)
		for epoch in range(ppo_epochs):
			np.random.shuffle(inds)
			for start in range(0, dataset_size, minibatch_size):
				mb_inds = inds[start:start+minibatch_size]
				mb_obs = torch.from_numpy(np.array([obs_buf[i] for i in mb_inds])).float().to(device)

				out = policy(mb_obs)
				# Compute new logprobs, entropies and values for minibatch actions
				new_logps = []
				ent_list = []
				vals = out['value'] if isinstance(out['value'], torch.Tensor) else torch.tensor(out['value'])
				# Ensure logits lists are accessible per-head
				atype_logits = out['action_type']
				station_logits = out['station']
				slot_logits = out['slot']
				alen_logits = out['atomic_len']
				atomic_logits_list = out['atomic_stations']

				for b, idx in enumerate(mb_inds):
					action = actions_buf[int(idx)]
					# per-head categorical dists for the b-th batch element
					atype_dist = torch.distributions.Categorical(logits=atype_logits[b])
					station_dist = torch.distributions.Categorical(logits=station_logits[b])
					slot_dist = torch.distributions.Categorical(logits=slot_logits[b])
					alen_dist = torch.distributions.Categorical(logits=alen_logits[b])

					# logprobs for selected indices
					atype_lp = atype_dist.log_prob(torch.tensor(action['action_type'], device=device))
					station_lp = station_dist.log_prob(torch.tensor(action['station'], device=device))
					slot_lp = slot_dist.log_prob(torch.tensor(action['slot'], device=device))
					alen_lp = alen_dist.log_prob(torch.tensor(action['atomic_len'], device=device))

					atomic_lp_sum = torch.tensor(0.0, device=device)
					atomic_ent_sum = torch.tensor(0.0, device=device)
					# only sum atomic heads up to atomic_len
					for i in range(int(action['atomic_len'])):
						logits_i = atomic_logits_list[i][b]
						dist_i = torch.distributions.Categorical(logits=logits_i)
						si = torch.tensor(int(action['atomic_stations'][i]) if i < len(action['atomic_stations']) else 0, device=device)
						atomic_lp_sum = atomic_lp_sum + dist_i.log_prob(si)
						atomic_ent_sum = atomic_ent_sum + dist_i.entropy()

					new_logp = atype_lp + station_lp + slot_lp + alen_lp + atomic_lp_sum
					ent = atype_dist.entropy() + station_dist.entropy() + slot_dist.entropy() + alen_dist.entropy() + atomic_ent_sum
					new_logps.append(new_logp)
					ent_list.append(ent)

				new_logp_tensor = torch.stack(new_logps)
				ent_tensor = torch.stack(ent_list)

				# old logprobs and advantages/returns for minibatch
				old_logp_mb = torch.tensor([logp_buf[int(i)] for i in mb_inds], dtype=torch.float32, device=device)
				adv_mb = advs_tensor[mb_inds].to(device)
				ret_mb = returns_tensor[mb_inds].to(device)
				val_mb = vals if isinstance(vals, torch.Tensor) else torch.tensor(vals, device=device)
				val_mb = val_mb if val_mb.dim() == 1 else val_mb.squeeze(-1)
				val_mb = val_mb[: len(mb_inds)]

				# normalize advantage
				adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

				# ratio and clipped surrogate loss
				ratio = torch.exp(new_logp_tensor - old_logp_mb)
				clip_eps = 0.2
				surr1 = ratio * adv_mb
				surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
				policy_loss = -torch.mean(torch.min(surr1, surr2))

				# value loss (MSE)
				values_pred = out['value'][: len(mb_inds)]
				value_loss = torch.mean((values_pred - ret_mb) ** 2)

				# entropy bonus
				entropy = torch.mean(ent_tensor)

				# total loss
				value_coef = 0.5
				entropy_coef = 0.01
				total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

				optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
				optimizer.step()

		mean_reward = float(rewards.mean()) if len(rewards) > 0 else 0.0
		mean_rewards_per_update.append(mean_reward)
		if verbose:
			print(f"Update {update+1}/{total_updates} collected {dataset_size} steps. Mean reward: {mean_reward:.3f}")
			print(f"  nonzero_reward_steps: {nonzero_reward_count}, action_effects: {action_effect_count}, sum_reward: {float(np.sum(rewards)):.3f}")

		# checkpoint save: use 1-based update numbering so "Update 1" doesn't map to internal index 0
		# this saves on logical updates 1, save_every, 2*save_every, ...
		if checkpoint_path and (save_every > 0) and (((update + 1) % save_every) == 0):
			try:
				dirpath = os.path.dirname(checkpoint_path)
				if dirpath:
					os.makedirs(dirpath, exist_ok=True)
				save_ckpt = {
					"update": int(update),
					"policy_state": policy.state_dict(),
					"optimizer_state": optimizer.state_dict(),
				}
				torch.save(save_ckpt, checkpoint_path)
				if verbose:
					print(f"Saved checkpoint to {checkpoint_path} (update {update})")
			except Exception as e:
				print(f"Failed to save checkpoint to {checkpoint_path}: {e}")

	# optionally plot rewards over updates
	if plot:
		try:
			# use a non-interactive backend so savefig works in headless environments
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
			plt.figure(figsize=(8, 4))
			plt.plot(np.arange(1, len(mean_rewards_per_update) + 1), mean_rewards_per_update, marker='o')
			plt.xlabel('Update')
			plt.ylabel('Mean reward')
			plt.title('Training progress: mean reward per update')
			plt.grid(True)
			plt.tight_layout()
			try:
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
					print(f"Failed to save plot to {plot_path}: {e}")
			# also save CSV
			try:
				import csv
				with open(csv_path, 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(['update', 'mean_reward'])
					for i, r in enumerate(mean_rewards_per_update, start=1):
						writer.writerow([i, r])
				if verbose:
					print(f"Saved rewards CSV to {csv_path}")
			except Exception:
				pass
		except Exception:
			if verbose:
				print("matplotlib not available; install matplotlib to enable plotting: pip install matplotlib")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train an agent on MiniMetroEnv')
	parser.add_argument('--mode', choices=['ppo', 'random'], default='ppo', help='Training mode: ppo or random baseline')
	parser.add_argument('--episodes', type=int, default=100, help='Number of episodes (for random baseline)')
	parser.add_argument('--updates', type=int, default=50, help='Number of PPO updates')
	parser.add_argument('--steps', type=int, default=16384, help='Steps per PPO update')
	parser.add_argument('--minibatch', type=int, default=64, help='PPO minibatch size')
	parser.add_argument('--epochs', type=int, default=4, help='PPO epochs per update')
	parser.add_argument('--plot', action='store_true', help='Save reward plot and CSV after training')
	parser.add_argument('--verbose', action='store_true', help='Enable verbose logging during training')
	parser.add_argument('--checkpoint-path', type=str, default='checkpoints/ppo_ckpt.pth', help='Path to save checkpoints')
	parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint to load before training')
	parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N updates')
	args = parser.parse_args()
	# to plot and verbose every time by default
	args.plot = True
	args.verbose = True

	env = MiniMetroEnv()
	agent = RLAgent(env)

	if args.mode == 'random':
		if args.verbose:
			print(f"Running random baseline for {args.episodes} episodes...")
		agent.train(args.episodes, plot=args.plot, plot_path='random_baseline_rewards.png', csv_path='random_baseline_rewards.csv',)
	else:
		if args.verbose:
			print(f"Running PPO trainer: {args.updates} updates, {args.steps} steps/update")
		train(
			total_updates=args.updates,
			steps_per_update=args.steps,
			minibatch_size=args.minibatch,
			ppo_epochs=args.epochs,
			plot=args.plot,
			verbose=args.verbose,
			checkpoint_path=args.checkpoint_path,
			load_checkpoint=args.load_checkpoint,
			save_every=args.save_every,
		)