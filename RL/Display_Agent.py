"""
Display a Mini Metro game seeded by a trained policy checkpoint.

Usage:
  python -m RL.Display_Agent --checkpoint checkpoints/chkpt_16384.pth --max-create-steps 2000

What it does:
- Creates a headless RL environment wrapper and loads the trained policy.
- Steps the policy until it has created a set of paths (or until max steps reached).
- Takes the environment.mediator (game state) populated by the policy and
  runs the regular pygame display loop using that mediator so you can watch
  the game built by the agent.

Notes:
- This script does not modify `src/main.py`; it creates and displays its own
  mediator instance initialized by the RL policy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# make src importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pygame
import torch

from RL_main import PPOPolicy, sample_action_from_logits, build_gym_action_from_sampled
from RL_agent import RLAgent
from RL_env import MiniMetroEnv

# Import UI constants from src
from config import framerate, screen_color, screen_height, screen_width


def build_policy_from_checkpoint(env: MiniMetroEnv, ckpt_path: str, device: str | torch.device = 'cpu') -> tuple[PPOPolicy, torch.device]:
    agent = RLAgent(env)
    obs0, _ = env.reset()
    obs_vec0 = agent.flatten_observation(obs0)
    obs_dim = int(obs_vec0.size)

    device = torch.device(device)
    policy = PPOPolicy(obs_dim, env).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('policy_state', ckpt)
    policy.load_state_dict(state)
    policy.eval()
    return policy, device


def run_policy_to_create_paths(policy: PPOPolicy, device: torch.device, env: MiniMetroEnv, max_steps: int = 2000, target_num_paths: int | None = None, stochastic: bool = False):
    """Run the policy in the RL env until target_num_paths reached or max_steps.

    Returns the environment.mediator populated by the policy.
    """
    agent = RLAgent(env)
    obs, _ = env.reset()

    created_steps = 0
    for step in range(max_steps):
        x = torch.from_numpy(agent.flatten_observation(obs)).float().to(device).unsqueeze(0)
        with torch.no_grad():
            out = policy(x)

        if stochastic:
            action, logp, _ = sample_action_from_logits(out)
        else:
            # deterministic: pick argmax per head
            def logits_argmax(t):
                t = t.detach().cpu()
                if t.dim() == 2:
                    return int(torch.argmax(t[0]).item())
                elif t.dim() == 1:
                    return int(torch.argmax(t).item())
                else:
                    return 0

            atomic_stations = []
            atype = logits_argmax(out['action_type'])
            station = logits_argmax(out['station'])
            slot = logits_argmax(out['slot'])
            alen = logits_argmax(out['atomic_len'])
            for i in range(alen):
                head = out['atomic_stations'][i]
                atomic_stations.append(int(torch.argmax(head.detach().cpu()).item()))

            action = {
                'action_type': int(atype),
                'station': int(station),
                'slot': int(slot),
                'atomic_len': int(alen),
                'atomic_stations': atomic_stations,
            }

        gym_action = build_gym_action_from_sampled(action, env)
        obs, r, terminated, truncated, info = env.step(gym_action)

        if info.get('action_effect', False):
            created_steps += 1

        # stop early if we've created the maximum allowed number of paths
        if target_num_paths is None:
            target_num_paths = env.mediator.num_paths
        if len(env.mediator.paths) >= target_num_paths:
            print(f"Reached target paths: {len(env.mediator.paths)} at step {step}")
            break

    print(f"Finished creation loop: created_effective_actions={created_steps}, total_paths={len(env.mediator.paths)}")
    return env.mediator


def display_mediator(mediator):
    pygame.init()
    flags = pygame.SCALED
    screen = pygame.display.set_mode((screen_width, screen_height), flags)
    clock = pygame.time.Clock()

    while True:
        dt_ms = clock.tick(framerate)
        mediator.increment_time(dt_ms)
        screen.fill(screen_color)
        mediator.render(screen)

        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                raise SystemExit
            # Mediator expects converted events; reuse the event.convert helper
            from event.convert import convert_pygame_event
            event = convert_pygame_event(pygame_event)
            mediator.react(event)

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/chkpt_16384.pth')
    parser.add_argument('--max-create-steps', type=int, default=2000)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    env = MiniMetroEnv()
    policy, device = build_policy_from_checkpoint(env, args.checkpoint)

    print("Running policy to create paths...")
    mediator = run_policy_to_create_paths(policy, device, env, max_steps=args.max_create_steps, stochastic=not args.deterministic)

    print("Launching display using generated mediator state...")
    display_mediator(mediator)


if __name__ == '__main__':
    main()
