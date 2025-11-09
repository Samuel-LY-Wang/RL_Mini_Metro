'''
The Mini Metro environment with no UI
For RL agents to interact with
'''

import os
import sys
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# make the project's `src/` directory importable as top-level modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# import as top-level modules (the code in src/ expects imports like `from config import ...`)
from config import num_metros, num_paths, num_stations
from mediator import Mediator


class MiniMetroEnv(gym.Env):
    """Gym wrapper for Mini Metro using the existing src.mediator.Mediator.

        Action space (Option B, extended for multi-station paths):
            - action_type: Discrete
                    0 = NOOP
                    1 = START_PATH (station -> begins a new path at `station`)
                    2 = ADD_STATION (station -> append station to path being created)
                    3 = FINISH_PATH (no station needed; finalize current path)
                    4 = REMOVE_PATH (slot -> remove path in slot)
                    5 = ABORT_PATH (cancel path creation in progress)
            - station: station index used by START_PATH and ADD_STATION
            - slot: slot index used by REMOVE_PATH

    Observation is a Dict with station passenger counts, station shape ids,
    path->station mask, metro locations, score and game_over flag.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, step_dt_ms: int = 100, reward_per_delivery: float = 1.0, max_steps: int = 1000, seed: Optional[int] = None, action_penalty: float = 0.01, final_score_scale: float = 1.0, reward_mode: str = 'delta', boarding_reward: float = 0.2, connected_station_reward: float = 0.01):
        super().__init__()
        self.step_dt_ms = int(step_dt_ms)
        # reward_per_delivery kept for backward compatibility but not used by default
        self.reward_per_delivery = float(reward_per_delivery)
        # per-action penalty (applied when the agent issues a non-NOOP action)
        self.action_penalty = float(action_penalty)
        # final reward scale applied to mediator.score when game-over occurs
        self.final_score_scale = float(final_score_scale)
        self.max_steps = int(max_steps)
        # reward_mode controls how step rewards are computed:
        #  - 'delta' (default): reward = change in mediator.score during the step
        #  - 'final': reward = 0 each step, full score awarded only at game over (legacy behavior)
        #  - 'absolute': reward = current mediator.score each step (not recommended)
        self.reward_mode = str(reward_mode)
        # small dense rewards to help learning
        # reward per passenger boarding onto a metro (station -> metro)
        self.boarding_reward = float(boarding_reward)
        # reward per station that is connected to at least one path (per-step)
        self.connected_station_reward = float(connected_station_reward)
        self._seed = None if seed is None else int(seed)
        if self._seed is not None:
            np.random.seed(self._seed)

        # mediator will be created on reset()
        self.mediator: Optional[Mediator] = None

        # observation space
        self.observation_space = spaces.Dict(
            {
                "stations_passenger_counts": spaces.Box(low=0.0, high=1.0, shape=(num_stations,), dtype=np.float32),
                "station_shape_ids": spaces.MultiDiscrete([32] * num_stations),
                "paths_station_mask": spaces.MultiBinary((num_paths, num_stations)),
                "metro_on_station": spaces.MultiBinary((num_metros, num_stations)),
                "metro_progress": spaces.Box(low=0.0, high=1.0, shape=(num_metros,), dtype=np.float32),
                "score": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "game_over": spaces.Discrete(2),
            }
        )

        # action space (extended interactive actions for multi-station creation)
        # action_type values:
        #   0 NOOP, 1 START, 2 ADD, 3 FINISH, 4 REMOVE, 5 ABORT, 6 CREATE_ATOMIC
        # For atomic creation the agent supplies a length and a fixed-size list of station indices.
        self.max_atomic_len = num_stations  # maximum stations in an atomic create (worst case)
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(7),
                "station": spaces.Discrete(num_stations),
                "slot": spaces.Discrete(num_paths),
                # atomic action fields
                "atomic_len": spaces.Discrete(self.max_atomic_len + 1),  # 0..max_atomic_len
                "atomic_stations": spaces.MultiDiscrete([num_stations] * self.max_atomic_len),
            }
        )

        self.current_step = 0

    def seed(self, seed: Optional[int] = None):
        self._seed = None if seed is None else int(seed)
        np.random.seed(self._seed)
        return [self._seed]

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        # create fresh mediator (game logic)
        self.mediator = Mediator()
        self.current_step = 0
        # Modern Gym API expects (obs, info) - return empty info dict
        return self._get_observation(), {}

    def step(self, action):
        if self.mediator is None:
            raise RuntimeError("Call reset() before step()")

        action_type = int(action.get("action_type", 0))
        slot = int(action.get("slot", 0))
        station_idx = int(action.get("station", 0))
        atomic_len = int(action.get("atomic_len", 0))
        atomic_stations = action.get("atomic_stations", None)

        reward = 0.0
        # record previous score to compute delta if using 'delta' reward mode
        prev_score = float(self.mediator.score)
        info = {}

        # apply action (interactive multi-step creation)
        action_effect = False
        try:
            if action_type == 1:  # START_PATH
                action_effect = bool(self._start_path(station_idx))
            elif action_type == 2:  # ADD_STATION
                action_effect = bool(self._add_station_to_path(station_idx))
            elif action_type == 3:  # FINISH_PATH
                action_effect = bool(self._finish_path())
            elif action_type == 4:  # REMOVE_PATH
                action_effect = bool(self._try_remove_path(slot))
            elif action_type == 5:  # ABORT_PATH
                action_effect = bool(self._abort_path())
            elif action_type == 6:  # CREATE_ATOMIC
                # atomic_stations expected as a sequence-like of length max_atomic_len
                if atomic_stations is None:
                    # some wrappers may pass numpy arrays; try to tolerate
                    atomic_stations = []
                # take first `atomic_len` entries
                try:
                    seq = [int(x) for x in list(atomic_stations)[:atomic_len]]
                except Exception:
                    seq = []
                if len(seq) > 0:
                    action_effect = bool(self._create_atomic_path(seq))
            # else 0 NOOP
        except Exception as e:
            info["action_error"] = str(e)
        # expose whether the action changed environment state for diagnostics
        info["action_effect"] = bool(action_effect)

        # small penalty for taking actions (non-NOOP) ONLY if the action changed state
        if action_type != 0 and action_effect:
            reward -= self.action_penalty

        # advance the world
        # count connected stations BEFORE the increment so we can reward current connectivity
        before_connected = 0
        if self.mediator is not None:
            seen = set()
            for p in self.mediator.paths:
                for st in p.stations:
                    seen.add(st)
            before_connected = len(seen)

        self.mediator.increment_time(self.step_dt_ms)

        # compute score-based reward according to reward_mode
        curr_score = float(self.mediator.score)
        if self.reward_mode == 'delta':
            # give the change in score as immediate reward
            reward += (curr_score - prev_score) * self.final_score_scale
        elif self.reward_mode == 'absolute':
            # give absolute score each step (cumulative over time; use with care)
            reward += curr_score * self.final_score_scale

        # dense auxiliary rewards: boardings and connected stations
        # mediator exposes last_step_boardings (set during increment_time)
        try:
            boardings = int(getattr(self.mediator, 'last_step_boardings', 0))
        except Exception:
            boardings = 0
        reward += boardings * self.boarding_reward

        # reward for number of connected stations (after the increment)
        after_connected = 0
        if self.mediator is not None:
            seen2 = set()
            for p in self.mediator.paths:
                for st in p.stations:
                    seen2.add(st)
            after_connected = len(seen2)
        reward += float(after_connected) * self.connected_station_reward

        # attach these diagnostics to info
        info['boardings'] = boardings
        info['connected_stations'] = after_connected

        # only give the final score as extra reward when using 'final' mode
        # (legacy behavior)

        self.current_step += 1
        obs = self._get_observation()
        terminated = bool(self.mediator.game_over)
        truncated = (self.current_step >= self.max_steps)
        if self.mediator.game_over:
            info["game_over_reason"] = "station_overcrowded"
            # final reward only if using legacy 'final' mode
            if self.reward_mode == 'final':
                reward += float(self.mediator.score) * self.final_score_scale

        # Modern Gym API: return (obs, reward, terminated, truncated, info)
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        if self.mediator is None:
            print("Environment not reset.")
            return
        if mode == "human":
            print(f"Step {self.current_step} | Score: {self.mediator.score} | Paths: {len(self.mediator.paths)} | Metros: {len(self.mediator.metros)}")
        elif mode == "rgb_array":
            raise NotImplementedError("rgb_array render not implemented")
        else:
            raise ValueError("Unknown render mode: " + str(mode))

    def close(self):
        self.mediator = None

    # ----------------------
    # Helpers
    # ----------------------
    def _get_observation(self):
        assert self.mediator is not None
        m = self.mediator

        station_counts = np.zeros((len(m.stations),), dtype=np.float32)
        for i, s in enumerate(m.stations):
            count = len(s.passengers) if hasattr(s, "passengers") else 0
            cap = getattr(s, "capacity", 10)
            station_counts[i] = float(count) / float(max(1, cap))

        shape_ids = np.zeros((len(m.stations),), dtype=np.int64)
        for i, s in enumerate(m.stations):
            # ShapeType in the project is an Enum whose .value is a string like "1".
            # Safely extract a numeric id for the shape.
            raw = getattr(s.shape, "type", 0)
            raw_val = getattr(raw, "value", raw)
            try:
                shape_ids[i] = int(raw_val)
            except Exception:
                # fallback to 0
                shape_ids[i] = 0

        path_mask = np.zeros((num_paths, len(m.stations)), dtype=np.int8)
        for p_idx, path in enumerate(m.paths):
            for st in path.stations:
                try:
                    s_idx = m.stations.index(st)
                    if p_idx < num_paths:
                        path_mask[p_idx, s_idx] = 1
                except ValueError:
                    continue

        metro_on_station = np.zeros((num_metros, len(m.stations)), dtype=np.int8)
        metro_progress = np.zeros((num_metros,), dtype=np.float32)
        for i in range(num_metros):
            if i < len(m.metros):
                metro = m.metros[i]
                curr_station = getattr(metro, "current_station", None)
                if curr_station is not None and curr_station in m.stations:
                    idx = m.stations.index(curr_station)
                    metro_on_station[i, idx] = 1
                progress = getattr(metro, "progress", None)
                metro_progress[i] = float(progress) if progress is not None else 0.0

        score = float(m.score)
        game_over = 1 if m.game_over else 0

        obs = {
            "stations_passenger_counts": station_counts,
            "station_shape_ids": shape_ids,
            "paths_station_mask": path_mask,
            "metro_on_station": metro_on_station,
            "metro_progress": metro_progress,
            "score": np.array(score, dtype=np.float32),
            "game_over": game_over,
        }
        return obs

    def _try_create_path(self, from_idx: int, to_idx: int):
        assert self.mediator is not None
        m = self.mediator
        if not (0 <= from_idx < len(m.stations) and 0 <= to_idx < len(m.stations)):
            return
        if len(m.paths) >= m.num_paths:
            return
        if from_idx == to_idx:
            return

        start_station = m.stations[from_idx]
        end_station = m.stations[to_idx]
        before = len(m.paths)
        m.start_path_on_station(start_station)
        m.add_station_to_path(end_station)
        m.finish_path_creation()
        after = len(m.paths)
        return after > before

    # Interactive helpers for multi-station creation
    def _start_path(self, station_idx: int):
        assert self.mediator is not None
        m = self.mediator
        if not (0 <= station_idx < len(m.stations)):
            return
        if len(m.paths) >= m.num_paths:
            return
        if m.is_creating_path:
            # already creating; ignore or optionally abort and restart
            return
        before = len(m.paths)
        start_station = m.stations[station_idx]
        m.start_path_on_station(start_station)
        after = len(m.paths)
        return after > before

    def _add_station_to_path(self, station_idx: int):
        assert self.mediator is not None
        m = self.mediator
        if not m.is_creating_path or m.path_being_created is None:
            return
        if not (0 <= station_idx < len(m.stations)):
            return
        station = m.stations[station_idx]
        # record before state
        before_stations = tuple(m.path_being_created.stations)
        before_loop = getattr(m.path_being_created, 'is_looped', False)
        m.add_station_to_path(station)
        after_stations = tuple(m.path_being_created.stations)
        after_loop = getattr(m.path_being_created, 'is_looped', False)
        return (after_stations != before_stations) or (after_loop != before_loop)

    def _finish_path(self):
        assert self.mediator is not None
        m = self.mediator
        if not m.is_creating_path or m.path_being_created is None:
            return
        # finish_path_creation finalizes the path; return True if there was a path being created
        m.finish_path_creation()
        return True

    def _abort_path(self):
        assert self.mediator is not None
        m = self.mediator
        if not m.is_creating_path or m.path_being_created is None:
            return
        m.abort_path_creation()
        return True

    def _create_atomic_path(self, station_indices: list[int]):
        """Create a path in one atomic action given a sequence of station indices.

        The sequence is interpreted as [start, station2, station3, ...]. If the
        sequence is empty or invalid the call is a no-op.
        """
        assert self.mediator is not None
        m = self.mediator
        if not station_indices:
            return
        if len(m.paths) >= m.num_paths:
            return
        if m.is_creating_path:
            # if a creation is already in progress, do not clobber it
            return

        # validate indices and map to station objects
        stations = []
        for idx in station_indices:
            if not isinstance(idx, int):
                continue
            if 0 <= idx < len(m.stations):
                stations.append(m.stations[idx])

        if not stations:
            return False

        # start, add intermediate stations, finish
        before = len(m.paths)
        m.start_path_on_station(stations[0])
        for st in stations[1:]:
            m.add_station_to_path(st)
        m.finish_path_creation()
        after = len(m.paths)
        return after > before

    def _try_remove_path(self, slot_idx: int):
        assert self.mediator is not None
        m = self.mediator
        if slot_idx < 0 or slot_idx >= len(m.path_buttons):
            return
        before = len(m.paths)
        if slot_idx < len(m.paths):
            path = m.paths[slot_idx]
            m.remove_path(path)
            after = len(m.paths)
            return after < before
        else:
            return False