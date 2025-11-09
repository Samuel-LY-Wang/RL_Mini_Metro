"""Debug runner for MiniMetroEnv reward behaviour.

Runs short random-action episodes under different reward modes and prints
mediator state so we can see why rewards may be zero.
"""
from RL_env import MiniMetroEnv
import time


def run_mode(mode: str, steps: int = 1000):
    print(f"\n=== Running reward_mode={mode} for {steps} steps ===")
    env = MiniMetroEnv(reward_mode=mode)
    obs, info = env.reset()
    total_reward = 0.0
    for i in range(steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_reward += float(r)
        # print occasional diagnostics
        if i % 50 == 0:
            med = env.mediator
            print(
                f"step={i:4d} reward={r:+.3f} total={total_reward:+.3f} "
                f"score={med.score} passengers={len(med.passengers)} paths={len(med.paths)} metros={len(med.metros)}"
            )
        if term or trunc:
            print("Environment terminated at step", i)
            break
    print(f"Finished: total_reward={total_reward:+.3f} final_score={env.mediator.score}")


if __name__ == "__main__":
    # Try short runs for a few reward modes to inspect signal
    for mode in ("delta", "absolute", "final"):
        run_mode(mode, steps=1000)
        # small pause to avoid immediate re-use issues
        time.sleep(0.2)

    # --- Oracle test: create a simple path and force passenger spawn ---
    print("\n=== Oracle test: make a path [0,1], spawn passengers and run ===")
    env = MiniMetroEnv(reward_mode="delta")
    obs, info = env.reset()
    med = env.mediator

    # create three atomic paths that together cover all stations (adapt to station count)
    n = len(med.stations)
    # split station indices into 3 groups as evenly as possible
    groups = []
    if n <= 3:
        groups = [[i] for i in range(n)]
    else:
        base = n // 3
        rem = n % 3
        start = 0
        for g in range(3):
            size = base + (1 if g < rem else 0)
            group = list(range(start, start + max(1, size)))
            groups.append(group)
            start += max(1, size)

    created_any = 0
    for grp in groups:
        # ensure group has at least 2 stations for a useful path; if single, skip
        if len(grp) < 2:
            continue
        created = env._create_atomic_path(grp)
        print(f"Created path {grp}? {bool(created)}")
        if created:
            created_any += 1
    print("Paths now:", len(med.paths))

    # force a passenger spawn to populate stations (use mediator.spawn_passengers())
    med.spawn_passengers()
    med.find_travel_plan_for_passengers()
    print("Passengers after spawn:", len(med.passengers))

    total = 0.0
    for i in range(2000):
        # take NOOPs to let metros run
        a = {"action_type": 0, "station": 0, "slot": 0, "atomic_len": 0, "atomic_stations": [0] * env.max_atomic_len}
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        if i % 100 == 0:
            print(f"oracle step={i} reward={r:+.3f} total={total:+.3f} score={env.mediator.score}")
        if term:
            print("oracle: terminated at step", i)
            break
    print("Oracle finished: total_reward=", total, "final_score=", env.mediator.score)