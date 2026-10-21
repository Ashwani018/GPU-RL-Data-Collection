"""cpu_baseline.py - APPROACH 0: single-core CPU, one game at a time.

Mirrors the paper's status-quo baseline (a Python loop over simulators on one
CPU core): for every step, loop over all N games and advance each by one tick.
Plain Python, no GPU, no compilation.

usage:  python src/cpu_baseline.py N K [--reps R] [--save f] [--traj f]
        python src/cpu_baseline.py --sweep     (N up to 262,144; large runs are timed once)
"""
import sys
import time

import numpy as np

from cartpole import build_sim, INIT_STEP
from common import (parse_args, median_ms, print_result, print_report,
                    SWEEP_NS, SWEEP_KS, CPU_MAX_N, CPU_FULL_REPS_MAX_SAMPLES)

sim = build_sim(lambda f: f)        # plain Python: the functions are NOT compiled
step_game = sim.step_game


def initial_states(N, seed):
    """Every approach starts from the same N initial states."""
    s = np.empty((N, 4), dtype=np.float32)
    for i in range(N):
        s[i] = sim.initial_state(seed, i, INIT_STEP)
    return s


def collect(states, N, K, seed):
    """K steps of N games, one game at a time. Returns how many episodes ended."""
    episodes = 0
    for t in range(K):                      # K steps ...
        for i in range(N):                  # ... of N games, one after another
            x, v, th, w, done = step_game(states[i, 0], states[i, 1], states[i, 2],
                                          states[i, 3], i, t, seed)
            states[i, 0] = x
            states[i, 1] = v
            states[i, 2] = th
            states[i, 3] = w
            episodes += done
    return episodes


def run(N, K, reps, seed, save=None, report=False):
    init = initial_states(N, seed)
    times = []
    for _ in range(reps):
        states = init.copy()                # every repetition starts from the same states
        t0 = time.perf_counter()
        episodes = collect(states, N, K, seed)
        times.append(time.perf_counter() - t0)
    print_result("cpu", N, K, reps, median_ms(times), states, save)
    if episodes:                            # sanity: gym random-action episodes last ~22 steps
        print(f"sanity: {episodes} episodes ended, mean length {N * K / episodes:.1f} steps",
              file=sys.stderr)
    if report:
        print_report("cpu", N, K, reps, seed, times, init[:3], states, launches=0)


def write_trajectories(path, seed):
    """A few recorded games for the animation. Columns: game,t,x,v,theta,omega,action"""
    with open(path, "w") as f:
        f.write("game,t,x,v,theta,omega,action\n")
        for g in range(4):
            x, v, th, w = sim.initial_state(seed, g, INIT_STEP)
            for t in range(300):
                f.write(f"{g},{t},{x:.7f},{v:.7f},{th:.7f},{w:.7f},{sim.choose_action(seed, g, t)}\n")
                x, v, th, w, _ = step_game(x, v, th, w, g, t, seed)


if __name__ == "__main__":
    a = parse_args(default_reps=3)
    if a.sweep:
        for K in SWEEP_KS:
            for N in SWEEP_NS:
                if N <= CPU_MAX_N:          # the CPU is too slow beyond this many games
                    reps = a.reps if N * K <= CPU_FULL_REPS_MAX_SAMPLES else 1   # big runs: time once
                    run(N, K, reps, a.seed)
    else:
        run(a.N, a.K, a.reps, a.seed, a.save, a.report)
    if a.traj:
        write_trajectories(a.traj, a.seed)
