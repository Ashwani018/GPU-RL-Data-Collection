"""gpu_vectorized.py - APPROACH 1: GPU vectorization (one kernel launch per step).

All N games run in parallel: one GPU thread = one game. But every step is a
separate kernel launch, so for K steps we launch K kernels, and each launch
reads the states from slow global memory and writes them back.

usage:  python src/gpu_vectorized.py N K [--reps R] [--save f]
        python src/gpu_vectorized.py --sweep
"""
import sys
import time

import numpy as np
from numba import cuda

from cartpole import build_sim, INIT_STEP
from common import (parse_args, median_ms, print_result, print_report, gpu_name,
                    SWEEP_NS, SWEEP_KS)

# The same simulator as the CPU baseline, compiled for the GPU.
sim = build_sim(cuda.jit(device=True))
step_game = sim.step_game
initial_state = sim.initial_state

THREADS = 256                                   # threads per block


@cuda.jit
def init_kernel(states, seed):
    """Write the N initial states (same values the CPU baseline uses)."""
    i = cuda.grid(1)                            # this thread's game
    if i < states.shape[0]:
        x, v, th, w = initial_state(seed, i, INIT_STEP)
        states[i, 0] = x
        states[i, 1] = v
        states[i, 2] = th
        states[i, 3] = w


@cuda.jit
def step_kernel(states, t, seed):
    """ONE step for every game."""
    i = cuda.grid(1)                            # this thread's game
    if i < states.shape[0]:
        x = states[i, 0]                        # read state from global memory
        v = states[i, 1]
        th = states[i, 2]
        w = states[i, 3]
        x, v, th, w, done = step_game(x, v, th, w, i, t, seed)   # advance one tick
        states[i, 0] = x                        # write it back to global memory
        states[i, 1] = v
        states[i, 2] = th
        states[i, 3] = w


def run(N, K, reps, seed, save=None, report=False):
    d_states = cuda.device_array((N, 4), dtype=np.float32)
    blocks = (N + THREADS - 1) // THREADS
    init = init_kernel[blocks, THREADS]
    step = step_kernel[blocks, THREADS]         # prepare the launch configuration once

    def collect():                              # K steps = K launches, back to back
        for t in range(K):
            step(d_states, t, seed)

    init(d_states, seed)                        # warm-up (compiles the kernels), not timed
    init_rows = d_states[:3].copy_to_host()     # first initial states, for the report
    collect()
    cuda.synchronize()

    times = []
    for _ in range(reps):
        init(d_states, seed)
        cuda.synchronize()
        t0 = time.perf_counter()
        collect()                               # the timed part: all K steps
        cuda.synchronize()                      # wait until the GPU is done
        times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    final = d_states.copy_to_host()             # copy the final states back (once, at the end)
    copy_s = time.perf_counter() - t0
    print_result("gpu_vectorized", N, K, reps, median_ms(times), final, save)
    if report:
        print_report("gpu_vectorized", N, K, reps, seed, times, init_rows, final, launches=K,
                     copy_back_s=copy_s, launch_config=(blocks, THREADS), device=gpu_name())


if __name__ == "__main__":
    if not cuda.is_available():
        sys.exit("No CUDA GPU found by Numba. See README (Setup).")
    a = parse_args(default_reps=10)
    if a.sweep:
        for K in SWEEP_KS:
            for N in SWEEP_NS:
                run(N, K, a.reps, a.seed)
    else:
        run(a.N, a.K, a.reps, a.seed, a.save, a.report)
