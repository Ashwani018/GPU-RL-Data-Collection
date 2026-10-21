"""gpu_fused.py - APPROACH 2: kernel fusion (K steps inside ONE kernel launch).

Same simulator, same one-thread-per-game layout as gpu_vectorized.py. The only
change: the loop over K steps moves INSIDE the kernel. Each thread reads its
state once into registers (tiny, very fast), runs K steps there, and writes the
result once. One launch instead of K, and global memory is touched twice
instead of 2*K times.

usage:  python src/gpu_fused.py N K [--reps R] [--save f]
        python src/gpu_fused.py --sweep
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
def fused_kernel(states, K, seed):
    """K steps for every game, in a single launch."""
    i = cuda.grid(1)                            # this thread's game
    if i < states.shape[0]:
        x = states[i, 0]                        # read state ONCE (into registers)
        v = states[i, 1]
        th = states[i, 2]
        w = states[i, 3]
        for t in range(K):                      # all K steps stay in registers
            x, v, th, w, done = step_game(x, v, th, w, i, t, seed)
        states[i, 0] = x                        # write state ONCE
        states[i, 1] = v
        states[i, 2] = th
        states[i, 3] = w


def run(N, K, reps, seed, save=None, report=False):
    d_states = cuda.device_array((N, 4), dtype=np.float32)
    blocks = (N + THREADS - 1) // THREADS
    init = init_kernel[blocks, THREADS]
    fused = fused_kernel[blocks, THREADS]       # prepare the launch configuration once

    init(d_states, seed)                        # warm-up (compiles the kernels), not timed
    init_rows = d_states[:3].copy_to_host()     # first initial states, for the report
    fused(d_states, K, seed)
    cuda.synchronize()

    times = []
    for _ in range(reps):
        init(d_states, seed)
        cuda.synchronize()
        t0 = time.perf_counter()
        fused(d_states, K, seed)                # the timed part: ONE launch for all K steps
        cuda.synchronize()                      # wait until the GPU is done
        times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    final = d_states.copy_to_host()             # copy the final states back (once, at the end)
    copy_s = time.perf_counter() - t0
    print_result("gpu_fused", N, K, reps, median_ms(times), final, save)
    if report:
        print_report("gpu_fused", N, K, reps, seed, times, init_rows, final, launches=1,
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
