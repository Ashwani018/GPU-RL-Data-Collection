"""common.py - small helpers used by all three programs (no GPU code here)."""
import argparse
import hashlib
import statistics
import sys

import numpy as np

# The sweep (as in the paper): N = 2^0 .. 2^20, K in {1, 10, 100, 1000}.
SWEEP_NS = [2 ** e for e in range(0, 21, 2)]
SWEEP_KS = [1, 10, 100, 1000]
# Pure-Python CPU is slow (about 7 microseconds per game-step), so the CPU baseline runs
# only up to this many games, and runs with more than CPU_FULL_REPS_MAX_SAMPLES samples
# (N*K) are timed once instead of --reps times.
CPU_MAX_N = 262_144
CPU_FULL_REPS_MAX_SAMPLES = 2_000_000

CSV_HEADER = "approach,N,K,reps,median_ms,samples_per_sec,state_hash"


def parse_args(default_reps):
    p = argparse.ArgumentParser(description="usage: program N K   or   program --sweep")
    p.add_argument("N", type=int, nargs="?", default=1024, help="number of parallel games")
    p.add_argument("K", type=int, nargs="?", default=100, help="steps per iteration")
    p.add_argument("--sweep", action="store_true", help="run the whole N x K sweep")
    p.add_argument("--reps", type=int, default=default_reps, help="timed repetitions (median)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", help="write final states (float32 N x 4) to this file")
    p.add_argument("--traj", help="(CPU only) write a few trajectories for the animation")
    p.add_argument("--report", action="store_true", help="print a readable input/output report")
    return p.parse_args()


def median_ms(times_s):
    return statistics.median(times_s) * 1000.0


def state_hash(states):
    """Identical hash = bit-identical final states."""
    return hashlib.sha1(np.ascontiguousarray(states).tobytes()).hexdigest()[:16]


def print_result(approach, N, K, reps, ms, final_states, save=None):
    """One CSV line: approach,N,K,reps,median_ms,samples_per_sec,state_hash"""
    samples_per_sec = N * K / (ms / 1000.0)
    print(f"{approach},{N},{K},{reps},{ms:.5f},{samples_per_sec:.6e},{state_hash(final_states)}",
          flush=True)
    if save:
        np.ascontiguousarray(final_states, dtype=np.float32).tofile(save)


def gpu_name():
    from numba import cuda          # imported here so the CPU program needs no GPU library
    try:
        name = cuda.get_current_device().name
    except Exception:               # e.g. the CUDA simulator has no real device
        return "unknown GPU"
    return name.decode() if isinstance(name, bytes) else str(name)


def _rows(states):
    return "\n".join(f"      game {i}: [x, v, theta, omega] = "
                     f"[{', '.join(f'{float(v): .5f}' for v in row)}]"
                     for i, row in enumerate(states))


def print_report(approach, N, K, reps, seed, times, init_rows, final_states, launches,
                 copy_back_s=0.0, launch_config=None, device=None):
    """A readable INPUT / OUTPUT report of one run (printed to stderr, so the CSV
    line on stdout stays clean)."""
    ms = [t * 1000.0 for t in times]
    med = statistics.median(ms)
    where = f"{device}, {launch_config[0]} blocks x {launch_config[1]} threads" if device else "CPU, 1 core"
    lines = [
        f"==================== RUN REPORT: {approach} ====================",
        "INPUT",
        f"  games simulated (N)        : {N:,}",
        f"  steps per iteration (K)    : {K:,}",
        f"  seed / repetitions         : {seed} / {reps}",
        "  simulator                  : CartPole (gym v1 equations, float32)",
        "  restart rule               : a game restarts if |x| > 2.4 or |theta| > 12 degrees",
        "  actions                    : push left/right, chosen by a hash of (game, step)",
        f"  runs on                    : {where}",
        "  initial states (first games):",
        _rows(init_rows),
        "OUTPUT",
        "  final states (first games) :",
        _rows(final_states[:len(init_rows)]),
        f"  iterations (steps K)       : {K:,}",
        f"  samples collected (N x K)  : {N * K:,}",
        f"  kernel launches            : {launches:,}",
        f"  execution time (median)    : {med:.3f} ms   (min {min(ms):.3f}, max {max(ms):.3f} over {reps} runs)",
        f"  copy final states to CPU   : {copy_back_s * 1000.0:.3f} ms   (once, at the end)",
        f"  TOTAL time                 : {med + copy_back_s * 1000.0:.3f} ms",
        f"  throughput                 : {N * K / (med / 1000.0):.3e} samples/s",
        f"  final-state hash           : {state_hash(final_states)}",
    ]
    print("\n".join(lines), file=sys.stderr)
