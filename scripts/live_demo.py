"""Live demo: watch each approach simulate CartPole, one approach after the other.

usage in a notebook cell:   %run scripts/live_demo.py
usage as a script:          python scripts/live_demo.py            (opens one window per approach)
options:  --games 256 --frames 100 --steps-per-frame 4 --only cpu|vectorized|fused
          --save-prefix demo   (also saves the last frame of each approach as demo_<name>.png)

The approaches run in turn: CPU, then GPU vectorized, then GPU fused. Each one gets its own
window (in a notebook: its own output block, which stays on screen when it finishes). The
robot cart slides left/right and its pole swings while that approach really computes:
  CPU             - a Python loop over the games
  GPU vectorized  - one kernel launch per step
  GPU fused       - one kernel launch for all the steps of a frame
Each approach simulates --games games (default 256); the first game is drawn. When its pole
leans past 12 degrees (or the cart leaves the track) the game restarts, as in the benchmark.

This is a DEMO, separate from the benchmark. Drawing and copying the state back are far more
expensive than the simulation, so every approach looks equally fast. What differs is the
"compute time" counter: it adds up ONLY the time of the computation itself (the timer wraps
the compute call, not the copy-back or the drawing). The real benchmark numbers come from the
sweeps (src/run_sweeps.py), not from this demo.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from numba import cuda

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cpu_baseline
import gpu_vectorized
import gpu_fused

p = argparse.ArgumentParser()
p.add_argument("--games", type=int, default=1000, help="games simulated by each approach")
p.add_argument("--frames", type=int, default=100, help="animation frames per approach")
p.add_argument("--steps-per-frame", type=int, default=4, help="simulation steps between two frames")
p.add_argument("--only", choices=["cpu", "vectorized", "fused"], help="run just one approach")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--save-prefix", help="save the last frame of each approach to <prefix>_<name>.png")
a = p.parse_args()
N, S, SEED = a.games, a.steps_per_frame, a.seed

if not cuda.is_available():
    sys.exit("No CUDA GPU found by Numba (the GPU approaches need one). See README (Setup).")

try:                                        # inside a notebook: update one output block per approach
    from IPython import get_ipython
    from IPython.display import display
    IN_NOTEBOOK = get_ipython() is not None
except ImportError:
    IN_NOTEBOOK = False
WINDOWS = (not IN_NOTEBOOK) and matplotlib.get_backend().lower() not in (
    "agg", "pdf", "svg", "ps", "cairo", "template")             # a real GUI is available

THREADS = gpu_vectorized.THREADS
blocks = (N + THREADS - 1) // THREADS
step_game = gpu_fused.step_game             # the same device function the real kernels use


@cuda.jit
def fused_chunk_kernel(states, t0, steps, seed):
    """Demo copy of gpu_fused.fused_kernel that starts at step t0, so the animation can
    continue over many frames. State is read once, `steps` steps run in registers, written once."""
    i = cuda.grid(1)
    if i < states.shape[0]:
        x = states[i, 0]
        v = states[i, 1]
        th = states[i, 2]
        w = states[i, 3]
        for t in range(t0, t0 + steps):
            x, v, th, w, done = step_game(x, v, th, w, i, t, seed)
        states[i, 0] = x
        states[i, 1] = v
        states[i, 2] = th
        states[i, 3] = w


# ---------------------------------------------------------------- the three approaches
cpu_states = cpu_baseline.initial_states(N, SEED)
d_vec = cuda.device_array((N, 4), dtype=np.float32)
d_fus = cuda.device_array((N, 4), dtype=np.float32)
gpu_vectorized.init_kernel[blocks, THREADS](d_vec, SEED)
gpu_fused.init_kernel[blocks, THREADS](d_fus, SEED)
vec_launch = gpu_vectorized.step_kernel[blocks, THREADS]
fused_launch = fused_chunk_kernel[blocks, THREADS]


def advance_cpu(t0):                        # games one after another, step by step
    for t in range(t0, t0 + S):
        for i in range(N):
            x, v, th, w, _ = cpu_baseline.step_game(cpu_states[i, 0], cpu_states[i, 1],
                                                    cpu_states[i, 2], cpu_states[i, 3], i, t, SEED)
            cpu_states[i] = (x, v, th, w)


def advance_vectorized(t0):                 # one launch per step
    for t in range(t0, t0 + S):
        vec_launch(d_vec, t, SEED)
    cuda.synchronize()


def advance_fused(t0):                      # one launch for all steps of the frame
    fused_launch(d_fus, t0, S, SEED)
    cuda.synchronize()


# warm-up on a scratch array so kernel compilation is not counted in the first frame
scratch = cuda.device_array((N, 4), dtype=np.float32)
gpu_vectorized.init_kernel[blocks, THREADS](scratch, SEED)
vec_launch(scratch, 0, SEED)
fused_launch(scratch, 0, S, SEED)
cuda.synchronize()

APPROACHES = [
    dict(key="cpu", name="CPU (1 core, plain Python)", color="#7f7f7f", advance=advance_cpu,
         read=lambda: cpu_states[0]),
    dict(key="vectorized", name="GPU vectorized (one kernel launch per step)", color="#1f77b4",
         advance=advance_vectorized, read=lambda: d_vec[0:1].copy_to_host()[0]),
    dict(key="fused", name="GPU fused (one kernel launch per frame)", color="#d62728",
         advance=advance_fused, read=lambda: d_fus[0:1].copy_to_host()[0]),
]
if a.only:
    APPROACHES = [ap for ap in APPROACHES if ap["key"] == a.only]


# ---------------------------------------------------------------- the robot cart
POLE_LEN, BODY_Y = 1.15, 0.46               # pole length and the height of the pole's base


def make_window(ap):
    """Build the figure for one approach: sky, track, robot cart with a pole, text."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor("#eef4fb")
    ax.set_facecolor("#eef4fb")
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-0.55, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-2.9, -0.55), 5.8, 0.5, color="#c9d3dd"))            # ground
    ax.add_patch(Rectangle((-2.4, -0.05), 4.8, 0.05, color="#34495e"))           # track
    for end in (-2.4, 2.4):                                                      # end stops
        ax.add_patch(Rectangle((end - 0.04, -0.05), 0.08, 0.35, color="#c0392b"))
    for sign in (-1, 1):                                                         # +-12 degree limits
        lim = 12 * math.pi / 180
        ax.plot([0, sign * POLE_LEN * 1.15 * math.sin(lim)],
                [BODY_Y, BODY_Y + POLE_LEN * 1.15 * math.cos(lim)],
                color="#c0392b", lw=0.8, ls=(0, (4, 4)), alpha=0.35)
    art = {}
    art["pole"], = ax.plot([], [], color="#5d4037", lw=7, solid_capstyle="round", zorder=3)
    art["ball"] = Circle((0, 0), 0.09, color="#e67e22", zorder=4)
    ax.add_patch(art["ball"])
    art["body"] = FancyBboxPatch((-0.3, 0.16), 0.6, 0.32, boxstyle="round,pad=0,rounding_size=0.07",
                                 fc=ap["color"], ec="#222222", lw=1.5, zorder=5)
    ax.add_patch(art["body"])
    art["wheels"] = [Circle((0, 0.09), 0.09, fc="#222222", ec="#888888", lw=2, zorder=6) for _ in range(2)]
    art["eye"] = Circle((0, 0), 0.075, fc="white", ec="#222222", lw=1, zorder=7)
    art["pupil"] = Circle((0, 0), 0.032, fc="#222222", zorder=8)
    art["hub"] = Circle((0, BODY_Y), 0.045, fc="#bdc3c7", ec="#222222", zorder=9)
    for patch in art["wheels"] + [art["eye"], art["pupil"], art["hub"]]:
        ax.add_patch(patch)
    ax.text(-2.85, 1.98, ap["name"], fontsize=14, fontweight="bold", va="top")
    art["info"] = ax.text(-2.85, 1.72, "", fontsize=11, va="top", family="monospace")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, art


def draw_frame(art, state, steps_done, compute_s, status):
    x, th = float(state[0]), float(state[2])
    art["body"].set_bounds(x - 0.3, 0.16, 0.6, 0.32)
    for wheel, dx in zip(art["wheels"], (-0.18, 0.18)):
        wheel.center = (x + dx, 0.09)
    art["eye"].center = (x + 0.14, 0.34)
    art["pupil"].center = (x + 0.155, 0.34)
    art["hub"].center = (x, BODY_Y)
    tip = (x + POLE_LEN * math.sin(th), BODY_Y + POLE_LEN * math.cos(th))
    art["pole"].set_data([x, tip[0]], [BODY_Y, tip[1]])
    art["ball"].center = tip
    art["info"].set_text(f"steps          {steps_done:>10,}\n"
                         f"samples        {steps_done * N:>10,}   ({N:,} games)\n"
                         f"compute time   {compute_s * 1000:>10.2f} ms   {status}")


# ---------------------------------------------------------------- run: one approach after the other
results = []
for ap in APPROACHES:
    fig, art = make_window(ap)
    draw_frame(art, ap["read"](), 0, 0.0, "starting")
    if IN_NOTEBOOK:
        handle = display(fig, display_id=True)      # this approach's own output block
    elif WINDOWS:
        plt.ion()
        plt.show(block=False)
    compute_s = 0.0
    for frame in range(a.frames):
        c0 = time.perf_counter()
        ap["advance"](frame * S)                                        # the computation only
        compute_s += time.perf_counter() - c0
        state = ap["read"]()                                            # copy back: not timed
        draw_frame(art, state, (frame + 1) * S, compute_s, "running" if frame + 1 < a.frames else "DONE")
        if IN_NOTEBOOK:
            handle.update(fig)
        elif WINDOWS:
            plt.pause(0.001)
    if a.save_prefix:
        fig.savefig(f"{a.save_prefix}_{ap['key']}.png", dpi=110)
    results.append((ap["name"], compute_s))
    if WINDOWS:
        plt.pause(1.0)                       # keep this approach's window up a moment
    if not WINDOWS:
        plt.close(fig)                       # notebook: the last frame stays in its output block

if WINDOWS:
    print("Close the windows to finish.")
    plt.ioff()
    plt.show()

total_steps = a.frames * S
print(f"\nDemo compute-only times ({N} games, {total_steps} steps; copy-back and drawing excluded):")
for name, secs in results:
    print(f"  {name:48s} {secs * 1000:10.2f} ms   ({secs / total_steps * 1e6:9.1f} microseconds per step)")
print("These are demo-size numbers. The benchmark numbers come from the sweeps (src/run_sweeps.py).")
