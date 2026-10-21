"""Animated "race": the three approaches collecting data, side by side.

usage: python scripts/visualize.py [N K] [--with-cpu] [--gif]
         (no N K)     the LARGEST configuration measured (max N x K); the CPU panel is
                      missing there if the CPU was not run (pure Python is too slow)
         --with-cpu   use the largest configuration that includes the CPU instead
         N K          a chosen configuration (must exist in results/timing.csv)
         --gif        save results/race.gif (or race_with_cpu.gif) instead of opening a window

How it works (kept simple, and honest):
 * The time of each approach is the REAL measured median time from timing.csv.
 * The animation is a slow-motion REPLAY: the slowest approach takes DURATION
   seconds on screen, the others take proportionally less.
 * The cart shown is one recorded game (results/trajectory.csv). All three
   approaches simulate the same game with the same math, so they would show the
   same motion; each panel plays it at its own speed. The sample counter and the
   progress bar show how much of the N x K samples is collected at that moment.
Needs: matplotlib.
"""
import csv
import math
import sys
import matplotlib

GIF = "--gif" in sys.argv
if GIF:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DURATION, FPS = 8.0, 25           # seconds the slowest approach takes on screen, frames/second

NAMES = [("cpu", "CPU (1 core)", "tab:gray"),
         ("gpu_vectorized", "GPU vectorized", "tab:blue"),
         ("gpu_fused", "GPU fused", "tab:red")]

# --- measured times (all configurations), then choose the configuration to show ---
all_ms = {}
for r in csv.DictReader(open("results/timing.csv")):
    all_ms[(r["approach"], int(r["N"]), int(r["K"]))] = float(r["median_ms"])
by_size = lambda c: (c[0] * c[1], c[0])
WITH_CPU = "--with-cpu" in sys.argv
if len(args) >= 2:
    N, K = int(args[0]), int(args[1])
elif WITH_CPU:
    N, K = max(((k[1], k[2]) for k in all_ms if k[0] == "cpu"), key=by_size)
else:
    N, K = max(((k[1], k[2]) for k in all_ms if k[0] == "gpu_fused"), key=by_size)
time_ms = {a: all_ms[(a, N, K)] for a, _, _ in NAMES if (a, N, K) in all_ms}
shown = [n for n in NAMES if n[0] in time_ms]
if not shown:
    sys.exit(f"No timing for N={N}, K={K} in results/timing.csv")
if len(shown) < 3:
    print("NOTE: missing approaches for this configuration:",
          [n[0] for n in NAMES if n[0] not in time_ms], "(use --with-cpu for a configuration with the CPU)")
slowest = max(time_ms[n[0]] for n in shown)

# --- recorded trajectory of game 0 ---
traj = [(float(r["x"]), float(r["theta"])) for r in csv.DictReader(open("results/trajectory.csv"))
        if r["game"] == "0"]
STEPS = len(traj)

fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.3), squeeze=False)
axes = axes[0]
POLE = 0.7
panels = []
for ax, (key, label, color) in zip(axes, shown):
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-0.7, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([-2.4, 2.4], [0, 0], color="black", lw=1)                    # track
    ax.plot([-2.4, -2.4], [-0.05, 0.05], color="black")
    ax.plot([2.4, 2.4], [-0.05, 0.05], color="black")
    cart = Rectangle((-0.2, 0), 0.4, 0.25, color=color)
    ax.add_patch(cart)
    (pole,) = ax.plot([], [], color="saddlebrown", lw=4)
    ax.add_patch(Rectangle((-2.4, -0.55), 4.8, 0.1, fill=False, ec="black"))  # progress bar frame
    bar = Rectangle((-2.4, -0.55), 0.0, 0.1, color=color)
    ax.add_patch(bar)
    ax.set_title(f"{label}\nreal time: {time_ms[key]:.3g} ms", fontsize=11)
    txt = ax.text(0, 1.2, "", ha="center", fontsize=9)
    panels.append((key, cart, pole, bar, txt))

fig.suptitle(f"Collecting N x K = {N * K:,} samples (N = {N:,} games, K = {K} steps)  -  "
             f"slow-motion replay of measured times", fontsize=10)
fig.tight_layout()


def draw(frame):
    now = frame / FPS
    for key, cart, pole, bar, txt in panels:
        seconds_on_screen = max(DURATION * time_ms[key] / slowest, 1e-9)
        progress = min(1.0, now / seconds_on_screen)
        x, theta = traj[min(STEPS - 1, int(progress * (STEPS - 1)))]
        cart.set_x(x - 0.2)
        pole.set_data([x, x + POLE * math.sin(theta)], [0.25, 0.25 + POLE * math.cos(theta)])
        bar.set_width(4.8 * progress)
        done = "  DONE" if progress >= 1.0 else ""
        txt.set_text(f"{progress * N * K:,.0f} samples{done}")
    return []


frames = int((DURATION + 1.5) * FPS)
ani = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS, blit=False)
if GIF:
    out = "results/race_with_cpu.gif" if WITH_CPU else "results/race.gif"
    ani.save(out, writer=PillowWriter(fps=FPS))
    print(f"Saved {out}  (N={N:,}, K={K:,})")
else:
    plt.show()
