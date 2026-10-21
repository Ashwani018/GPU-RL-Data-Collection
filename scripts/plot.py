"""Make the result plots from results/timing.csv.

usage: python scripts/plot.py            the bar chart and the printed numbers use the LARGEST
                                          configuration measured (max N x K; GPU only if the CPU was
                                          not run there) and, as a second chart, the largest one that
                                          includes the CPU
       python scripts/plot.py N K        use one chosen configuration instead
Needs: matplotlib. Writes PNG files into results/.
"""
import csv
import math
import sys
import matplotlib.pyplot as plt

APPROACHES = ["cpu", "gpu_vectorized", "gpu_fused"]
LABELS = {"cpu": "CPU (1 core)", "gpu_vectorized": "GPU vectorized (K launches)",
          "gpu_fused": "GPU fused (1 launch)"}
COLORS = {"cpu": "tab:gray", "gpu_vectorized": "tab:blue", "gpu_fused": "tab:red"}

rows = {}
for r in csv.DictReader(open("results/timing.csv")):
    rows[(r["approach"], int(r["N"]), int(r["K"]))] = r
Ns = sorted({k[1] for k in rows})
Ks = sorted({k[2] for k in rows})


def ms(a, N, K):
    r = rows.get((a, N, K))
    return float(r["median_ms"]) if r else None


def tput(a, N, K):
    r = rows.get((a, N, K))
    return float(r["samples_per_sec"]) if r else None


# ---- correctness: vectorized and fused must give bit-identical final states ----
bad = [(N, K) for (a, N, K) in rows if a == "gpu_fused" and ("gpu_vectorized", N, K) in rows
       and rows[("gpu_fused", N, K)]["state_hash"] != rows[("gpu_vectorized", N, K)]["state_hash"]]
total = sum(1 for (a, N, K) in rows if a == "gpu_fused" and ("gpu_vectorized", N, K) in rows)
print(f"Correctness: vectorized == fused (bit for bit) in {total - len(bad)} of {total} configs")
if bad:
    print("  MISMATCH at (N, K):", bad)

# ---- Plot 1: throughput vs N, one panel per K (paper Fig. 4 style) ----
fig, axes = plt.subplots(1, len(Ks), figsize=(4 * len(Ks), 3.8), sharey=True)
for ax, K in zip(axes, Ks):
    for a in APPROACHES:
        pts = [(N, tput(a, N, K)) for N in Ns if tput(a, N, K)]
        if pts:
            ax.plot(*zip(*pts), marker="o", color=COLORS[a], label=LABELS[a])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(f"K = {K}")
    ax.set_xlabel("Parallel games (N)")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("Throughput (samples / second)")
axes[0].legend(fontsize=7)
fig.suptitle("Data-collection throughput vs number of games")
fig.tight_layout()
fig.savefig("results/plot1_throughput_vs_N.png", dpi=130)

# ---- which configurations to show: the largest measured, and the largest with a CPU time ----
def pick_configs():
    gpu = [(k[1], k[2]) for k in rows if k[0] == "gpu_fused"]
    cpu = [(k[1], k[2]) for k in rows if k[0] == "cpu"]
    by_size = lambda c: (c[0] * c[1], c[0])
    return (max(gpu, key=by_size) if gpu else None), (max(cpu, key=by_size) if cpu else None)


if len(sys.argv) > 2:
    big, with_cpu = (int(sys.argv[1]), int(sys.argv[2])), None
else:
    big, with_cpu = pick_configs()
N0, K0 = big


# ---- Plot 2: time per approach for one configuration (bar chart) ----
def bars(N, K, filename):
    have = [(a, ms(a, N, K)) for a in APPROACHES if ms(a, N, K)]
    missing = [a for a in APPROACHES if not ms(a, N, K)]
    if missing:
        print(f"NOTE: N={N:,}, K={K:,}: no timing for {missing} "
              "(the CPU is only run up to 262,144 games, pure Python is too slow beyond that).")
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars_ = ax.bar([LABELS[a] for a, _ in have], [t for _, t in have],
                   color=[COLORS[a] for a, _ in have])
    if "cpu" in missing:                       # two GPU bars: a plain axis starting at 0 is clearer
        ax.set_ylabel("Time (ms)")
        ax.set_ylim(0, max(t for _, t in have) * 1.15)
        ax.text(0.98, 0.95, "CPU not run at this size", transform=ax.transAxes, ha="right",
                va="top", fontsize=8)
    else:                                      # CPU and GPU differ by orders of magnitude: log axis
        ax.set_yscale("log")
        ax.set_ylabel("Time (ms, log scale)")
    ax.set_title(f"Time to collect {N * K:,} samples (N={N:,}, K={K:,})", fontsize=10)
    for b, (_, t) in zip(bars_, have):
        ax.text(b.get_x() + b.get_width() / 2, t, f"{t:.3g} ms", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)


bars(N0, K0, "results/plot2_time_bars.png")
if with_cpu and with_cpu != big:
    bars(*with_cpu, "results/plot2b_time_bars_with_cpu.png")

# ---- Plot 3: fusion speedup at the same K = vectorized time / fused time ----
fig, ax = plt.subplots(figsize=(6, 4))
for K in Ks:
    pts = [(N, ms("gpu_vectorized", N, K) / ms("gpu_fused", N, K)) for N in Ns
           if ms("gpu_vectorized", N, K) and ms("gpu_fused", N, K)]
    if pts:
        ax.plot(*zip(*pts), marker="o", label=f"K = {K}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Parallel games (N)")
ax.set_ylabel("Fusion speedup (x)")
ax.set_title("Fusion speedup at equal K (K launches vs 1 launch)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/plot3_fusion_speedup.png", dpi=130)

# ---- Plot 4: throughput vs K at a fixed N, GPU approaches (paper Fig. 8 style) ----
fig, ax = plt.subplots(figsize=(6, 4))
for a in ["gpu_vectorized", "gpu_fused"]:
    pts = [(K, tput(a, N0, K)) for K in Ks if tput(a, N0, K)]
    if pts:
        ax.plot(*zip(*pts), marker="o", color=COLORS[a], label=LABELS[a])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Steps per iteration (K)")
ax.set_ylabel("Throughput (samples / second)")
ax.set_title(f"Effect of K (N = {N0})")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/plot4_throughput_vs_K.png", dpi=130)

# ---- Numbers for the report ----
def numbers(N, K):
    print(f"\nConfiguration N={N:,}, K={K:,}:")
    t = {a: ms(a, N, K) for a in APPROACHES}
    for a in APPROACHES:
        print(f"  {LABELS[a]:30s} {t[a]:12.4f} ms" if t[a] else f"  {LABELS[a]:30s}      (not run)")
    if t["cpu"] and t["gpu_vectorized"]:
        print(f"  GPU vectorized vs CPU : {t['cpu'] / t['gpu_vectorized']:8.1f}x")
    if t["cpu"] and t["gpu_fused"]:
        print(f"  GPU fused vs CPU      : {t['cpu'] / t['gpu_fused']:8.1f}x")
    if t["gpu_vectorized"] and t["gpu_fused"]:
        print(f"  Fusion speedup (same K): {t['gpu_vectorized'] / t['gpu_fused']:7.1f}x")
    f1, fK = tput("gpu_fused", N, min(Ks)), tput("gpu_fused", N, max(Ks))
    if f1 and fK:
        print(f"  Paper-style fusion speedup, fused K={max(Ks)} vs K={min(Ks)}: {fK / f1:.1f}x "
              f"(paper: 11.3x on CartPole)")


numbers(N0, K0)
if with_cpu and with_cpu != big:
    numbers(*with_cpu)
print("\nSaved plots to results/plot*.png")
