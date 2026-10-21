"""Comparison reports of the three approaches, built from results/timing.csv.

usage: python scripts/report.py            tables for the LARGEST configuration measured
                                           (max N x K, GPU only if the CPU was not run there)
                                           and for the largest one that includes the CPU
       python scripts/report.py N K        table for ONE chosen configuration
       python scripts/report.py --all      report for the WHOLE sweep (all N and K)

One configuration: prints a table and saves results/report_N<N>_K<K>.md.
Whole sweep:       one table per K (a row per N) plus a "key findings" block; saves
                   results/report_all.md and results/summary_all.csv.
If the CPU baseline was not run for a configuration (N*K too large for pure
Python) it shows "not run" and the CPU speedups are left out.
"""
import csv
import sys

NAMES = {"cpu": "CPU (1 core)", "gpu_vectorized": "GPU vectorized", "gpu_fused": "GPU fused"}

rows = {}
for r in csv.DictReader(open("results/timing.csv")):
    rows[(r["approach"], int(r["N"]), int(r["K"]))] = r
if not rows:
    sys.exit("results/timing.csv is empty")


def ms(a, N, K):
    r = rows.get((a, N, K))
    return float(r["median_ms"]) if r else None


def tput(a, N, K):
    r = rows.get((a, N, K))
    return float(r["samples_per_sec"]) if r else None


def pick_configs():
    """(largest configuration measured, largest configuration that has a CPU time)."""
    gpu = [(k[1], k[2]) for k in rows if k[0] == "gpu_fused"]
    cpu = [(k[1], k[2]) for k in rows if k[0] == "cpu"]
    by_size = lambda c: (c[0] * c[1], c[0])
    return (max(gpu, key=by_size) if gpu else None), (max(cpu, key=by_size) if cpu else None)


def sp(x):
    """Format a speedup; values below 1 keep enough digits to be readable."""
    return f"{x:,.1f}x" if x >= 1 else f"{x:.4f}x"


def md_table(head, body):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in body]


# ============================================================ one configuration
def one_config(N, K):
    if not any(k[1] == N and k[2] == K for k in rows):
        sys.exit(f"No results for N={N}, K={K} in results/timing.csv")
    launches = {"cpu": 0, "gpu_vectorized": K, "gpu_fused": 1}
    t = {a: ms(a, N, K) for a in NAMES if ms(a, N, K)}
    head = ["Approach", "Iterations (K)", "Samples (N x K)", "Kernel launches",
            "Execution time (ms)", "Throughput (samples/s)", "Speedup vs CPU",
            "Speedup vs GPU vectorized", "Final-state hash"]
    body = []
    for a in NAMES:
        if a not in t:
            body.append([NAMES[a], f"{K:,}", f"{N * K:,}", f"{launches[a]:,}", "not run", "-", "-", "-", "-"])
            continue
        vs_cpu = sp(t["cpu"] / t[a]) if "cpu" in t else "-"
        vs_vec = sp(t["gpu_vectorized"] / t[a]) if "gpu_vectorized" in t else "-"
        body.append([NAMES[a], f"{K:,}", f"{N * K:,}", f"{launches[a]:,}", f"{t[a]:,.3f}",
                     f"{tput(a, N, K):.3e}", vs_cpu, vs_vec, rows[(a, N, K)]["state_hash"]])
    lines = [f"## Comparison of the approaches (N = {N:,} games, K = {K:,} steps)", ""]
    lines += md_table(head, body)
    lines += ["", "Execution time = time to collect all N x K samples (median over the repetitions, "
                  "state copy-back to the CPU not included)."]
    lines.append("Repetitions timed: " + ", ".join(
        f"{NAMES[a]} {rows[(a, N, K)]['reps']}" for a in NAMES if (a, N, K) in rows) +
        " (a single repetition is the one measured time, not a median).")
    if "gpu_vectorized" in t and "gpu_fused" in t:
        same = rows[("gpu_vectorized", N, K)]["state_hash"] == rows[("gpu_fused", N, K)]["state_hash"]
        lines.append(f"Correctness: GPU vectorized and GPU fused final states bit-identical: "
                     f"{'YES' if same else 'NO (!)'}.")
        lines.append(f"Fusion speedup at the same K (vectorized time / fused time): "
                     f"{t['gpu_vectorized'] / t['gpu_fused']:.1f}x.")
    if "cpu" in t:
        lines.append("The CPU hash can differ from the GPU hash because the CPU and GPU math libraries "
                     "round sin/cos slightly differently; use scripts/compare.py for the CPU-vs-GPU check.")
    else:
        _, cpu_cfg = pick_configs()
        if cpu_cfg:
            cn, ck = cpu_cfg
            rate_us = ms("cpu", cn, ck) * 1000.0 / (cn * ck)
            hours = N * K * rate_us * 1e-6 / 3600.0
            lines.append(f"The CPU was not run at this size. At the CPU's measured {rate_us:.1f} microseconds "
                         f"per game-step (largest CPU run: N = {cn:,}, K = {ck:,}), pure Python would take "
                         f"roughly {hours:.1f} hours here (an estimate, not a measurement).")
    lines.append(f"Fastest approach: {NAMES[min(t, key=t.get)]}.")
    text = "\n".join(lines)
    print(text)
    out = f"results/report_N{N}_K{K}.md"
    open(out, "w").write(text + "\n")
    print(f"\nSaved {out}")


# ============================================================ whole sweep
def whole_sweep():
    Ns = sorted({k[1] for k in rows})
    Ks = sorted({k[2] for k in rows})
    out = ["# Sweep report: CPU vs GPU vectorized vs GPU fused", "",
           f"Configurations: N from {min(Ns):,} to {max(Ns):,} games, K in {Ks} steps per iteration. "
           "Times are the median time (ms) to collect all N x K samples. "
           "Fusion speedup = vectorized time / fused time at the same K. "
           "'-' = the CPU baseline was not run (too many games for pure Python). "
           "CPU runs with N x K above 2,000,000 were timed once; all other timings are the median of repetitions.", ""]
    csv_rows = [["N", "K", "samples", "cpu_ms", "vectorized_ms", "fused_ms", "fused_vs_vectorized",
                 "vectorized_vs_cpu", "fused_vs_cpu", "fused_samples_per_sec", "vec_equals_fused"]]
    mismatches = checked = 0

    for K in Ks:
        head = ["N (games)", "CPU (ms)", "GPU vectorized (ms)", "GPU fused (ms)",
                "Fusion speedup (vec/fused)", "Vectorized vs CPU", "Fused vs CPU",
                "Fused throughput (samples/s)", "Vec = fused?"]
        body = []
        for N in Ns:
            c, v, f = ms("cpu", N, K), ms("gpu_vectorized", N, K), ms("gpu_fused", N, K)
            if v is None or f is None:
                continue
            same = rows[("gpu_vectorized", N, K)]["state_hash"] == rows[("gpu_fused", N, K)]["state_hash"]
            checked += 1
            mismatches += 0 if same else 1
            body.append([f"{N:,}", f"{c:,.3f}" if c else "-", f"{v:,.3f}", f"{f:,.3f}", sp(v / f),
                         sp(c / v) if c else "-", sp(c / f) if c else "-",
                         f"{tput('gpu_fused', N, K):.3e}", "yes" if same else "NO"])
            csv_rows.append([N, K, N * K, f"{c:.5f}" if c else "", f"{v:.5f}", f"{f:.5f}",
                             f"{v / f:.3f}", f"{c / v:.4f}" if c else "", f"{c / f:.3f}" if c else "",
                             f"{tput('gpu_fused', N, K):.6e}", int(same)])
        out += [f"## K = {K:,} steps per iteration", ""] + md_table(head, body) + [""]

    # ---------------- key findings (everything computed from the data) ----------------
    f_ = ["## Key findings (computed from the table data)", ""]
    f_.append(f"- Correctness: GPU vectorized and GPU fused final states were bit-identical in "
              f"{checked - mismatches} of {checked} configurations.")
    for a in NAMES:
        best = max(((tput(a, N, K), N, K) for (x, N, K) in rows if x == a), default=None)
        if best:
            f_.append(f"- Peak throughput, {NAMES[a]}: {best[0]:.3e} samples/s (N = {best[1]:,}, K = {best[2]:,}).")
    fused_best = max(((tput("gpu_fused", N, K), N, K) for (x, N, K) in rows if x == "gpu_fused"), default=None)
    cpu_best = max(((tput("cpu", N, K), N, K) for (x, N, K) in rows if x == "cpu"), default=None)
    if fused_best and cpu_best:
        f_.append(f"- Peak GPU fused throughput is {fused_best[0] / cpu_best[0]:,.0f}x the peak CPU throughput "
                  "(pure-Python CPU without inference; not comparable with the paper's 1024x).")
    f_.append("")
    f_.append("**Where the GPU first beats the CPU** (smallest N where the GPU time is below the CPU time):")
    for K in Ks:
        parts = []
        for a in ("gpu_vectorized", "gpu_fused"):
            n = next((N for N in Ns if ms("cpu", N, K) and ms(a, N, K) and ms(a, N, K) < ms("cpu", N, K)), None)
            parts.append(f"{NAMES[a]}: " + (f"N = {n:,}" if n else "not within the CPU range"))
        f_.append(f"- K = {K:,}: " + "; ".join(parts))
    f_.append("")
    f_.append("**Fusion speedup (vectorized time / fused time) per K:**")
    for K in Ks:
        s = [(ms("gpu_vectorized", N, K) / ms("gpu_fused", N, K), N) for N in Ns
             if ms("gpu_vectorized", N, K) and ms("gpu_fused", N, K)]
        if s:
            lo, hi, last = min(s), max(s), s[-1]
            f_.append(f"- K = {K:,}: from {lo[0]:.1f}x (N = {lo[1]:,}) to {hi[0]:.1f}x (N = {hi[1]:,}); "
                      f"{last[0]:.1f}x at the largest N ({last[1]:,}).")
    small = [(K, [1000.0 * ms("gpu_vectorized", N, K) / K for N in Ns if N <= 4096 and ms("gpu_vectorized", N, K)])
             for K in Ks if K >= 10]
    if small:
        f_.append("")
        f_.append("**Vectorized time per kernel launch** (vectorized time / K, for N <= 4,096):")
        for K, v in small:
            if v:
                f_.append(f"- K = {K:,}: {min(v):.1f} to {max(v):.1f} microseconds per launch.")
        f_.append("If this stays about the same while N grows, the vectorized run is limited by the cost of "
                  "launching kernels (here from Python), not by the GPU's work; fusion removes those launches.")
    out += f_

    text = "\n".join(out)
    print(text)
    open("results/report_all.md", "w").write(text + "\n")
    with open("results/summary_all.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    print("\nSaved results/report_all.md and results/summary_all.csv")


if "--all" in sys.argv:
    whole_sweep()
else:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) >= 2:
        one_config(int(args[0]), int(args[1]))
    else:
        biggest, with_cpu = pick_configs()
        one_config(*biggest)
        if with_cpu and with_cpu != biggest:
            print("\n" + "=" * 100 + "\n")
            one_config(*with_cpu)
