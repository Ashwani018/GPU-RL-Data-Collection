"""Run every experiment and write results/timing.csv (one line per run).

usage (laptop plugged in, high-performance power mode):  python src/run_sweeps.py

Sweep: N = 2^0 .. 2^20 (steps of x4), K in {1, 10, 100, 1000}, as in the paper.
The pure-Python CPU baseline runs only up to N = 262,144 games (see common.py), so the
whole sweep takes roughly 45-60 minutes, almost all of it the CPU; the GPU part takes minutes.
At the end it also writes the sweep report (results/report_all.md, summary_all.csv).
"""
import os
import subprocess
import sys

from common import CSV_HEADER

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "timing.csv")


def script(name):
    return [sys.executable, os.path.join(HERE, name)]


with open(OUT, "w") as out:
    out.write(CSV_HEADER + "\n")
    for name in ("gpu_vectorized.py", "gpu_fused.py", "cpu_baseline.py"):
        print(f"--- {name} --sweep", flush=True)
        proc = subprocess.Popen(script(name) + ["--sweep"], stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:                 # show progress and save it
            print(line, end="", flush=True)
            out.write(line)
            out.flush()
        if proc.wait() != 0:
            sys.exit(f"{name} failed")

# Extra files: final states (for the CPU-vs-GPU check) and trajectories (for the animation).
print("--- extra files", flush=True)
extras = [
    ("cpu_baseline.py", ["--save", os.path.join(RESULTS, "final_cpu.bin"),
                         "--traj", os.path.join(RESULTS, "trajectory.csv")]),
    ("gpu_vectorized.py", ["--save", os.path.join(RESULTS, "final_vectorized.bin")]),
    ("gpu_fused.py", ["--save", os.path.join(RESULTS, "final_fused.bin")]),
]
for name, extra in extras:
    subprocess.run(script(name) + ["1024", "10", "--reps", "1"] + extra,
                   stdout=subprocess.DEVNULL, check=True)
# Report for the whole sweep: one table per K plus the key findings.
print("--- sweep report (results/report_all.md)", flush=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "scripts", "report.py"), "--all"],
               cwd=os.path.dirname(HERE), check=True)
print("Done. Now run:  python scripts/compare.py  and  python scripts/plot.py")
