# GPU data collection: CPU vs GPU vectorization vs kernel fusion (Python)

A small re-implementation of the two ideas in *Optimizing Data Collection in Deep
Reinforcement Learning* (Gleeson et al., MLBench 2022, arXiv 2207.07736), on the
CartPole simulator. The GPU kernels are written by hand in Python with Numba CUDA
(`@cuda.jit`); no PyTorch, JAX or XLA.

| Approach | File | What it does |
|---|---|---|
| CPU baseline | `src/cpu_baseline.py` | plain Python, one core, loop over the N games, K steps |
| GPU vectorization | `src/gpu_vectorized.py` | one thread per game, **K kernel launches** (1 step each) |
| GPU kernel fusion | `src/gpu_fused.py` | one thread per game, **1 launch** running K steps in registers |

`src/cartpole.py` is the simulator used by all three: **one source, compiled two
ways** (left as plain Python for the CPU, compiled by Numba for the GPU), so only the
scheduling differs. `src/common.py` has argument parsing and the sweep settings.

## Setup (Windows laptop with the RTX 4060)
Needs Python 3.10+ and an up-to-date NVIDIA driver.
```
pip install -r requirements.txt
python -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
```
This must print `True` and list the 4060. If it prints `False`, update the NVIDIA
driver, and try `pip install -U "numba-cuda[cu12]"` (or install the CUDA Toolkit).

## Run
Laptop plugged in, power mode on high performance, close other GPU-heavy programs:
```
python src/run_sweeps.py          # all experiments -> results/timing.csv (several minutes),
                                  # and at the end the sweep report: results/report_all.md + summary_all.csv
                                  # (one table per K, a row per N, plus key findings)
python scripts/compare.py         # correctness check of the final states
python scripts/plot.py            # results/plot1..4_*.png and the numbers for the report
python scripts/report.py          # comparison tables: largest configuration measured (N=1,048,576, K=1,000,
                                  # GPU only) and the largest one that includes the CPU
python scripts/visualize.py --gif              # results/race.gif, largest configuration measured
python scripts/visualize.py --with-cpu --gif   # results/race_with_cpu.gif, largest one with the CPU
                                  # (drop --gif for a window)
%run scripts/live_demo.py        # in a notebook cell: live simulation, one approach after the other
                                  # (each in its own window/output block, a robot cart moving while it computes)
```
For a readable input/output report of a single run, add `--report`, for example
`python src/gpu_fused.py 1024 100 --report`. It shows N, K, seed, simulator and restart rule,
the device and launch configuration, the first initial and final states, the number of
iterations (K), samples collected, kernel launches, execution time (median, min, max),
the one-time copy-back of the final states, the total time, throughput and the final-state hash.
Two notebooks contain the whole project (every file is written by a `%%writefile` cell, with a
heading above every cell):

- `GPU_RL_Benchmark_with_Live_Demo.ipynb` - the same, plus the live simulation demo.
`plot.py`, `report.py` and `visualize.py` show the largest configuration measured by default
(N = 1,048,576, K = 1,000). The CPU baseline runs only up to N = 262,144 games (pure Python would
need about two hours at 1,048,576), so at the largest size the CPU is "not run" and the comparison
there is GPU vectorized vs GPU fused; a second chart, table and animation use the largest
configuration that includes the CPU (N = 262,144). All three scripts accept an optional `N K`.
The live demo (`scripts/live_demo.py`) is for presentations. The approaches run one after the
other (CPU, GPU vectorized, GPU fused); each gets its own window (a notebook output block, or a real
window when run as a script on a laptop) with a robot cart that slides left/right while its pole swings,
computed for real by that approach. Its compute-time counters wrap only the computation, not the
copy-back or the drawing, but it uses a small size, so it is separate from the benchmark and its
numbers are not used in the results.
A single run: `python src/gpu_fused.py 65536 1000` prints
`approach,N,K,reps,median_ms,samples_per_sec,state_hash`.

Optional checks: `python tests/check_order.py`, `python tests/check_vs_gym.py` (needs gymnasium).

## What comes from the paper, and what is our choice
From the paper: CartPole with a 4-float state (cart position, cart velocity, pole angle,
pole angular velocity); N parallel games and K steps per iteration, samples = N x K;
a single-core CPU baseline, GPU vectorization, and a handwritten kernel that fuses K
simulation steps into one launch (simulation only, no neural network); sweeps over
N = 2^0..2^20 and K in {1, 10, 100, 1000}.

Not specified by the paper (our choices):
- **Equations:** gym CartPole-v1 (the paper's code was ported from gym; it does not print them), in float32.
- **Actions:** the paper's loop uses a neural network. We use a deterministic hash of (game, step) to pick push-left or push-right.
- **Resets:** when |x| > 2.4 or |theta| > 12 degrees the game restarts from a gym-style random state (each value in [-0.05, 0.05]), so every game does exactly K steps. gym's 500-step limit is not used (random-action episodes last about 22 steps).
- **Reward:** +1 per step as in gym, so it is implied (N x K) and not stored.

## Caveats to state in the report
1. Simulation only; no inference in the loop (the paper also leaves that as future work).
2. **Python backend.** Every kernel launch from Python costs tens of microseconds of CPU time. The vectorized approach pays that K times per iteration, and fusion removes it. This is the same kind of overhead the paper measured for eager PyTorch (one Python-to-backend call per step), so the fusion speedup here includes it and will be larger than in a C++ launch loop.
3. The paper's 1024x GPU-vs-CPU number compares against Python gym **including inference and CPU-GPU copies**. Our CPU baseline is plain Python with no inference, so compare trends, not absolute ratios.
4. "Fusion speedup" is reported two ways: same K (vectorized time / fused time, the cleanest view) and paper-style (fused at K=1000 vs K=1).
5. All three are timed the same way: wall clock, GPU synchronized before stopping, one warm-up run (kernel compilation is excluded), median of repeats. The vectorized launches are issued back to back with one wait at the end (its strongest form).
6. The CPU baseline runs only up to N = 262,144 games (pure Python is slow, about 7 microseconds per game-step on Colab). CPU runs with N x K above 2,000,000 are timed once instead of as a median of 3. The full sweep takes roughly 45-60 minutes, almost all of it the CPU.
7. One machine per run. The first full results were measured on a **Google Colab NVIDIA T4** (Linux, shared, about 2 CPU cores, clocks can vary). The paper used an RTX 2080 Ti, the same GPU generation (Turing) but a larger chip, so compare trends, not absolute numbers. If the same sweep is also run on the Windows RTX 4060 laptop, note that WDDM launch overhead is higher than on Linux and the display GPU has a ~2 s kernel watchdog.
8. `race.gif` is a slow-motion replay of one recorded game using the real measured times; it is not live during collection (a single fused launch cannot be watched while it runs). `scripts/live_demo.py` (only in the notebook with the live demo) is a separate live demo at a small size whose compute-time counters exclude drawing; it is not part of the benchmark.

## What has been verified so far (on a Mac, no NVIDIA GPU)
- CPU baseline runs; random-action episodes average about 22-24 steps (gym: about 22).
- Physics matches gym CartPole-v1 on 1,196 steps and 55 episode ends (max single-step difference 2e-7).
- The simulator compiles under Numba with float32 types end to end, and matches plain Python to 1.5e-7.
- Running the real GPU scripts in Numba's CUDA simulator gives final states identical to the CPU baseline for three (N, K) configurations.
- **The kernels have not yet been run on a real GPU.** Expect: vectorized and fused give bit-identical states; CPU and GPU can differ in the last bits (different `sin`/`cos` libraries), which `compare.py` allows.
