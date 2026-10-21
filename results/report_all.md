# Sweep report: CPU vs GPU vectorized vs GPU fused

Configurations: N from 1 to 1,048,576 games, K in [1, 10, 100, 1000] steps per iteration. Times are the median time (ms) to collect all N x K samples. Fusion speedup = vectorized time / fused time at the same K. '-' = the CPU baseline was not run (too many games for pure Python). CPU runs with N x K above 2,000,000 were timed once; all other timings are the median of repetitions.

## K = 1 steps per iteration

| N (games) | CPU (ms) | GPU vectorized (ms) | GPU fused (ms) | Fusion speedup (vec/fused) | Vectorized vs CPU | Fused vs CPU | Fused throughput (samples/s) | Vec = fused? |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.031 | 0.041 | 0.047 | 0.8813x | 0.7405x | 0.6526x | 2.139e+04 | yes |
| 4 | 0.058 | 0.040 | 0.037 | 1.1x | 1.5x | 1.6x | 1.091e+05 | yes |
| 16 | 0.208 | 0.038 | 0.037 | 1.0x | 5.4x | 5.6x | 4.278e+05 | yes |
| 64 | 0.760 | 0.045 | 0.039 | 1.2x | 16.9x | 19.6x | 1.653e+06 | yes |
| 256 | 2.974 | 0.039 | 0.039 | 0.9835x | 76.9x | 75.6x | 6.509e+06 | yes |
| 1,024 | 8.746 | 0.038 | 0.040 | 0.9591x | 230.3x | 220.8x | 2.586e+07 | yes |
| 4,096 | 26.396 | 0.039 | 0.041 | 0.9419x | 678.9x | 639.4x | 9.923e+07 | yes |
| 16,384 | 103.593 | 0.039 | 0.039 | 1.0x | 2,638.6x | 2,667.9x | 4.219e+08 | yes |
| 65,536 | 511.101 | 0.043 | 0.041 | 1.0x | 11,839.3x | 12,405.4x | 1.591e+09 | yes |
| 262,144 | 1,696.166 | 0.065 | 0.061 | 1.1x | 26,114.9x | 27,980.3x | 4.324e+09 | yes |
| 1,048,576 | - | 0.170 | 0.169 | 1.0x | - | - | 6.191e+09 | yes |

## K = 10 steps per iteration

| N (games) | CPU (ms) | GPU vectorized (ms) | GPU fused (ms) | Fusion speedup (vec/fused) | Vectorized vs CPU | Fused vs CPU | Fused throughput (samples/s) | Vec = fused? |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.060 | 0.390 | 0.042 | 9.3x | 0.1553x | 1.4x | 2.392e+05 | yes |
| 4 | 0.259 | 0.541 | 0.042 | 12.9x | 0.4788x | 6.2x | 9.494e+05 | yes |
| 16 | 0.946 | 0.559 | 0.044 | 12.9x | 1.7x | 21.7x | 3.676e+06 | yes |
| 64 | 4.133 | 0.603 | 0.046 | 13.0x | 6.9x | 89.3x | 1.383e+07 | yes |
| 256 | 15.765 | 0.596 | 0.046 | 13.0x | 26.4x | 343.4x | 5.576e+07 | yes |
| 1,024 | 63.958 | 0.339 | 0.045 | 7.6x | 188.4x | 1,427.6x | 2.285e+08 | yes |
| 4,096 | 253.064 | 0.354 | 0.045 | 7.9x | 715.3x | 5,669.0x | 9.175e+08 | yes |
| 16,384 | 1,061.423 | 0.307 | 0.054 | 5.7x | 3,457.2x | 19,780.5x | 3.053e+09 | yes |
| 65,536 | 4,152.037 | 0.310 | 0.085 | 3.6x | 13,412.7x | 48,939.6x | 7.725e+09 | yes |
| 262,144 | 16,862.260 | 0.358 | 0.202 | 1.8x | 47,144.7x | 83,613.1x | 1.300e+10 | yes |
| 1,048,576 | - | 1.406 | 0.691 | 2.0x | - | - | 1.517e+10 | yes |

## K = 100 steps per iteration

| N (games) | CPU (ms) | GPU vectorized (ms) | GPU fused (ms) | Fusion speedup (vec/fused) | Vectorized vs CPU | Fused vs CPU | Fused throughput (samples/s) | Vec = fused? |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.704 | 3.172 | 0.127 | 25.0x | 0.2220x | 5.6x | 7.882e+05 | yes |
| 4 | 2.903 | 3.105 | 0.129 | 24.1x | 0.9347x | 22.6x | 3.110e+06 | yes |
| 16 | 11.083 | 3.014 | 0.137 | 22.0x | 3.7x | 80.7x | 1.166e+07 | yes |
| 64 | 42.814 | 3.066 | 0.148 | 20.7x | 14.0x | 288.6x | 4.314e+07 | yes |
| 256 | 177.485 | 3.243 | 0.165 | 19.7x | 54.7x | 1,078.8x | 1.556e+08 | yes |
| 1,024 | 691.020 | 3.225 | 0.166 | 19.5x | 214.2x | 4,168.8x | 6.177e+08 | yes |
| 4,096 | 2,738.574 | 3.171 | 0.164 | 19.3x | 863.7x | 16,708.8x | 2.499e+09 | yes |
| 16,384 | 11,256.859 | 3.040 | 0.224 | 13.6x | 3,702.9x | 50,235.9x | 7.312e+09 | yes |
| 65,536 | 44,948.592 | 3.057 | 0.625 | 4.9x | 14,704.1x | 71,877.5x | 1.048e+10 | yes |
| 262,144 | 186,857.584 | 3.325 | 2.149 | 1.5x | 56,197.6x | 86,931.1x | 1.220e+10 | yes |
| 1,048,576 | - | 13.738 | 8.773 | 1.6x | - | - | 1.195e+10 | yes |

## K = 1,000 steps per iteration

| N (games) | CPU (ms) | GPU vectorized (ms) | GPU fused (ms) | Fusion speedup (vec/fused) | Vectorized vs CPU | Fused vs CPU | Fused throughput (samples/s) | Vec = fused? |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.729 | 28.992 | 1.011 | 28.7x | 0.2321x | 6.7x | 9.891e+05 | yes |
| 4 | 26.780 | 29.815 | 1.065 | 28.0x | 0.8982x | 25.2x | 3.758e+06 | yes |
| 16 | 105.253 | 30.132 | 1.154 | 26.1x | 3.5x | 91.2x | 1.386e+07 | yes |
| 64 | 436.888 | 30.151 | 1.190 | 25.3x | 14.5x | 367.1x | 5.377e+07 | yes |
| 256 | 1,720.442 | 31.097 | 0.955 | 32.6x | 55.3x | 1,801.9x | 2.681e+08 | yes |
| 1,024 | 6,931.884 | 30.607 | 0.777 | 39.4x | 226.5x | 8,917.6x | 1.317e+09 | yes |
| 4,096 | 27,851.006 | 30.144 | 0.783 | 38.5x | 923.9x | 35,585.1x | 5.233e+09 | yes |
| 16,384 | 112,164.093 | 30.267 | 1.136 | 26.7x | 3,705.8x | 98,775.1x | 1.443e+10 | yes |
| 65,536 | 446,820.358 | 29.709 | 3.582 | 8.3x | 15,040.1x | 124,723.8x | 1.829e+10 | yes |
| 262,144 | 1,778,428.639 | 32.724 | 12.364 | 2.6x | 54,347.1x | 143,842.2x | 2.120e+10 | yes |
| 1,048,576 | - | 136.228 | 42.515 | 3.2x | - | - | 2.466e+10 | yes |

## Key findings (computed from the table data)

- Correctness: GPU vectorized and GPU fused final states were bit-identical in 44 of 44 configurations.
- Peak throughput, CPU (1 core): 1.691e+05 samples/s (N = 16, K = 10).
- Peak throughput, GPU vectorized: 8.011e+09 samples/s (N = 262,144, K = 1,000).
- Peak throughput, GPU fused: 2.466e+10 samples/s (N = 1,048,576, K = 1,000).
- Peak GPU fused throughput is 145,858x the peak CPU throughput (pure-Python CPU without inference; not comparable with the paper's 1024x).

**Where the GPU first beats the CPU** (smallest N where the GPU time is below the CPU time):
- K = 1: GPU vectorized: N = 4; GPU fused: N = 4
- K = 10: GPU vectorized: N = 16; GPU fused: N = 1
- K = 100: GPU vectorized: N = 16; GPU fused: N = 1
- K = 1,000: GPU vectorized: N = 16; GPU fused: N = 1

**Fusion speedup (vectorized time / fused time) per K:**
- K = 1: from 0.9x (N = 1) to 1.2x (N = 64); 1.0x at the largest N (1,048,576).
- K = 10: from 1.8x (N = 262,144) to 13.0x (N = 64); 2.0x at the largest N (1,048,576).
- K = 100: from 1.5x (N = 262,144) to 25.0x (N = 1); 1.6x at the largest N (1,048,576).
- K = 1,000: from 2.6x (N = 262,144) to 39.4x (N = 1,024); 3.2x at the largest N (1,048,576).

**Vectorized time per kernel launch** (vectorized time / K, for N <= 4,096):
- K = 10: 33.9 to 60.3 microseconds per launch.
- K = 100: 30.1 to 32.4 microseconds per launch.
- K = 1,000: 29.0 to 31.1 microseconds per launch.
If this stays about the same while N grows, the vectorized run is limited by the cost of launching kernels (here from Python), not by the GPU's work; fusion removes those launches.
