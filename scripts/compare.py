"""Compare the final states written by two programs (--save files).

usage: python scripts/compare.py [a.bin b.bin]

Default compares results/final_cpu.bin, final_vectorized.bin, final_fused.bin
(all N=1024, K=10). Expected: vectorized == fused bit for bit. CPU vs GPU can
differ in the last bits because the CPU and GPU math libraries implement
sin/cos slightly differently, so we only check they are very close.
"""
import sys
from array import array


def load(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def compare(name_a, name_b):
    a, b = load(name_a), load(name_b)
    assert len(a) == len(b), "different sizes"
    diffs = [abs(x - y) for x, y in zip(a, b)]
    worst = max(diffs)
    games = len(a) // 4
    bad = len({i // 4 for i, d in enumerate(diffs) if d > 1e-3})
    print(f"{name_a}  vs  {name_b}")
    print(f"  identical bits: {a == b} | max abs difference: {worst:.2e} | "
          f"games differing by > 1e-3: {bad} of {games}")
    return a == b, worst


if len(sys.argv) == 3:
    compare(sys.argv[1], sys.argv[2])
else:
    same, _ = compare("results/final_vectorized.bin", "results/final_fused.bin")
    print("  PASS" if same else "  FAIL: vectorized and fused must be bit-identical")
    _, worst = compare("results/final_cpu.bin", "results/final_fused.bin")
    print("  PASS" if worst < 1e-3 else "  FAIL: CPU and GPU differ too much")
