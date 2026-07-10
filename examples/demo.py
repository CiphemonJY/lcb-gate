"""Why a single green run lies, in three acts. Deterministic (seeded), stdlib only.

Run: ``python examples/demo.py`` (asserts its own expected outcomes; exit 0).
"""

import random
import sys
from pathlib import Path

try:
    import lcb_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcb_gate import compare, min_trials, run_gate


def flaky_eval(true_pass_rate, salt):
    """A stochastic 'agent eval': passes with the given true probability."""
    def trial(i):
        return random.Random((salt, i)).random() < true_pass_rate
    return trial


print("ACT 1 — the 75% eval that looks fine once")
trial = flaky_eval(0.75, salt=101)
singles = [trial(i) for i in range(5)]
print(f"    five individual CI runs: {['green' if s else 'RED' for s in singles]}")
res = run_gate(trial, n=30, threshold=0.9)
print(f"    gate verdict: {res}")
assert any(singles)          # at least one single run looked green
assert not res.passed        # the gate is not fooled

print("\nACT 2 — the genuinely good eval, and what n it takes")
trial = flaky_eval(0.97, salt=202)
small = run_gate(trial, n=25, threshold=0.9)
large = run_gate(trial, n=150, threshold=0.9)
print(f"    n=25:  {small}")
print(f"    n=150: {large}")
print(f"    (a PERFECT record needs n >= {min_trials(0.9)} to clear 0.9 at 95%)")
assert large.passed

print("\nACT 3 — candidate vs champion, paired seeds (common random numbers)")


def champion(seed):
    return random.Random(seed).random()


def candidate(seed):
    # same shared noise as the champion (CRN), plus a real but small edge:
    # wins the pair 60% of the time
    r = random.Random(seed)
    base = r.random()
    return base + (0.01 if r.random() < 0.6 else -0.01)


underpowered = compare(candidate, champion, seeds=range(30))
powered = compare(candidate, champion, seeds=range(400))
print(f"    n=30:  {underpowered}")
print(f"    n=400: {powered}")
print("    same candidate, same true edge — only the evidence changed")
print("    (30 pairs is so underpowered the sample can even reverse the sign)")
assert not underpowered.better   # honest: cannot certify on 30 pairs
assert powered.better            # certified on 400

print("\nAll demo assertions passed.")
