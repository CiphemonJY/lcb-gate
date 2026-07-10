"""Core suite — deterministic, zero dependencies: ``python tests/test_lcb.py``."""

import sys
from pathlib import Path

try:
    import lcb_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcb_gate import compare, min_trials, run_gate, wilson_lcb


# ------------------------------------------------------------------ stats

def test_wilson_known_value_90_of_100():
    # hand-computed: z=1.6449, LCB ≈ 0.8396
    assert abs(wilson_lcb(90, 100, 0.95) - 0.8396) < 1e-3


def test_wilson_perfect_record_closed_form():
    # for phat=1 the bound reduces to 1/(1+z²/n)
    z2 = 1.6448536269514722 ** 2
    assert abs(wilson_lcb(20, 20, 0.95) - 1 / (1 + z2 / 20)) < 1e-9


def test_wilson_edges():
    assert wilson_lcb(0, 0) == 0.0
    assert wilson_lcb(0, 10) == 0.0


def test_wilson_never_exceeds_point_estimate():
    for s, n in [(1, 2), (5, 10), (45, 50), (99, 100)]:
        assert wilson_lcb(s, n) <= s / n


def test_wilson_tightens_with_n():
    assert wilson_lcb(9, 10) < wilson_lcb(90, 100) < wilson_lcb(900, 1000)


def test_min_trials_table():
    assert min_trials(0.8) == 11
    assert min_trials(0.9) == 25
    assert min_trials(0.95) == 52


def test_min_trials_is_exact_boundary():
    n = min_trials(0.9)
    assert wilson_lcb(n, n) >= 0.9          # perfect record at n passes
    assert wilson_lcb(n - 1, n - 1) < 0.9   # one fewer trial cannot pass


# ------------------------------------------------------------------ gate

def test_gate_passes_one_failure_in_twenty():
    res = run_gate(lambda i: i != 3, n=20, threshold=0.75)
    assert res.passed and res.passes == 19


def test_gate_fails_chronic_flake():
    res = run_gate(lambda i: i % 2 == 0, n=20, threshold=0.9)  # 50% pass rate
    assert not res.passed


def test_gate_early_stop_on_hopeless_run():
    res = run_gate(lambda i: False, n=100, threshold=0.9)
    assert not res.passed
    assert res.trials_run < 100     # settled long before 100 runs


def test_gate_early_stop_verdict_matches_full_run():
    trial = lambda i: i % 7 != 0    # noqa: E731 — ~86% pass rate, deterministic
    early = run_gate(trial, n=50, threshold=0.7, stop_early=True)
    full = run_gate(trial, n=50, threshold=0.7, stop_early=False)
    assert early.passed == full.passed


def test_gate_perfect_record_at_min_trials():
    n = min_trials(0.9)
    res = run_gate(lambda i: True, n=n, threshold=0.9)
    assert res.passed


# ------------------------------------------------------------------ compare

def _champ(seed):
    return (seed % 100) / 100


def test_compare_detects_70pct_winner():
    def cand(seed):
        return _champ(seed) + (0.01 if seed % 10 < 7 else -0.01)
    res = compare(cand, _champ, seeds=range(100))
    assert res.better
    assert abs(res.win_rate - 0.7) < 1e-9


def test_compare_null_candidate_not_proven():
    res = compare(_champ, _champ, seeds=range(100))   # all ties -> 0.5 win rate
    assert not res.better
    assert res.win_rate == 0.5


def test_compare_underpowered_n_not_proven():
    def cand(seed):
        return _champ(seed) + (0.01 if seed % 10 < 6 else -0.01)   # true 60% winner
    assert not compare(cand, _champ, seeds=range(10)).better       # n=10: can't know
    assert compare(cand, _champ, seeds=range(500)).better          # n=500: proven


# ------------------------------------------------------- bare-python runner

if __name__ == "__main__":
    import inspect
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)
             and not inspect.signature(f).parameters]
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError:
            print(f"  FAIL  {name}")
            failures.append(name)
    print(f"\n{'ALL PASS' if not failures else f'FAILED: {failures}'}"
          f" — {len(cases) - len(failures)}/{len(cases)}")
    sys.exit(1 if failures else 0)
