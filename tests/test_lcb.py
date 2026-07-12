"""Core suite — deterministic, zero dependencies: ``python tests/test_lcb.py``."""

import random
import sys
from pathlib import Path

try:
    import lcb_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcb_gate import (
    compare,
    eb_lcb,
    eb_ucb,
    min_trials,
    rank,
    run_gate,
    score_gate,
    wilson_lcb,
)


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


# ------------------------------------------------------ empirical-bernstein

def test_eb_known_value_zero_variance():
    # hand: margin = 7*ln(40)/(3*49) = 0.17566, so LCB = 0.9 - 0.17566
    assert abs(eb_lcb([0.9] * 50) - 0.7243) < 1e-3


def test_eb_perfect_scores_closed_form():
    # zero variance -> margin = 7*ln(40)/(3*99), LCB = 1.0 - margin
    assert abs(eb_lcb([1.0] * 100) - 0.9131) < 1e-3


def test_eb_edges():
    assert eb_lcb([]) == 0.0
    assert eb_lcb([0.5]) == 0.0        # < 2 obs -> no variance -> no bound
    assert eb_ucb([]) == 1.0
    assert eb_ucb([0.5]) == 1.0


def test_eb_ucb_clamps():
    assert eb_ucb([0.9] * 50) == 1.0   # mean + margin exceeds 1 -> clamped


def test_eb_brackets_mean():
    for v in ([0.9] * 50, [0.8, 1.0] * 10, [0.1, 0.5, 0.9, 0.3, 0.7]):
        assert eb_lcb(v) <= sum(v) / len(v) <= eb_ucb(v)


def test_eb_tightens_with_n():
    # mean and variance fixed, only n grows -> the bound tightens upward
    assert eb_lcb([0.8, 1.0] * 5) < eb_lcb([0.8, 1.0] * 50) < eb_lcb([0.8, 1.0] * 500)


def test_eb_lower_variance_is_tighter():
    # same mean 0.9, lower variance -> higher (tighter) lower bound
    assert eb_lcb([0.9] * 100) > eb_lcb([0.8, 1.0] * 50)


def test_eb_rejects_out_of_range():
    for bad in ([0.5, 1.5], [-0.1, 0.5]):
        raised = False
        try:
            eb_lcb(bad)
        except ValueError:
            raised = True
        assert raised


# ----------------------------------------------------------------- score gate

def test_score_gate_passes_high_mean():
    assert score_gate(lambda i: 0.95, n=100, threshold=0.8).passed


def test_score_gate_fails_mediocre():
    assert not score_gate(lambda i: 0.5, n=100, threshold=0.8).passed


def test_score_gate_perfect_needs_n():
    # EB analog of the min_trials story: a perfect stream still needs enough n
    assert not score_gate(lambda i: 1.0, n=50, threshold=0.9).passed
    assert score_gate(lambda i: 1.0, n=200, threshold=0.9).passed


def test_score_gate_early_stop_on_hopeless():
    res = score_gate(lambda i: 0.0, n=200, threshold=0.8)
    assert not res.passed and res.trials_run < 200      # settled once ran > 0.2n


def test_score_gate_early_stop_matches_full_run():
    trial = lambda i: ((i * 37) % 100) / 100    # noqa: E731 — deterministic spread
    early = score_gate(trial, n=60, threshold=0.4, stop_early=True)
    full = score_gate(trial, n=60, threshold=0.4, stop_early=False)
    assert early.passed == full.passed


def test_score_gate_rejects_out_of_range():
    raised = False
    try:
        score_gate(lambda i: 1.5, n=10)
    except ValueError:
        raised = True
    assert raised


# ------------------------------------------------------------------------ rank

def _judge(p, tag):
    # stochastic but seed-stable: a string seed is stable across Python versions
    # and legal on 3.13+ (tuple seeds became a TypeError there)
    def f(seed):
        return 1.0 if random.Random(f"{tag}:{seed}").random() < p else 0.0
    return f


def test_rank_picks_clear_winner():
    arms = [_judge(0.9, "A"), _judge(0.6, "B"), _judge(0.4, "C")]
    res = rank(arms, n_max=6000, seeds=range(2000))
    assert res.proven and res.winner == 0


def test_rank_requires_two_candidates():
    raised = False
    try:
        rank([_judge(0.5, "x")])
    except ValueError:
        raised = True
    assert raised


def test_rank_seed_determinism():
    arms = [_judge(0.9, "A"), _judge(0.6, "B"), _judge(0.4, "C")]
    a = rank(arms, seeds=range(400))
    b = rank(arms, seeds=range(400))
    assert a.winner == b.winner and a.rounds == b.rounds


def test_rank_estimates_shape():
    arms = [_judge(0.9, "A"), _judge(0.6, "B"), _judge(0.4, "C")]
    res = rank(arms, n_max=3000, seeds=range(1000))
    assert len(res.estimates) == 3
    for e in res.estimates:
        assert hasattr(e, "index") and hasattr(e, "mean")
        assert hasattr(e, "lcb") and hasattr(e, "ucb") and hasattr(e, "trials")
    assert res.estimates[res.winner].mean == max(e.mean for e in res.estimates)


def test_rank_equal_arms_not_proven():
    arms = [_judge(0.7, "p"), _judge(0.7, "q")]
    assert not rank(arms, n_max=600, seeds=range(300)).proven


def test_rank_false_selection_under_null():
    reps, false = 300, 0
    for rep in range(reps):
        arms = [_judge(0.5, f"{rep}:{a}") for a in range(3)]  # 3 truly-equal arms
        if rank(arms, n_max=900, confidence=0.95, seeds=range(300)).proven:
            false += 1                                        # ANY proof is a false selection
    # target error is 0.05; 0.10 leaves MC/anytime slack. A plain per-look
    # Bonferroni would blow past this — the anytime 6/(pi^2 t^2) factor is what
    # makes it pass.
    assert false / reps <= 0.10


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
