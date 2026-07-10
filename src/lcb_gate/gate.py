"""The gate: N-run consensus with an LCB verdict, and paired A/B comparison."""

from dataclasses import dataclass

from .stats import wilson_lcb


@dataclass
class GateResult:
    passes: int
    n: int              # the gate's denominator — unrun trials count as failures
    trials_run: int
    lcb: float
    threshold: float
    confidence: float
    passed: bool

    def __str__(self):
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{verdict}: {self.passes}/{self.trials_run} passed "
                f"(gate n={self.n}); LCB {self.lcb:.3f} "
                f"{'>=' if self.passed else '<'} {self.threshold} "
                f"@ {self.confidence:.0%} confidence")


def run_gate(trial, n=25, threshold=0.9, confidence=0.95, stop_early=True):
    """Run ``trial(i) -> bool`` up to ``n`` times; pass iff the Wilson LCB of
    the pass rate (over the FIXED denominator n) clears ``threshold``.

    Unrun trials count as failures, so the reported bound is always
    conservative. With ``stop_early`` the loop exits as soon as the verdict is
    mathematically settled either way — the verdict is identical to running
    all n trials, only cheaper.
    """
    passes = 0
    ran = 0
    for i in range(n):
        # verdict already settled?
        if stop_early:
            if wilson_lcb(passes, n, confidence) >= threshold:
                break                                     # pass even if the rest fail
            if wilson_lcb(passes + (n - ran), n, confidence) < threshold:
                break                                     # fail even if the rest pass
        passes += 1 if trial(i) else 0
        ran += 1
    lcb = wilson_lcb(passes, n, confidence)
    return GateResult(passes=passes, n=n, trials_run=ran, lcb=lcb,
                      threshold=threshold, confidence=confidence,
                      passed=lcb >= threshold)


@dataclass
class CompareResult:
    wins: float         # ties count 0.5
    n: int
    win_rate: float
    lcb: float
    confidence: float
    better: bool        # LCB of the win rate > 0.5

    def __str__(self):
        verdict = "BETTER" if self.better else "NOT PROVEN"
        return (f"{verdict}: candidate won {self.wins}/{self.n} paired trials "
                f"({self.win_rate:.1%}); win-rate LCB {self.lcb:.3f} "
                f"{'>' if self.better else '<='} 0.5 @ {self.confidence:.0%}")


def compare(candidate, champion, n=200, confidence=0.95, seeds=None):
    """Paired comparison: is ``candidate`` better than ``champion``?

    Both callables receive the SAME seed per trial (common random numbers) —
    pairing removes shared noise, which is where most of the statistical power
    comes from. Callables return a comparable score (float or bool); ties
    count as half a win. Verdict ``better`` requires the win-rate LCB to
    exceed 0.5 — "not proven" is the honest default for underpowered n.
    """
    seed_list = list(seeds) if seeds is not None else list(range(n))
    total = len(seed_list)
    if total == 0:
        raise ValueError("compare() needs at least one seed")
    wins = 0.0
    for s in seed_list:
        c, ch = candidate(s), champion(s)
        if c > ch:
            wins += 1.0
        elif c == ch:
            wins += 0.5
    lcb = wilson_lcb(wins, total, confidence)
    return CompareResult(wins=wins, n=total, win_rate=wins / total, lcb=lcb,
                         confidence=confidence, better=lcb > 0.5)
