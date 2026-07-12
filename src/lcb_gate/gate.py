"""The gate: N-run consensus with an LCB verdict, and paired A/B comparison."""

import math
from dataclasses import dataclass

from .stats import eb_lcb, eb_ucb, wilson_lcb


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
class ScoreGateResult:
    mean: float         # observed mean over trials_run; the imputation lives only in lcb
    n: int              # the gate's denominator — unrun trials imputed at 0.0 (the worst score)
    trials_run: int
    lcb: float          # empirical-Bernstein lower bound on the mean, over the fixed denominator n
    threshold: float
    confidence: float
    passed: bool

    def __str__(self):
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{verdict}: mean score {self.mean:.3f} over {self.trials_run} runs "
                f"(gate n={self.n}); mean-LCB {self.lcb:.3f} "
                f"{'>=' if self.passed else '<'} {self.threshold} "
                f"@ {self.confidence:.0%} confidence")


def score_gate(trial, n=25, threshold=0.9, confidence=0.95, stop_early=True):
    """Run ``trial(i) -> float in [0, 1]`` up to ``n`` times; pass iff the
    empirical-Bernstein lower bound of the MEAN score (over the FIXED
    denominator n) clears ``threshold``.

    The continuous-score analog of :func:`run_gate`: unrun trials are imputed
    at the worst value (0.0), so the reported bound is always conservative.
    Each score is validated to be in [0, 1] (ValueError otherwise) — an
    unnormalized score would give a silently invalid bound. With ``stop_early``
    the loop exits as soon as the verdict is settled: a pass-side check on the
    zero-imputed bound (the pessimal completion in the regime where it can
    trigger) and a fail-side check on the best-case mean ceiling (a rigorous
    upper bound on the final LCB for ANY completion, since eb_lcb <= sample
    mean <= that ceiling).
    """
    scores = []
    total = 0.0
    for i in range(n):
        ran = len(scores)
        if stop_early:
            # pass even if every remaining trial scores 0 (worst-case imputation)
            if eb_lcb(scores + [0.0] * (n - ran), confidence) >= threshold:
                break
            # fail even if every remaining trial scores 1 (best-case mean ceiling)
            if (total + (n - ran)) / n < threshold:
                break
        s = trial(i)
        if not 0.0 <= s <= 1.0:
            raise ValueError(
                f"score_gate trial {i} returned {s!r}; scores must be in [0, 1]")
        scores.append(s)
        total += s
    ran = len(scores)
    lcb = eb_lcb(scores + [0.0] * (n - ran), confidence)
    # observed mean over the runs actually taken; n == 0 yields a clean FAIL
    # (lcb 0.0) instead of a ZeroDivisionError, mirroring run_gate(n=0).
    mean = total / ran if ran else 0.0
    return ScoreGateResult(mean=mean, n=n, trials_run=ran, lcb=lcb,
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


@dataclass
class RankEstimate:
    index: int
    mean: float
    lcb: float
    ucb: float
    trials: int


@dataclass
class RankResult:
    winner: int         # index of the empirical leader; certified only when proven
    estimates: list     # list[RankEstimate], one per candidate, in index order
    rounds: int
    trials: int         # total candidate evaluations (rounds x number of candidates)
    confidence: float
    proven: bool

    def __str__(self):
        if self.proven:
            w = self.estimates[self.winner]
            return (f"BEST: candidate #{self.winner} (mean {w.mean:.3f}, "
                    f"LCB {w.lcb:.3f}) beats every rival's UCB over "
                    f"{self.rounds} rounds / {self.trials} evals "
                    f"@ {self.confidence:.0%}")
        return (f"NOT PROVEN: no candidate's LCB cleared every rival's UCB "
                f"in {self.rounds} rounds / {self.trials} evals "
                f"@ {self.confidence:.0%}")


def rank(candidates, n_max=2000, confidence=0.95, seeds=None, min_rounds=2):
    """Multiplicity-honest best-of-K racing: which candidate is provably best?

    ``candidates`` is a sequence of callables ``candidate(seed) -> float`` in
    [0, 1]. Each round scores EVERY candidate on the SAME seed (common random
    numbers), then certifies the empirical leader the moment its
    empirical-Bernstein LCB clears every rival's UCB. The confidence used for
    those bounds is split two ways to control the selection error at the order
    of ``1 - confidence``: Bonferroni over the K arms (the ``/k`` below) AND an
    anytime union over rounds via the Basel factor ``6/(pi^2 * t^2)``
    (sum_t 1/t^2 = pi^2/6), which is what keeps repeated peeking honest. The
    worst-case guarantee carries a small constant factor — winner and rivals
    are data-dependent, so a fully rigorous union spans the winner's lower
    bound AND every rival's upper bound — but empirical-Bernstein is
    conservative enough that the measured false-selection rate is ~0 (see
    tests/test_lcb.py::test_rank_false_selection_under_null).

    The honest default verdict is ``NOT PROVEN`` — an underpowered field never
    certifies a winner. Racing stops at ``n_max`` total evaluations. Scores are
    validated to be in [0, 1] (ValueError otherwise). This is the continuous,
    K-way generalization of :func:`compare`.
    """
    k = len(candidates)
    if k < 2:
        raise ValueError("rank() needs at least two candidates")
    # range(n_max) yields far more distinct seeds than the rounds <= n_max/k we
    # can afford, keeping CRN reproducible without importing itertools.
    seed_iter = iter(seeds) if seeds is not None else iter(range(n_max))
    scores = [[] for _ in range(k)]
    budget = 1.0 - confidence
    rounds = 0
    winner = 0
    proven = False
    means = [0.0] * k
    for s in seed_iter:
        for j, cand in enumerate(candidates):
            v = cand(s)
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"rank() candidate #{j} returned {v!r} on seed {s!r}; "
                    f"scores must be in [0, 1]")
            scores[j].append(v)
        rounds += 1
        means = [sum(sc) / len(sc) for sc in scores]
        winner = max(range(k), key=lambda j: means[j])
        if rounds >= min_rounds:          # empirical-Bernstein needs >= 2 samples
            conf = 1.0 - budget * 6.0 / (math.pi * math.pi * k * rounds * rounds)
            lcb_w = eb_lcb(scores[winner], conf)
            proven = all(lcb_w > eb_ucb(scores[j], conf)
                         for j in range(k) if j != winner)
        if proven or rounds * k >= n_max:
            break
    # Report intervals at the SAME conf that drove the final decision, so a
    # printed winner.lcb > rivals.ucb is visibly consistent with `proven`.
    if rounds >= min_rounds:
        conf = 1.0 - budget * 6.0 / (math.pi * math.pi * k * rounds * rounds)
    else:
        conf = 1.0 - budget / k
    estimates = [RankEstimate(index=j, mean=means[j],
                              lcb=eb_lcb(scores[j], conf),
                              ucb=eb_ucb(scores[j], conf),
                              trials=len(scores[j]))
                 for j in range(k)]
    return RankResult(winner=winner, estimates=estimates, rounds=rounds,
                      trials=sum(len(sc) for sc in scores),
                      confidence=confidence, proven=proven)
