"""Wilson score interval, lower bound — the workhorse statistic.

The Wilson bound is preferred over the normal (Wald) approximation because it
behaves at the edges: a perfect record never yields LCB = 1.0, and small n
yields honestly wide bounds. ``successes`` may be fractional (ties in paired
comparisons count as half a win).

For continuous scores in [0, 1] (not just pass/fail) the empirical-Bernstein
bounds (``eb_lcb`` / ``eb_ucb``) are the tight, edge-behaving analog of Wilson:
variance-adaptive, so they tighten automatically when scores cluster and stay
honestly wide when they scatter — with a finite-sample (non-asymptotic)
guarantee that holds for every n >= 2.
"""

import math
from statistics import NormalDist


def wilson_lcb(successes, n, confidence=0.95):
    """One-sided Wilson score lower bound for a binomial proportion.

    Returns 0.0 for n == 0 (no evidence -> no lower bound).
    """
    if n == 0:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def min_trials(threshold, confidence=0.95):
    """Smallest n such that a PERFECT record clears ``threshold``.

    For phat = 1 the Wilson LCB reduces to 1 / (1 + z²/n), so the requirement
    is n >= z² · threshold / (1 - threshold). Useful for sizing a gate: below
    this n, the gate is unpassable even by a flawless run — by design.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    z = NormalDist().inv_cdf(confidence)
    return math.ceil(z * z * threshold / (1 - threshold))


def _eb_stats(values, confidence):
    """Empirical-Bernstein ``(mean, margin)`` for observations in [0, 1].

    Private: callers guarantee the precondition n >= 2 and every value in
    [0, 1]. ``margin`` is the one-sided Maurer & Pontil (2009) half-width
    ``sqrt(2·V·L/n) + 7·L/(3(n-1))`` with the UNBIASED sample variance
    ``V = sum((x-mean)**2)/(n-1)`` and ``L = ln(2/delta)``, ``delta = 1 -
    confidence``. The '2' inside ``ln(2/delta)`` is load-bearing (M&P union two
    sub-events at delta/2 each); do not simplify it to ``ln(1/delta)``.
    """
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)   # unbiased sample variance
    log_term = math.log(2.0 / (1.0 - confidence))          # ln(2/delta)
    margin = math.sqrt(2.0 * var * log_term / n) + 7.0 * log_term / (3.0 * (n - 1))
    return mean, margin


def eb_lcb(values, confidence=0.95):
    """One-sided empirical-Bernstein LOWER bound on the mean of [0, 1] obs.

    The tight, edge-behaving analog of ``wilson_lcb`` for continuous scores
    (Maurer & Pontil 2009): a finite-sample lower confidence bound on E[X] for
    i.i.d. X in [0, 1] that is variance-adaptive — it tightens when the scores
    cluster and stays honestly wide when they scatter. Returns 0.0 for < 2
    observations (no variance -> no bound). Raises ValueError if any value is
    outside [0, 1] (the bound is invalid otherwise).
    """
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("eb_lcb values must lie in [0, 1]")
    if len(values) < 2:
        return 0.0
    mean, margin = _eb_stats(values, confidence)
    return max(0.0, mean - margin)


def eb_ucb(values, confidence=0.95):
    """One-sided empirical-Bernstein UPPER bound on the mean of [0, 1] obs.

    Mirror of :func:`eb_lcb` (the same theorem applied to ``1 - x``). Returns
    1.0 for < 2 observations (no variance -> no bound). Raises ValueError if
    any value is outside [0, 1].
    """
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError("eb_ucb values must lie in [0, 1]")
    if len(values) < 2:
        return 1.0
    mean, margin = _eb_stats(values, confidence)
    return min(1.0, mean + margin)
