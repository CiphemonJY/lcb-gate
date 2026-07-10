"""Wilson score interval, lower bound — the workhorse statistic.

The Wilson bound is preferred over the normal (Wald) approximation because it
behaves at the edges: a perfect record never yields LCB = 1.0, and small n
yields honestly wide bounds. ``successes`` may be fractional (ties in paired
comparisons count as half a win).
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
