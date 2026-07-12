"""lcb-gate — statistical pass/fail gating for stochastic tests.

A single green run of a stochastic test proves almost nothing. This package
runs the trial N times and passes only when the Wilson lower confidence bound
of the observed pass rate clears your threshold — so "it passed" means
"it passes at ≥ X% with statistical backing", not "it got lucky once".

For continuous verifier scores (an LLM-judge in [0, 1] rather than a bare
pass/fail) the same idea runs on the empirical-Bernstein bound: ``score_gate``
gates a mean score, and ``rank`` races K candidates to a multiplicity-honest
best-of-K verdict.
"""

from .stats import eb_lcb, eb_ucb, min_trials, wilson_lcb
from .gate import (
    CompareResult,
    GateResult,
    RankEstimate,
    RankResult,
    ScoreGateResult,
    compare,
    rank,
    run_gate,
    score_gate,
)

__version__ = "0.2.0"

__all__ = [
    "CompareResult",
    "GateResult",
    "RankEstimate",
    "RankResult",
    "ScoreGateResult",
    "compare",
    "eb_lcb",
    "eb_ucb",
    "min_trials",
    "rank",
    "run_gate",
    "score_gate",
    "wilson_lcb",
    "__version__",
]
