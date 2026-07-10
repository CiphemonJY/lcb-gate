"""lcb-gate — statistical pass/fail gating for stochastic tests.

A single green run of a stochastic test proves almost nothing. This package
runs the trial N times and passes only when the Wilson lower confidence bound
of the observed pass rate clears your threshold — so "it passed" means
"it passes at ≥ X% with statistical backing", not "it got lucky once".
"""

from .stats import min_trials, wilson_lcb
from .gate import CompareResult, GateResult, compare, run_gate

__version__ = "0.1.0"

__all__ = [
    "CompareResult",
    "GateResult",
    "compare",
    "min_trials",
    "run_gate",
    "wilson_lcb",
    "__version__",
]
