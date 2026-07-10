"""pytest integration: the ``lcb`` fixture.

Usage::

    def test_agent_completes_task(lcb):
        lcb.check(lambda i: run_my_agent(seed=i).succeeded, n=25, threshold=0.9)

The test fails with a statistical explanation unless the Wilson LCB of the
pass rate clears the threshold.
"""

import pytest

from .gate import compare, run_gate


class LcbHelper:
    def check(self, trial, n=25, threshold=0.9, confidence=0.95, stop_early=True):
        """Fail the surrounding test unless the gate passes. Returns GateResult."""
        result = run_gate(trial, n=n, threshold=threshold,
                          confidence=confidence, stop_early=stop_early)
        if not result.passed:
            pytest.fail(f"lcb-gate {result}")
        return result

    def check_better(self, candidate, champion, n=200, confidence=0.95, seeds=None):
        """Fail the surrounding test unless candidate provably beats champion."""
        result = compare(candidate, champion, n=n, confidence=confidence, seeds=seeds)
        if not result.better:
            pytest.fail(f"lcb-gate {result}")
        return result


@pytest.fixture
def lcb():
    return LcbHelper()
