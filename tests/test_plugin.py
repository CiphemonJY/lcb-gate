"""Fixture integration — pytest only (excluded from the bare-python runner)."""

import pytest

from lcb_gate.plugin import LcbHelper


def test_fixture_is_registered(lcb):
    assert isinstance(lcb, LcbHelper)


def test_check_passes_clean_run(lcb):
    res = lcb.check(lambda i: True, n=25, threshold=0.9)
    assert res.passed


def test_check_fails_flaky_run():
    helper = LcbHelper()
    with pytest.raises(pytest.fail.Exception):
        helper.check(lambda i: i % 2 == 0, n=20, threshold=0.9)


def test_check_better_fails_on_tie():
    helper = LcbHelper()
    with pytest.raises(pytest.fail.Exception):
        helper.check_better(lambda s: 1.0, lambda s: 1.0, seeds=range(50))
