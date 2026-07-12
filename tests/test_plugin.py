"""Fixture integration — pytest only (excluded from the bare-python runner)."""

import random

import pytest

from lcb_gate.plugin import LcbHelper


def _judge(p, tag):
    def f(seed):
        return 1.0 if random.Random(f"{tag}:{seed}").random() < p else 0.0
    return f


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


def test_check_score_passes_clean_run(lcb):
    res = lcb.check_score(lambda i: 0.95, n=100, threshold=0.8)
    assert res.passed


def test_check_score_fails_low_mean():
    helper = LcbHelper()
    with pytest.raises(pytest.fail.Exception):
        helper.check_score(lambda i: 0.4, n=50, threshold=0.8)


def test_check_best_picks_winner(lcb):
    res = lcb.check_best([_judge(0.9, "A"), _judge(0.4, "B")], n_max=4000, seeds=range(1500))
    assert res.proven


def test_check_best_fails_on_tie():
    helper = LcbHelper()
    with pytest.raises(pytest.fail.Exception):
        helper.check_best([_judge(0.6, "p"), _judge(0.6, "q")], n_max=400, seeds=range(200))
