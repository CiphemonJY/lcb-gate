# Changelog

## 0.2.0 — 2026-07-12

- `eb_lcb` / `eb_ucb` — one-sided empirical-Bernstein bounds on the mean of
  bounded [0, 1] observations (Maurer & Pontil 2009); the variance-adaptive
  analog of Wilson for continuous scores.
- `score_gate` / `ScoreGateResult` — gate on the mean of a continuous score
  trial (`trial(i) -> float in [0, 1]`); pass iff the EB mean-LCB clears the
  threshold, with settled-verdict early stopping and [0, 1] validation.
- `rank` / `RankResult` / `RankEstimate` — multiplicity-honest best-of-K racing
  (LUCB-style with EB intervals, common random numbers, union + anytime
  confidence split); honest `NOT PROVEN` default.
- pytest plugin: `lcb.check_score` and `lcb.check_best` helpers.
- README: 'Gating an LLM verifier' section (motivation: arXiv:2607.05391).

## 0.1.0 — 2026-07-10

Initial release.

- `wilson_lcb` / `min_trials` — one-sided Wilson score lower bound and gate sizing.
- `run_gate` — N-run pass/fail gating with settled-verdict early stopping.
- `compare` — paired champion/candidate comparison (common random numbers,
  ties count as half a win), verdict on the win-rate LCB vs 0.5.
- pytest plugin: `lcb` fixture (`check`, `check_better`) via the `pytest11`
  entry point.
