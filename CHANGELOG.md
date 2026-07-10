# Changelog

## 0.1.0 — 2026-07-10

Initial release.

- `wilson_lcb` / `min_trials` — one-sided Wilson score lower bound and gate sizing.
- `run_gate` — N-run pass/fail gating with settled-verdict early stopping.
- `compare` — paired champion/candidate comparison (common random numbers,
  ties count as half a win), verdict on the win-rate LCB vs 0.5.
- pytest plugin: `lcb` fixture (`check`, `check_better`) via the `pytest11`
  entry point.
