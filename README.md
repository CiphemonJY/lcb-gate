# lcb-gate

[![ci](https://github.com/CiphemonJY/lcb-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/CiphemonJY/lcb-gate/actions/workflows/ci.yml)

**Statistical pass/fail gating for stochastic tests.** Run the trial N times;
pass only when the **Wilson lower confidence bound** of the pass rate clears
your threshold. Built for agent evals — anything where a single green run
proves almost nothing. Stdlib only, no dependencies.

```
pip install lcb-gate
```

The demo ships in the repo (not the wheel):

```
git clone https://github.com/CiphemonJY/lcb-gate && cd lcb-gate
python examples/demo.py     # why a single green run lies, in three acts
```

## The problem

Agent behaviors, LLM evals, and flaky integration tests are *stochastic*. CI
treats them as deterministic: one green run → merged. Two failure modes follow:

- **The lucky pass.** A 75%-reliable agent task passes a single CI run 75% of
  the time. You ship a coin flip and call it tested.
- **The winner's curse.** You try 20 variants, one "beats the baseline" on a
  single eval run, and you promote it. Most single-run wins are noise: in one
  production self-improvement pipeline, fresh-seed re-verification killed
  **85% of apparent improvements** found by a first-pass scan.

Averages don't fix this — a point estimate without n is a vibe. What fixes it
is a *lower confidence bound*: "with 95% confidence the true pass rate is at
least X". Gate on X.

## Usage

### As a library

```python
from lcb_gate import run_gate

result = run_gate(lambda i: run_my_agent(seed=i).succeeded, n=25, threshold=0.9)
print(result)
# PASS: 25/25 passed (gate n=25); LCB 0.902 >= 0.9 @ 95% confidence
assert result.passed
```

`run_gate` stops early — in either direction — the moment the verdict is
mathematically settled, so hopeless runs don't burn the full budget. The
verdict is identical to running all n trials.

### As a pytest fixture

```python
def test_agent_completes_checkout(lcb):
    lcb.check(lambda i: run_agent_eval(seed=i).ok, n=25, threshold=0.9)
```

The plugin registers automatically on install. Failures explain themselves:

```
Failed: lcb-gate FAIL: 21/30 passed (gate n=30); LCB 0.551 < 0.9 @ 95% confidence
```

### Candidate vs champion (promotion gates)

"Is the new prompt/model/policy actually better?" is the same statistics with
a different threshold — the win-rate LCB must exceed 0.5:

```python
from lcb_gate import compare

result = compare(candidate=eval_new, champion=eval_old, n=400)
print(result)
# BETTER: candidate won 243.0/400 paired trials (60.8%); win-rate LCB 0.567 > 0.5 @ 95%
```

Both callables receive the **same seed per trial** (common random numbers):
pairing removes the noise both variants share, which is where most of the
statistical power comes from. Ties count as half a win. The honest default
verdict is `NOT PROVEN` — an underpowered comparison never certifies.

```python
def test_new_prompt_beats_production(lcb):
    lcb.check_better(eval_new, eval_old, n=400)
```

## Sizing your gate

A perfect record at small n still can't clear a high bar — by design. Minimum
trials for a *flawless* run to pass, at 95% confidence:

| threshold | min n (all passing) |
|-----------|---------------------|
| 0.80      | 11                  |
| 0.90      | 25                  |
| 0.95      | 52                  |
| 0.99      | 268                 |

`min_trials(threshold)` computes this. If your eval budget can't afford the
n, lower the threshold honestly rather than pretending n=5 certifies 99%.

## CI for agents — the full stack

This gate answers "does the behavior hold up statistically **across runs**?"
Its sibling project [grounding-gate](https://github.com/CiphemonJY/grounding-gate)
answers "was each claim structurally grounded **within a run**?" Together:

1. **Per-trace, structural**: grounding-gate rejects terminals whose claims
   were never observed (zero tokens, zero LLM calls).
2. **Across-runs, statistical**: lcb-gate runs the eval N times and certifies
   the pass rate's lower bound.
3. **On promotion**: `compare()` with paired seeds gates champion swaps.

A GitHub Actions job needs nothing special — the gate lives inside the tests:

```yaml
- run: pip install lcb-gate
- run: pytest tests/agent_evals -q     # each test is an N-run certified gate
```

## Roadmap

- Sequential probability ratio test (SPRT) mode — even fewer trials at the
  same error rates.
- `@pytest.mark.lcb(n=..., threshold=...)` marker API.
- JSON report artifact for tracking LCBs over time.

## License

MIT
