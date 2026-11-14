# Estimated simple-regret stopping for vanilla Bayesian optimization

The three functions in `optimizer_py/BoLoop.py` now use the same stopping rule:
`bo_matern_kernel`, `bo_rbf_old` (CLI `rbf`) and `bo_rbf_kernel` (CLI `rbf-set`).
It replaces the previous posterior-mean plateau heuristic. TuRBO's stopping
policy, the physical sampler, the logarithmic objective and the final Monte
Carlo validation are unchanged.

## What is being estimated?

Write the positive physical response as S(x). The GP models the objective
f(x) = -log(S(x)), which is maximized. The observations are negative logarithms
of Monte Carlo response estimates, with their existing delta-method variances.

The simple regret of an incumbent x_inc is

```text
r = f(x*) - f(x_inc) = log(S(x_inc) / S(x*))
```

where x* is the unknown global optimum. The true regret is unavailable, so the
code uses the fitted GP's latent posterior mean mu(x) and standard deviation
sigma(x), in original log-response units, to form

```text
U(x) = mu(x) + c * sigma(x)
L(x) = mu(x) - c * sigma(x)
x_inc = argmax L(x_i), over evaluated configurations only
B = max U(x) - L(x_inc), over the whole search domain
```

If these bounds cover the true function throughout the domain, then r <= B.
The implementation approximates the maximization, and therefore records an
**estimated** bound rather than a rigorous global regret certificate.

For a desired relative response tolerance epsilon, it compares B against

```text
delta = log(1 + epsilon)
```

Thus r < delta would mean S(x_inc) < (1 + epsilon) * S(x*). The default epsilon
is 0.001, or 0.1%, giving delta approximately 0.0009995003. This is relative to
the optimum, not a 0.1% improvement over the preceding incumbent.
Rescaling response units adds a constant to mu, U and L and leaves B unchanged.

## How the UCB maximum is searched

The code searches the original normalized design domain, `[0,1]^D`, even when
the GP uses periodic or pooled features. It applies the same feature map as the
BO acquisition, exactly once. BoTorch's `UpperConfidenceBound` receives beta=c^2,
since its convention is mu + sqrt(beta) * sigma.

Each check evaluates UCB on a fixed, seeded, scrambled Sobol grid and all
evaluated configurations. The twenty highest-scoring points, or all points if
there are fewer, initialize multistart continuous UCB optimization. The largest
value among the grid, evaluated configurations and optimized candidate is kept.
Posterior grid evaluation is batched to limit memory use. This search does not
launch any new particle simulations.

Keeping the grid maximum prevents a poor local optimizer result from lowering
the estimate below the sampled values. It does not establish a global maximum.
A numerical error, non-finite result or optimizer warning triggers one bounded
SLSQP recovery attempt. If recovery also fails or warns, the check is invalidated,
the consecutive-pass count resets, and BO continues. Even a finite warned UCB
result cannot trigger stopping. See [OPTIMIZER_STABILITY.md](OPTIMIZER_STABILITY.md)
for batching, baseline caching and checkpoint details.

## Settings in input.py

| Setting | Default | Meaning |
| --- | --- | --- |
| `BO_REGRET_ENABLED` | `True` | Enable early stopping; `False` leaves the hard budget |
| `BO_REGRET_REL_TOL` | `1e-3` | Relative response tolerance epsilon |
| `BO_REGRET_CONFIDENCE` | `2.0` | Multiplier c of latent posterior standard deviation |
| `BO_REGRET_MIN_EVALS` | `5 * D` | Minimum successful new observations before checking |
| `BO_REGRET_CHECK_EVERY` | `D` | Successful new observations between checks |
| `BO_REGRET_PATIENCE` | `3` | Consecutive checks with estimated B strictly below delta |
| `BO_REGRET_SAMPLES` | `1024` | Number of fixed Sobol grid points |

With K=4 and D=20, checks occur after 100, 120, 140, ... successful new
evaluations. The earliest possible early stop is after 140 new evaluations,
in addition to the initial design. A failing check resets the pass count.
Skipped duplicates and invalid observations do not advance this counter.

`MAX_ITER` / `--iterations` remains the hard limit on refinement loop attempts,
excluding initialization and validation. Because skipped attempts still count
toward that hard limit, it may be reached with fewer successful evaluations.
A budget shorter than `BO_REGRET_MIN_EVALS` produces no regret checks.
Neither absence of checks nor reaching the budget is reported as convergence.

## Running and interpreting output

```bash
cd bayesian_optimization/optimizer_py
USE_FEATURE_MAPPING=1 python bo_main.py --method bo --kernel rbf --no-plot
USE_FEATURE_MAPPING=1 python bo_main.py --method bo --kernel matern --no-plot
```

Use the project's Python environment and put the compiled simulator directory
on `PYTHONPATH`, as for previous runs. No C++ rebuild is needed for this change.

The driver writes the following to the run's result directory:

- `regret_history.csv`: successful evaluation count, loop iteration, estimated
  maximum UCB, incumbent LCB/index, estimated regret, delta, search status and
  consecutive-pass count for every eligible check. A failed check records
  infinite regret, `search_ok=False`, and zero consecutive passes.
- `bo_settings.json`: stopping settings, `stop_reason`, successful evaluation
  count, attempted iterations and the final LCB incumbent's index in `bo_history.npz`.
  Stop reasons are `estimated_simple_regret` or `iteration_budget`.
- The existing history and final-validation reports.

The final LCB incumbent is always the first shortlisted finalist, even if its
noisy observed score is not the best. Remaining slots are filled by distinct
configurations with the best observed scores. Finalists are independently
reevaluated with `FINAL_PARTICLES` particles per species and `FINAL_REPEATS`
replicates, then ranked by arithmetic raw-response means as before. The reported
final validation winner may differ from the GP's LCB incumbent; the regret
estimate refers to the latter, not automatically to the validation winner.

Existing direct function calls still return the original five-element tuple.
Passing `return_diagnostics=True` adds a sixth element containing stopping
settings, check history, stop reason and incumbent information; the driver uses
this option to save reports and preserve the incumbent in final validation.

## Statistical limitations

- With c=2, intervals are approximately 95% pointwise Gaussian posterior
  intervals under the fitted GP, not simultaneous intervals over the domain
  or over repeated checks. Patience does not change that into a formal guarantee.
- Kernel/noise misspecification and empirical hyperparameter fitting can make
  posterior uncertainties inaccurate. The supplied within-run Monte Carlo
  variances are approximate, including the existing log-response delta method.
- The finite Sobol grid and local maximizations can miss a higher UCB region,
  especially in high dimensions. Increasing grid size/restarts can improve the
  search but does not certify it globally.
- Pooled `rbf-set` features discard geometry; adequate representation of the
  true response is an additional modeling assumption.
- High remaining uncertainty prevents stopping even if the posterior mean
  appears flat. The default 0.1% tolerance may be unresolved at the chosen
  particle budget, in which case the hard budget is the expected termination.
- Independent final validation assesses the selected configurations' responses
  and ranking. It does not establish global optimality or numerical discretization
  accuracy of the trajectory simulator.
- With [common optimization energies](COMMON_RANDOM_ENERGIES.md), the GP is
  conditional on that finite bank. Spatial rescrambling alone does not assess
  its energy-integration error; final validation refreshes both energies and space.

For a thesis, call this an **estimated GP simple-regret stopping criterion**.
It does not compute cumulative regret or prove sublinear cumulative-regret growth.

## Verification

The `vanilla_bo` CTest suite includes analytic checks of the UCB/LCB formula and
invariance to log-response offsets; actual mapped-GP UCB optimization; minimum
evaluation count, cadence, patience and failed-check handling; rejection of
mean-only convergence while uncertainty remains; stopping integration for all
three loops; and preservation of the LCB incumbent in final validation.

Reduced-particle integration runs with the real C++ simulator also exercised
both periodic RBF and Matérn, including UCB checks, report saving and independent
final validation. Test-only settings used six initial configurations, 32 particles
per species during search, 64 during validation and checks after each new
evaluation. A separate deliberately loose-tolerance RBF run verified early exit.
These verify execution paths, not convergence or physical accuracy at production
settings; the defaults above were retained in `input.py`.

```bash
# From the repository root, using your configured build directory:
ctest --test-dir bayesian_optimization/build-cloud --output-on-failure
```
