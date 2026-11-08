# Multiple trust regions and independent final validation

Run from `bayesian_optimization/optimizer_py` with the prepared Python environment
and the compiled `simulator` directory on `PYTHONPATH`:

```bash
python bo_main.py
# Explicitly explore five regions with 50 refinement iterations each:
python bo_main.py --regions 5 --iterations 250
```

The driver saves results before offering optional interactive plots (`--plot`).
Each run uses a new directory under `results/`; tracked configuration CSVs are
not overwritten. `--output-dir` can name another **new** directory.

## Selecting how many regions to explore

The initial design is simulated only once. A coil-set GP ranks those points by
posterior mean of the negative log-response. The top quarter (at least twice the
maximum region count) forms the promising candidate pool.

For its fitted kernel correlation matrix `C`, the effective rank is

```
r_eff = trace(C)^2 / sum(C*C)
```

It is near one for redundant geometries and near the pool size for uncorrelated
ones. The automatic count is `ceil(r_eff)`, capped by five, available seeds and
the iteration budget. At least one failure window (`max(4,D)` iterations at Q=1)
is reserved per region when the total budget allows it. With D=50 and the default
MAX_ITER=200, automatic selection therefore starts **at most four regions**.
The metric is a practical diversity heuristic, not proof of the number of optima.

Starting with the best posterior mean, greedy selection chooses the least
correlated remaining promising point. Automatic selection stops when every
remaining point has correlation at least 0.95 with an existing seed. An explicit
`--regions 4` or `--regions 5` overrides this threshold but still removes effectively
identical kernel geometries. A short explicitly requested budget is allowed and
reported; it may finish before a region can shrink.

MAX_ITER / `--iterations` is the **total** refinement budget across runs. For
200 iterations in each of five regions, request `--regions 5 --iterations 1000`.
Runs are sequential because each Monte Carlo already launches species threads.
These are independent multi-start TuRBO runs, not a coupled TuRBO-m scheduler.
Each GP starts with the shared initial observations, then learns from its own
new observations. Its centre follows its own seed/incumbent, never another
region's superior initial point. The existing single-region function still works.

## Final selection

The best observation from each region's own seed/new points and the best original
observation form the shortlist. Duplicate normalized designs are removed, then
up to FINALISTS=5 are kept by observed score. This is a shortlist, not a certificate
of global optimality.

Every finalist is reevaluated automatically with FINAL_PARTICLES=100000 particles
**per species in every Monte Carlo**, using the same eight logarithmic energy
strata. FINAL_REPEATS=3 independent energy/spatial seeds are used by default.
Thus one finalist costs 8.4 million trajectories with the current 28 species.
No optimization observations are reused in these validation averages.

Ranking uses the arithmetic mean of raw responses, not exponentiated mean logs.
Standard errors come from the variance between independent replicate estimates.
Approximate 95% Student-t intervals use `repeats-1` degrees of freedom; three
replicates give a limited uncertainty estimate. `--validation-repeats 5` permits
more replicates. Overlapping intervals are flagged, without claiming a formal
pairwise hypothesis test. These are nominal candidate-wise intervals, not
selection-adjusted confidence intervals for the winner. The within-run stratified variance remains a separate
diagnostic and does not replace the between-scramble error estimate.

Outputs include:

- `initial_design.npz`, `region_selection.json` and one `region_*.npz` history;
- `validation_runs.csv`: individual responses, variances, seeds and particle counts,
  appended after every Monte Carlo so completed evaluations survive interruption;
- `final_validation.csv`: ranked means, standard errors and intervals;
- `finalist_*.csv` and `best_configuration.csv`: physical centres and normals.

For quick smoke runs, `--final-particles` can reduce the validation count. Results
record the actual count; a reduced-budget run is not a 100000-particle validation.

## Spatial scrambling

The C++ Sobol generator retains its interface and sphere mapping. The disabled
scrambling overwrite is removed, and each bit is now flipped using a seeded hash
of its preceding original bits and level: a hash-based nested Owen scramble.
The initial zero index is included to preserve dyadic prefix balance. Dimension,
species and energy-bin seeds are separated, and a supplied objective seed now
controls energy sampling as well as spatial scrambling. The existing XOR-shift
hash alone was not a nested Owen scramble.

The default per-bin count 125 (or 12500 during final validation) is not a power of
two; exact digital-net balance applies to power-of-two prefixes. The requested
particle budget is preserved rather than silently rounded upward.

Run all regression suites after building:

```bash
ctest --test-dir bayesian_optimization/build-cloud --output-on-failure
```
