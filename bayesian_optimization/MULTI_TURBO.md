# Multiple trust regions and independent final validation

`bo_main.py` also supports `--method bo` for the standard-kernel single-loop
routines; see [VANILLA_BO.md](VANILLA_BO.md) for kernel choices and examples.
The default remains `--method turbo`.

Run from `bayesian_optimization/optimizer_py` with the prepared Python environment
and the compiled `simulator` directory on `PYTHONPATH`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python bo_main.py --method turbo
# 12 regions, with at least 70 evaluations available per region if all stagnate:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python bo_main.py --method turbo --regions 12 --iterations 840
# Explore fifteen regions under one total budget:
python bo_main.py --method turbo --regions 15 --iterations 1050
```

The driver saves results before displaying interactive plots. `--no-plot`
disables them for headless runs. `input.py` defaults to twelve regions and permits
up to fifteen. Smaller explicit counts remain available for quick tests.
Each run uses a new directory under `results/`; tracked configuration CSVs are
not overwritten. `--output-dir` can name another **new** directory.

## Selecting how many regions to explore

The initial Sobol design is simulated only once. Points are ranked directly by
their observed negative log-response (lower received energy is better). The top
quarter, or at least twice the maximum region count, forms the promising pool:
with 100 initial points and a maximum of 15 regions this is the best 30 points.
A fixed permutation-invariant coil kernel with position scale 0.5 m and normal
concentration 1 supplies a geometry correlation. No GP fit is needed just to
choose starting points, and fitted lengthscale collapse cannot destroy this
diversity metric. The fixed scales are a selection heuristic, not a fitted model.

For this fixed kernel correlation matrix `C`, the effective rank is

```
r_eff = trace(C)^2 / sum(C*C)
```

It is near one for redundant geometries and near the pool size for uncorrelated
ones. `TURBO_REGIONS=12` fixes the default requested count. Set it to zero or use
`--regions 0` to recover automatic `ceil(r_eff)` selection, capped by
`TURBO_MAX_REGIONS`, available seeds and the budget for one ten-evaluation
failure window per region. An explicit region count needs at least one budget
iteration per region.
The metric is a practical diversity heuristic, not proof of the number of optima.

Starting with the best observed score, greedy selection chooses the least
correlated remaining promising point. Automatic selection stops when every
remaining point has correlation at least 0.95 with an existing seed. An explicit
`--regions 10`, `12` or `15` overrides this threshold but still removes effectively
identical kernel geometries. A short explicitly requested budget is allowed and
reported; it may finish before a region can shrink.

MAX_ITER / `--iterations` is the **total** refinement-attempt budget across runs.
Before each region starts, the remaining budget is divided among remaining
regions, rounding up for the next region. If it terminates early, its unused
allowance is available to the remaining regions. No new region is restarted
inside a terminated region. The requested budget is an upper limit; all selected
regions can terminate before it is exhausted. Duplicate/invalid attempts still
consume iteration budget but do not increment the stagnation counter.
Runs are sequential because each Monte Carlo already launches species threads.
These are independent multi-start TuRBO runs, not a coupled TuRBO-m scheduler.
Each GP starts with the shared initial observations, then learns from its own
new observations. Its centre follows its own seed/incumbent, never another
region's superior initial point. The existing single-region function still works.

## Fast shrinking and termination

```python
TURBO_REGIONS = 12
TURBO_MAX_REGIONS = 15
TURBO_LENGTH_INIT = .8
TURBO_FAILURE_TOLERANCE = 10
TURBO_LENGTH_MIN = .01
TURBO_CANDIDATES = 512
```

The scalar length is the side length of a normalized hypercube, clipped to
`[0,1]^D`. Every ten consecutive accepted observations without a **strict new
observed best in that region** halve it, independently of D. An improvement resets
the failure window. This replaces the old offset-dependent success test
`1e-3*abs(-log(response))`, which could classify real improvements as failures.
Ten consecutive improvements still double the length, up to 1.6.

When length becomes strictly less than 0.01 the region terminates. Its last
observation remains in the history and finalist shortlist, but its GP is not
refitted again; the terminal posterior-plot metrics are NaN rather than stale
predictions. `stop_reason`, `length_history`, `final_length`, attempted and
successful evaluation counts, and `final_model_evaluations` are saved in each
`region_*.npz`. A length exactly equal to 0.01 does not terminate the region.

Seven halvings take 0.8 to 0.00625: **70 evaluations with no improvement**.
Thus 10/12/15 fully stagnant regions need at least 700/840/1050 evaluations to
all reach the threshold. The unchanged default `MAX_ITER=500` gives twelve
regions approximately 41--42 iterations each; these may stop by budget around
length 0.05. Improvements and expansions may require more than 70 evaluations.
Budget exhaustion and the length threshold are separate termination reasons.

This is an aggressive multistart variant, not the dimension-dependent schedule
from the original TuRBO algorithm. For D=20, `0.8^20=0.01153` before boundary
clipping, but width 0.8 still spans 80% of each coordinate. Halving the width
divides volume by `2^20`, over a million: volume alone does not establish whether
a region is physically local. Noise can cause false record improvements and
premature rejection; this schedule does not certify local or global convergence.
Independent final Monte Carlo validation remains important.

For CPU practicality, Thompson sampling uses a configurable 512-point Sobol
candidate pool instead of the previous approximately 4000-point joint draw.
This is a coarser finite candidate search; increase `TURBO_CANDIDATES` for greater
resolution at higher memory/factorization cost. The local GP and kernel remain
unchanged. Posterior diagnostics use the region's own incumbent and observations,
with checked raw variances, removing the old 60-restart global search used only
to draw a posterior-mean curve.

## Final selection

The best observation from each region's own seed/new points and the best original
observation form the shortlist. Duplicate normalized designs are removed, then
up to FINALISTS are kept by observed score. This is a shortlist, not a certificate
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

- `initial_design.npz`, `region_selection.json` (including requested, used and
  unused global budget) and one `region_*.npz` history;
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
species and energy-bin seeds are separated. Optimization reuses common energy
draws while changing spatial scrambles; final validation refreshes both. See
[COMMON_RANDOM_ENERGIES.md](COMMON_RANDOM_ENERGIES.md). The existing XOR-shift
hash alone was not a nested Owen scramble.

The default per-bin count 125 (or 12500 during final validation) is not a power of
two; exact digital-net balance applies to power-of-two prefixes. The requested
particle budget is preserved rather than silently rounded upward.

Run all regression suites after building:

```bash
ctest --test-dir bayesian_optimization/build-cloud --output-on-failure
```
