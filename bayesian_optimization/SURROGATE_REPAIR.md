# Stable, predictive coil-set BO

The `bo_K4_seed67_dcdnizla` investigation reproduced two problems: maximum
likelihood drove the position scale to about `1e-7` metres, leaving distinct
designs almost uncorrelated, and the fused squared-distance formula then lost
precision. Finite likelihood and broad calibration intervals did not detect
that the surrogate was uninformative.

## Kernel arithmetic and uncertainty

`Kernels_v3.CoilSetKernel` now evaluates exact distances with chunked
`torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist')`. Both full and
paired-diagonal calculations use this stable path. The mathematical kernel,
normalization, physical Cartesian mapping and optional RFF mode are retained.
Independent equal-valued inputs retain independent autograd paths.

### CPU performance

The exact coil-set kernel reuses one constant self-covariance matrix during
fixed-geometry fitting. Only the amplitude and constant mean are trained in this
case, so recomputing every coil pair at each likelihood evaluation is unnecessary.
Saved input and scale values are compared on each reuse. Changes to observations,
scales, normalization, dtype or device invalidate the cached result. The cache is
bypassed whenever input or geometry gradients are needed. It is omitted from
checkpoints, and copies protect it from callers modifying returned matrices.

Self-covariance normalization reuses the already computed diagonal. Forward-only
Gaussian calculations reuse temporary distance storage. Small cross-covariances
are evaluated in one operation rather than repeatedly splitting the training
points; this shortcut is limited to 2 MiB per distance matrix, including batches.
Larger calculations retain the existing chunking. Direct distances, float64,
permutation invariance, acquisition restart counts and numerical checks are retained.
RFF remains optional and is not enabled by this optimization.

For `n` observations, `k` coils and `d=6` per-coil coordinates, the first exact
Gram calculation still costs `O(n^2 k^2 d)`. A cache hit costs `O(n k d + n^2)` for
value comparisons and the returned matrix copy, with one extra stored `n*n`
matrix and an input snapshot. At 400 observations in float64 that matrix takes
about 1.22 MiB. GP factorization still costs `O(n^3)`; this does not change the
statistical model or its approximation level.

CPU measurements against commit `f7ef3f2` used four coils, two PyTorch threads,
PyTorch 2.14.1 and synthetic responses, including complete fixed-scale fitting
and covariance checks:

| Observations | Repeated geometry matrix, before / after | Complete GP fit, before / after |
| --- | --- | --- |
| 100 | 2.12 / 0.05 ms | 41.2 / 22.2 ms |
| 400 | 27.71 / 0.09 ms | 234.9 / 64.8 ms |

Fitted posterior means and variances agreed within `5e-16` in these comparisons.
These are local timings, not a promised speedup for complete BO iterations;
simulations, acquisition optimization and regret searches still take time.

To measure the current code on your CPU, from `optimizer_py` in the usual Python
environment, run:

```bash
python benchmark_coil_kernel.py --points 400 --threads 2
```

The benchmark uses synthetic targets, prints timings and launches no particle
simulations or result files. Compare `--threads 1`, `2` and `4` on your machine.
To use two numerical-library threads for a normal run without changing `input.py`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python bo_main.py --method bo --kernel coil-set
```

No C++ rebuild or additional Python package is needed.

`PosteriorChecks.py` checks representative kernel symmetry, positive
semidefiniteness, full/diagonal consistency and normalized unit diagonal. Raw
posterior variance is checked before GPyTorch's reporting floor: material negative
variance or variance exceeding the corresponding prior raises an error. Only
negligible negative roundoff is clamped to zero. Fitting, model diagnostics and
regret use these checks. A numerically invalid regret search cannot count toward
stopping.

## Select geometry by prediction

For vanilla `--kernel coil-set`, `CoilSetModel.py` compares 12 fixed geometry
models: four position scales in metres and three normal concentrations. A local
seeded 80/20 split separates fitting from scale selection. Each model fits only
its amplitude and constant mean on the training fold, with the calibrated
single-run log variance; the lowest prediction RMSE on the selection fold wins.

The selected position scale and concentration remain fixed during subsequent
likelihood fitting and its recovery attempts. Different candidates supply
genuinely different geometry scales without letting unrestricted likelihood
optimization collapse them. Selection repeats after 50 new observations.

Settings in `input.py`:

```python
BO_COIL_LENGTHSCALES = (.25, .5, 1., 2.)  # metres
BO_COIL_KAPPAS = (.25, 1., 4.)
BO_COIL_SELECT_EVERY = 50
BO_COIL_MIN_SELECTION = 25
```

These are modeling choices to validate, not universal physical lengthscale bounds.
If no candidate improves the constant-mean selection baseline, the code saves
observations and stops before further simulations. Runs smaller than 25 points
use 0.5 m and concentration 1 without claiming predictive validation; they cannot
justify regret stopping. The selection score is a tuning diagnostic and is
optimistic after choosing among candidates. Independent validation remains needed.
Standard RBF/Matérn geometry fitting is unchanged.

Model diagnostics record the scales, selection RMSE, constant baseline, selection
skill and calibration checks. Passing a wide calibration interval alone is no
longer enough for the coil-set stopping gate. Checkpoints contain the exact last
successful GP `state_dict` as `gp_state__*` NumPy arrays, plus scale-selection
metadata. Pending observations may outnumber observations in that fitted state;
consult `fitted_evaluations` and `status`.
The next spatial seed is explicit integer state: `optimization_seed_state()` reads
it without advancing, and `set_optimization_seed()` restores it. No iterator
pickling or serialization internals are used, including on Python 3.14.

## Acquisition and regret

### Fast CPU defaults for the coil-set loop

Minute-long updates were largely acquisition and diagnostic searches, rather
than likelihood fitting alone. The coil-set defaults now use:

```python
BO_COIL_ACQUISITION = 'log-ei'
BO_COIL_REFIT_EVERY = 10
BO_COIL_SELECT_EVERY = 50
BO_COIL_RESTARTS = 12
BO_COIL_RAW_SAMPLES = 256
BO_COIL_OPT_MAXITER = 60
```

`log-ei` uses BoTorch's stable analytic single-point log expected improvement,
with the largest posterior mean at evaluated designs as its incumbent. This is
a **plug-in approximation for a noisy incumbent**: it treats that denoised value
as fixed instead of integrating uncertainty in which evaluated design is best.
Calibrated observation noise is still in the GP, and latent means and variances
are still computed exactly. This changes the acquisition rule, not the objective,
particle samples, exact set kernel or log scaling. Set
`BO_COIL_ACQUISITION = 'log-nei'` to retain the previous sampled noisy-improvement
rule. Budgets remain configurable with either rule.

`GPConditioning.py` conditions on **every** accepted observation. Between full
hyperparameter fits it keeps the last fitted mean/covariance parameters and
`Standardize` statistics, transforms both new scores and observation noise with
those same statistics, and clears prediction caches. The next posterior rebuilds
the exact Cholesky factor using all observations. It does not ignore recent data
or reuse old variances. Full fits recompute standardization and refit on all data
every ten observations; geometry-scale reselection also forces a full fit.
An update failure triggers a full-fit recovery attempt. `BO_COIL_REFIT_EVERY = 1`
restores hyperparameter optimization after each observation.

The checkpoint's `fitted_evaluations` counts all observations in the conditioning
data. `hyperparameter_evaluations` records when parameters were last optimized.
When reconstructing a saved GP, use saved `outcome_transform` buffers to transform
the raw checkpoint targets and variances; recomputing fresh statistics without a
refit changes their units relative to the saved parameters.

The coil-set posterior-mean curve now reports the best posterior mean **at
evaluated points**, eliminating a 30-restart global optimization used only for
plotting. This scope is recorded as `posterior_mean_scope` in settings.
Raw variance, calibration, kernel PSD and duplicate-candidate checks remain active.
RBF and Matérn retain their acquisition and refitting behavior.

Dense regret records (every two evaluations by default) use the fixed Sobol grid
plus all observations, with exact marginal variances and no gradient search.
`search_method=grid` identifies these diagnostic gaps, which can underestimate
unexplored UCB peaks. They **never trigger stopping**. If stopping is enabled,
each actual stopping decision still performs the validated multistart UCB search,
records `search_method=optimized`, and requires the existing model and optimizer
checks. `plot_regret.py` distinguishes these two kinds of records.

Iteration console output separates acquisition, model conditioning/refit,
diagnostics and simulation seconds. Timings are also saved in checkpoint
`timings_json` and the final settings' `iteration_timings`; the initial model
selection/fit is timed separately. Final independent high-particle validation is
retained.

In a two-thread CPU benchmark with 100 observations and four coils, acquisition
optimization took 26.84 s for the old 60-restart sampled log-NEI search and 1.79 s
for the new 12-restart analytic search (about 15 times faster). These are synthetic
timing measurements, not a claim that the acquisition choices have equal quality.
The earlier kernel-only benchmark above does not represent whole BO iteration time.

A complete-loop CPU smoke comparison at 100 observations averaged 27.7 s of
overhead per iteration before these changes and 1.75 s afterward. At 400
observations the new loop averaged 2.29 s. These short synthetic runs used real
fits, candidate searches and regret diagnostics, but stubbed the simulator,
calibration diagnostic and initial scale selection identically; they do not
include initial Monte Carlo design/calibration or prove equal optimization quality.
All new-loop proposals and metrics were finite, with no numerical warnings.
To benchmark candidate-search cost locally without launching particle simulations:

```bash
python benchmark_coil_kernel.py --points 100 --threads 2 --acquisition
```

This optional comparison needs the usual compiled-simulator import path because
it uses the physical mapping. It also runs the slower legacy search for comparison.

The q=1 acquisitions reserve restart slots for known observations, perturbations
near promising observations, and global Sobol samples. Both close and broader
normalized perturbation widths (0.05 and 0.15) are used. Cached acquisition scoring
stays batched. The reported posterior optimum cannot fall below the best scored
known point. KG retains its one-shot fantasy initialization; only real candidate
slots are replaced with observed/local seeds.

Observed points remain optimizer starts, but are excluded from final BO proposals
using the same per-coordinate `1e-6` tolerance as the loop's duplicate check.
An optimized duplicate falls back to the best finite unseen pool point. Posterior
mean diagnostics explicitly permit observed points. KG duplicate recovery scores
complete one-shot trees before extracting an unseen real candidate; fantasy
coordinates are preserved. An empty finite unseen pool raises an error instead
of repeatedly consuming iterations on duplicates.

`KGInitialization.py` bounds both raw KG initialization and the internal
posterior-value optimization used to choose fantasy locations. Both stages obey
`BO_OPT_BATCH_LIMIT` and `BO_OPT_INIT_BATCH_LIMIT`, including values below their
defaults, without reducing the requested sample or restart counts.

New `regret_history.csv` columns:

```text
upper_mean, upper_sigma, incumbent_mean, incumbent_sigma, mean_gap, uncertainty_gap
```

With confidence multiplier `c`, the recorded gap decomposes as
`mean_gap + uncertainty_gap`, where
`uncertainty_gap = c*(upper_sigma + incumbent_sigma)`.
`plot_regret.py` writes `regret_components.png` and `.pdf` when these fields are
available. Old reports remain readable. The stopping tolerance is unchanged and
early stopping remains disabled by default.

## Reproduce offline validation

Activate the usual environment, set `PYTHONPATH` to the compiled simulator
directory, and run from `optimizer_py`:

```bash
python validate_surrogate.py \
  --run-dir results/bo_K4_seed67_dcdnizla \
  --output-dir results/my_surrogate_validation
```

The script checks particle count, current, radius, energy bank, bins and parameter
bounds against the saved run. It withholds 20% of the observations from **all**
model fitting and scale selection, selects scales on the remaining data using
its own inner split, then reports outer prediction error, constant baseline,
correlation and predictive coverage including observation noise. No particle
simulations run. This is retrospective validation of historical BO data, not a
new independent sample of geometries.

## Reproduce a short physical pilot

```bash
python run_surrogate_pilot.py \
  --run-dir results/bo_K4_seed67_dcdnizla \
  --output-dir results/my_surrogate_pilot \
  --evaluations 10 --final-particles 100000 --repeats 3
```

The pilot reuses all saved observations at their original optimization particle
count and common energy bank. Fresh spatial seeds start at least at `SEED+500000`
and advance past a saved `next_spatial_seed` when present, within the optimization
seed range. It compares the best **new** design with the
original independently selected winner, using three new simulations each at
100,000 particles per species. Final validation starts at `SEED+3000000` and
refreshes energies and spatial scrambles independently. Reports and checkpoints
go to a new directory; the original results are not overwritten.

The log objective, detector integration, particle physics, common-energy method
and final validation estimator are unchanged. Better numerical validity does not
guarantee improvement in every short pilot or global convergence.
