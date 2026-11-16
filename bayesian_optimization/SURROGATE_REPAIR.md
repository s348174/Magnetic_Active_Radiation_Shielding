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
optimization collapse them. Selection repeats after 20 new observations.

Settings in `input.py`:

```python
BO_COIL_LENGTHSCALES = (.25, .5, 1., 2.)  # metres
BO_COIL_KAPPAS = (.25, 1., 4.)
BO_COIL_SELECT_EVERY = 20
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
