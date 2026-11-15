# Calibrated noise and guarded regret stopping

This branch corrects the observation noise used by vanilla RBF, periodic RBF,
pooled-feature RBF and Matérn BO. The objective remains **negative log received
energy**. Common optimization energies, fresh spatial Owen scrambles, detector
integration, C++ trajectories and independent final validation are unchanged.
TuRBO retains its existing noise handling.

## Run

Use the same Python environment and simulator search path as before. From
`bayesian_optimization/optimizer_py`, for a build in `../Release`:

```bash
export PYTHONPATH="$PWD/../Release${PYTHONPATH:+:$PYTHONPATH}"
USE_FEATURE_MAPPING=1 python bo_main.py --method bo --kernel rbf
```

Replace `../Release` with your actual build directory if needed. There is no
C++ change to rebuild. Use `--kernel matern` or `--kernel rbf-set` for the other
vanilla kernels; both use the same calibration and model checks.

## 1. Measure the noise that changes between evaluations

After the initial design, the code repeats three active-field configurations:
the best initial observation, near radial coils and far aligned coils. Identical
geometries are deduplicated. `BO_NOISE_REPEATS = 6` in `input.py` means up to
18 additional simulations per run, at the same `N`, current, coil radius,
energy bins and common energy seed as optimization. These simulations are
additional to the initial and refinement budgets.

Each repeat uses a fresh spatial seed starting at `SEED + 2000000`; these do not
consume BO seeds and are separate from final validation. For each configuration,
the sample variance of its log responses is computed with `ddof=1`. Their
within-configuration variances are averaged (equal replicate counts give equal
degrees of freedom). Between-configuration differences are never pooled as noise.
The zero-current baseline is not used.

This one fixed variance describes a **single simulation**, not a repeat average.
It replaces the old particle-based variance for every initial and new vanilla
observation. A numerical floor of `1e-12` replaces the old additive `1e-6` and
original-unit clamp. Outcome standardization still transforms both targets and
noise, and GPyTorch's standardized noise floor is reduced to `1e-12` so it does
not silently undo the calibration. Particle-based variances remain in the
calibration report for comparison.

## 2. Start a fresh GP and check its predictions

Every run starts with a fresh kernel, even if `_bo_loop` is called with saved
observations through `initial_data`. Previous particle variances are replaced;
kernel parameters are fitted normally and may warm-start between iterations
within this new run. There is no imposed physical signal-amplitude floor.

Calibration repeat observations are **not appended to GP training**. After the
initial fit and each subsequent fit, independent repeat means are compared with
predictions. The comparison uses mean negative log responses, matching the GP's
target, and combines latent GP variance with each repeat mean's variance
(`single-run sample variance / repeats`).

Checks cover absolute means and statistically resolved pairwise differences.
Contrasts use the joint GP covariance, so a common uncertain offset cannot hide
a missed configuration difference. Conservative Bonferroni Student-t thresholds
with `repeats - 1` degrees of freedom account for the small calibration sample
and several comparisons. This is a practical model diagnostic, not exact
frequentist coverage or a global-optimality certificate. Three configurations
cannot validate the entire search space. Six repeats also give uncertain noise
estimates; increase `BO_NOISE_REPEATS` if the calibration varies substantially.

## 3. Keep regret diagnostic until the corrected model is checked

`BO_REGRET_ENABLED = False` is the default for the first corrected run. BO uses
its full refinement budget while continuing to record regret every
`BO_REGRET_RECORD_EVERY` evaluations (default 2).

After reviewing the diagnostics, set `BO_REGRET_ENABLED = True` to restore
stopping. A failed model check always blocks stopping and immediately resets
patience, including between regret checks. A numerically small UCB-LCB gap alone
can therefore no longer stop a model that misses the measured contrasts.
The existing tolerance, minimum budget and passing-check count are unchanged.
Passing the calibration checks remains a necessary diagnostic, not proof of
global accuracy. Changing the log objective is unnecessary to fix the noise.

## Saved reports

- `noise_calibration.json`: physical settings, pooled variance, individual
  variances, held-out means, geometries and every calibration repeat/seed.
- `model_diagnostics.json`: updated after every fit, including the initial fit;
  contains kernel outputscale, signal variance in original log units, supplied
  noise, predicted calibration range, residual/contrast checks and failure reasons.
- `initial_design.npz` and `bo_checkpoint.npz`: calibrated training variances.
- `regret_history.csv`: adds `model_ok`; numerical `below_threshold` still means
  the gap is small, while `model_ok=False` blocks a passing stopping check.

`plot_regret.py` continues to support old reports. New plots mark estimates from
failed model checks with red crosses, and show when early stopping is disabled.
The independent high-particle-count finalist validation remains a separate stage
with fresh energy and spatial seeds.

## GP fitting recovery and checkpoints

Vanilla RBF and Matérn share a bounded fitting sequence:

1. Fit with warm-start L-BFGS-B: at most 300 iterations and 50 line-search steps
   per iteration, with one attempt from this initialization.
2. On failure, reset the kernel lengthscales/outputscale to 1 and the standardized
   constant mean to 0, then try L-BFGS-B again with the same budget. This supplies
   a genuinely fresh start even though the custom kernels have no priors.
3. On failure, reset again and use Adam with learning rate 0.05 and at most
   300 steps. Adam does not require the line search behind the `ABNORMAL` warning.

Every accepted fit must have finite parameters and finite training marginal
likelihood. Adam may stop at its budget; acceptance is not a claim of optimizer
convergence or model accuracy. The repeat-based model checks and regret safeguards
still apply. Failed attempts never supply acquisition predictions. If all methods
fail, the incoming parameters are restored and the run stops with a nonzero exit
status and a concise message giving the checkpoint path. Finalist validation is
not launched from a failed fit. Physical settings and calibrated noise are not
changed to make fitting succeed.

`bo_checkpoint.npz` is now saved atomically before the initial fit and immediately
after every valid simulation, before refitting. Its new scalar fields are:

- `status`: `fit_pending`, `ready`, or `model_fitting_failed`.
- `evaluations`: new valid simulations completed, excluding the initial design.
- `fitted_evaluations`: new observations included in the last successful fit.
- `iteration`: attempted BO iteration; `initial_evaluations`: initial design size.
- `error`: fitting failure text, or an empty string.

For a pending/failed fit, the new observation and acquisition value are present;
the corresponding posterior mean and variance are `NaN` until computed. Earlier
regret and model-diagnostic entries still refer only to earlier completed updates.
`plot_regret.py` handles missing variance values as gaps. Checkpoints can be read
with `np.load(path, allow_pickle=False)`; no Python objects are serialized.

Restarting the launch command still creates a new run; automatic resume is not
implemented. Preserved `X`, `y` and `variance` can be supplied through the existing
loop `initial_data` API, with the same physical settings and common energy bank.
