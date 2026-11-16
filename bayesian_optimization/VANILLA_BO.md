# Vanilla Bayesian optimization

Run from `bayesian_optimization/optimizer_py`, using the prepared Python
environment and the compiled simulator directory on `PYTHONPATH`.

```bash
# Existing default: multiple TuRBO regions with the coil-set kernel.
python bo_main.py --method turbo --regions 1 --iterations 200

# Single global BO loop with a standard kernel.
python bo_main.py --method bo --kernel matern --nu 2.5 --iterations 200
python bo_main.py --method bo --kernel rbf --iterations 200
python bo_main.py --method bo --kernel rbf-set --iterations 200
python bo_main.py --method bo --kernel coil-set --iterations 200
```

Without `--method`, the driver still runs TuRBO. In BO mode the default kernel is
Matérn with `nu=2.5`; supported smoothness values are 0.5, 1.5 and 2.5.
`--regions` only applies to TuRBO; `--kernel` and `--nu` only configure vanilla BO.
The existing TuRBO kernel and stopping policy are unchanged. Your current
`TURBO_MAX_REGIONS=1` setting is respected; increase that limit in `input.py` to
request multiple regions. Plots remain enabled by default; `--no-plot` disables
them for headless terminals.

| CLI kernel | Existing function | GP features | Acquisition |
| --- | --- | --- | --- |
| `matern` | `bo_matern_kernel` | Normalized parameters, or the existing optional periodic map | Knowledge gradient, two fantasies |
| `rbf` | `bo_rbf_old` | Same map as Matérn | Log noisy expected improvement |
| `rbf-set` | `bo_rbf_kernel` | Mean, population standard deviation and maximum of Cartesian coil features | Log noisy expected improvement |
| `coil-set` | `bo_coil_set_kernel` | Cartesian position and unit normal for every coil, via `map_coilwise` | Log noisy expected improvement |

`coil-set` uses the existing **exact `CoilSetKernel` in `Kernels_v3.py`**, wrapped
in `ScaleKernel`. Stable direct distances preserve its mathematical definition. It sums pairwise
coil similarities using an RBF on positions and a von Mises--Fisher factor on
unit normals, then normalizes the configuration kernel to unit diagonal before
applying the learned output scale. Position lengthscale and normal concentration
(`kappa`) are shared across coils, preserving permutation invariance. The random
Fourier feature approximation and `CoilMomentKernel` are not used.
Position scales and concentrations are selected by predictive validation and then
fixed during likelihood fitting; amplitude and mean remain trainable. See
[SURROGATE_REPAIR.md](SURROGATE_REPAIR.md) for the scale grid, predictive safeguards,
restart strategy, regret decomposition, saved GP states and reproducible validation.

The acquisition optimizer still searches normalized `5*K` parameters. The GP
always receives contiguous Cartesian `6*K` coil features through `map_coilwise`;
this includes the physical angular conversion. `USE_FEATURE_MAPPING` is ignored
for `coil-set`, so do not add the optional sine/cosine map. For example:

```bash
python bo_main.py --method bo --kernel coil-set
```

The same mapping is used for training, acquisition values and gradients, posterior
diagnostics, calibration checks and regret searches. `--nu` remains specific to
Matérn and is rejected for `coil-set`. Exact coil-set kernel evaluation costs
approximately `O(n^2*K^2)` for `n` designs; existing chunking and acquisition batch
limits remain active. Dense GP factorization still costs `O(n^3)`.

The original feature maps and acquisition choices are retained. Consequently,
these runs compare complete BO configurations, not kernels in isolation. Two KG
fantasies give a cheap but variable Monte Carlo approximation. The pooled
`rbf-set` features are permutation invariant but lossy: different arrangements
can share the same pooled features. The `rbf` option avoids that pooling.

The standard kernels use ARD; the custom coil-set kernel uses its shared
lengthscale and concentration. All have an output scale. A shared loop fits
`SingleTaskGP` using the negative log-response and calibrated single-run noise
(`train_Yvar`). `Standardize` transforms both consistently. See
[CALIBRATED_GP_NOISE.md](CALIBRATED_GP_NOISE.md) for automatic repeated-scramble
calibration, model checks and robust fitting. The independent final validation is retained.
The physical flux sampling and objective's logarithmic response are unchanged.
Optimization uses [common energy draws with fresh spatial Owen scrambles](COMMON_RANDOM_ENERGIES.md);
final validation refreshes both. Each evaluation prints a weighted relative
received-energy diagnostic without changing the total-energy objective.

`Q=1` is required. `INIT` and `N` in `input.py` control initial configurations and
optimization particles per species. `MAX_ITER`, or `--iterations`, bounds the
single loop's refinement iterations, excluding initialization and validation.
The [estimated simple-regret criterion](SIMPLE_REGRET_STOPPING.md) is recorded for
all vanilla kernels. Early stopping defaults to disabled; when enabled, it also
requires the repeat-based model checks to pass. Duplicate candidates
and invalid observations are skipped, so iterations need not equal evaluations.
The estimated criterion is not a mathematical convergence guarantee.

Only finite responses with finite nonnegative variances are appended; inputs,
responses and variances always advance together. Acquisition diagnostics are
saved before refitting the GP, in native units: KG for Matérn, log-NEI for RBF.
KG subtracts the optimized current posterior mean. Stopping uses the estimated
maximum UCB minus the best evaluated LCB in original log-response units.

BO output uses a new directory under `results/`, or a **new** `--output-dir`:

- `initial_design.npz`: the initial observations, evaluated once;
- `bo_settings.json`: kernel, acquisition, seed and requested refinement budget;
- `bo_history.npz`: all observations and acquisition/posterior diagnostics;
- `bo_checkpoint.npz`: atomically saved before fitting each completed observation,
  then updated with posterior diagnostics after a successful update;
- `regret_history.csv`: estimated simple-regret checks and consecutive-pass counts;
- the same final-validation CSVs and configuration files as TuRBO.

The final LCB incumbent is included first, followed by distinct evaluated
configurations with the best observed scores, up to `FINALISTS` in total.
They are independently rechecked and ranked by raw response. Validation still uses
`FINAL_PARTICLES=100000` particles per species and `FINAL_REPEATS=3` by default.
For a smoke test, first reduce `N` and `INIT` in `input.py`, then for example run:

```bash
python bo_main.py --method bo --kernel rbf --iterations 2 \
    --final-particles 64 --validation-repeats 2 --no-plot
```

Reduced-budget results check execution, not optimization quality. Changing these
Python options requires restarting Python, but no C++ rebuild.

See [OPTIMIZER_STABILITY.md](OPTIMIZER_STABILITY.md) for memory controls,
numerical-warning recovery and checkpoint behavior.
