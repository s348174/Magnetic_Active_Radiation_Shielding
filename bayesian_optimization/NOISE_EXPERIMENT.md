# Standalone common-energy noise experiment

`optimizer_py/run_noise_experiment.py` evaluates fixed coil configurations using
one common energy bank and repeated fresh spatial Owen scrambles. It does not
run a GP, acquisition optimizer or BO stopping rule. Existing optimization loops,
input settings, detector integration and trajectory code are unchanged.

## Run it

Activate the same Python environment and set the simulator import path as for
`bo_main.py`. For example, from the repository root, with a `build-linux` build:

```bash
source /path/to/your/pyenv/bin/activate
cd bayesian_optimization/optimizer_py
export PYTHONPATH="$PWD/../build-linux${PYTHONPATH:+:$PYTHONPATH}"

# Replace this directory with your completed run's results directory:
python run_noise_experiment.py --run-dir "results/bo_K4_seed67_YOUR_RUN" --repeats 6
```

Use `../build-cloud` instead of `../build-linux` if that is your configured build
directory. No C++ rebuild is required. Run the launcher from `optimizer_py` because
the existing simulator and input data paths are relative to that directory.

The particle count defaults to N in `input.py`; the common energy seed defaults
to BO_ENERGY_SEED. With the same spectrum, count and seed, the energy arrays
match the common bank used by optimization. Start with these defaults to diagnose
your original run. Periodic features are irrelevant here: no surrogate is fitted.

For a cheaper preliminary check:

```bash
python run_noise_experiment.py --run-dir "results/bo_K4_seed67_YOUR_RUN" \
    --particles 256 --repeats 4
```

Changing the particle count changes the energy bank. The preliminary check
therefore measures a different conditional response from the original N=1000 run.
To run without saved results, omit `--run-dir`; the reference will be a generated
example rather than your saved optimum. `python run_noise_experiment.py --help`
lists all options. `--output-dir` must name a new directory.

For a second-bank sensitivity check, repeat the same experiment with
`--energy-seed 68`. Comparing banks is separate from estimating spatial noise.

## Configurations

- `reference`: `best_configuration.csv` from the supplied run, when available;
  otherwise the best observed score in `bo_history.npz` or `bo_checkpoint.npz`.
  Without saved results, use generated middle-radius coils with radial normals.
- `best_initial`: best stored negative-log score in `initial_design.npz`, if
  available and different from the reference.
- `near_radial`: centres spread over a sphere at the minimum allowed radius,
  with radial normals.
- `far_aligned`: centres at the maximum allowed radius, with parallel normals.
- `zero_current`: reference geometry with I=0. It is a diagnostic baseline;
  the detector and its numerical behavior are not modified or corrected.

Duplicate field configurations are skipped. The generated near/far cases use
the bounds in `input.py` and a deterministic spherical distribution. Saved
designs must match the current K and parameter bounds; Cartesian saved CSVs
must have K coil rows. Use the original run's physical settings in `input.py`.

Without duplicates, a completed run supplies up to five cases. Six scrambles
therefore cost at most 30 simulations, without the original optimization's
hundreds of evaluations. The same energy arrays are reused throughout. Each
simulation gets a distinct spatial seed, starting at SEED+2000000 by default.

## Reports

Each launch creates a separate `results/noise_experiment_K...` folder:

- `settings.json`: physical parameters, energy/spatial seeds, particle count,
  incident energy and configuration sources.
- `configurations/*.csv`: the actual Cartesian centres and normals used.
- `runs.csv`: every simulation's response, current, seeds, legacy within-run
  variance, legacy GP log variance, and relative received-energy score. Appended
  and flushed after each simulation so completed rows survive interruption.
- `summary.csv`: per-configuration mean, spatial-scramble standard deviation,
  standard error of the mean, nominal 95% Student-t interval, empirical log
  variances and comparison with the variance currently supplied to vanilla BO.
  Updated after each complete configuration.
- `comparisons.csv`: differences against the reference and against zero current,
  with standard errors, nominal Welch-t intervals and relative differences in ppm.
  Negative differences mean less received energy than the named reference.

For responses S_1,...,S_R at one configuration:

```text
empirical_log_variance_single = sample variance of log(S_r)
mean_se = sample standard deviation of S_r / sqrt(R)
empirical_log_variance_mean = sample variance of S_r / (R * mean(S_r)^2)
legacy_gp_log_variance = mean(legacy_within_variance_r / (S_r^2 + 1e-6) + 1e-6)
```

The legacy-to-empirical ratio compares single-evaluation variances, not the noise
of one evaluation against the smaller uncertainty of a replicate average. An
infinite ratio means no variation was resolved in the recorded log-responses;
it does not establish zero physical error. No artificial variance floor is added
to the experimental estimate.

The received percentage uses the same flux-weighted incident-energy calculation
as optimization. It remains a smoothed detector score and is not clipped at 100%.
No percentages or baselines replace the total-energy response.

## Interpretation

Compare the spread of configuration means with their between-scramble errors.
If differences are resolved but the legacy variance is much larger, this supports
correcting the GP noise model. If configuration differences are unresolved, more
scrambles or a larger particle budget may be needed. If differences remain tiny
and well resolved, the tested configurations may truly have similar energy scores.

Intervals are exploratory, conditional on the fixed energy bank and the existing
discretization. They are approximate, unadjusted for multiple comparisons, and do
not measure detector/trajectory integration bias or finite-bank integration error.
The zero-current result is reported as measured; the experiment does not repair
its deviation from 100%. Four/five cases cannot establish a globally flat response.

The launcher and this guide are separate from regression tests. Generated output
stays under the ignored `results/` directory; existing test files are untouched.
