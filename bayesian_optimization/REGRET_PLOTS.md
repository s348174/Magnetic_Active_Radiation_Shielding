# Visualizing estimated simple-regret convergence

`optimizer_py/plot_regret.py` plots existing run reports without changing the BO
loops, fitting another GP or launching particle simulations. It needs NumPy,
pandas and Matplotlib, but does not import `input.py`, PyTorch or the simulator.
An older four-coil run can therefore be plotted while current input settings
describe six coils or a different current.

```bash
# In your usual Python environment, from optimizer_py:
python plot_regret.py --run-dir "results/bo_K6_seed67_YOUR_RUN"
# The folder name alone also works for optimizer_py/results/:
python plot_regret.py --run-dir "bo_K6_seed67_YOUR_RUN"
```

By default it saves PNG and vector PDF files in `RUN_DIR/regret_plots/`, using a
headless backend. Optional `--show` displays the figures on a configured desktop.
`--output-dir` chooses another folder. Rerunning replaces these plot files only;
the input CSVs, settings and histories are never modified. No C++ rebuild or
periodic-feature flag is needed. Vector PDFs are suitable for the thesis.

Paths are resolved from the current directory first. For relative paths or a
bare run name, the script also checks its own directory and results folders for
a unique matching run. Thus a mistaken `../results/RUN` can resolve to
`optimizer_py/results/RUN`; the resolved absolute directory is printed. Ambiguous
matches require an explicit path. Absolute paths are never redirected to a
different run. Missing-file errors list the checked locations. A run without an
original regret CSV cannot be reconstructed from its PNGs alone.

## Main figure: regret_convergence

1. **Estimated simple-regret bound and tolerance.** The first panel plots the
   recorded B = max UCB - best evaluated LCB against delta in negative-log-response
   units. The vertical scale is logarithmic; if a genuine recorded zero occurs,
   it switches to a symmetric-log scale so zero is not silently removed. Failed
   or non-finite checks are red crosses at the top of the panel, not zero regret.
   The bound line has gaps at failed checks.
2. **UCB and LCB evolution.** Both are shifted by the first valid incumbent LCB
   to remove the large common negative-log offset. Their vertical difference
   remains B. Shading shows this gap; it is not a confidence interval for one
   fixed configuration. The incumbent can change between checks.
3. **Consecutive passing checks.** This displays the recorded patience counter,
   passing/failing stopping checkpoints and, when settings are available, the required count.
   Grey points are diagnostic records; they never advance or reset patience.
   A dotted vertical line identifies the saved stopping point when the recorded
   reason is `estimated_simple_regret`.

The horizontal axis counts successful **new** observations, excluding the initial
design; it is not wall-clock time or attempted-loop iterations. The reports do
not store per-check elapsed time, so a seconds/minutes axis cannot be recovered
from these files alone.

For a relative physical interpretation, B corresponds to an estimated response
ratio exp(B), or an excess of 100*expm1(B) percent over the optimum, conditional
on the GP bounds being valid. The default delta=log(1.001) represents 0.1%.
This is not an independently measured shielding improvement.

## Supporting figure: optimization_context

When `bo_history.npz` or `bo_checkpoint.npz` and the initial design count are
available, a second figure shows:

- the running best observed response as a percentage of the best initial
  response, starting at 100%; smaller is better. This is a noisy observed
  incumbent, not an independently validated performance estimate;
- the saved average latent posterior variance at the five best observations.
  This is local GP uncertainty, not the Monte Carlo noise variance or global
  uncertainty. Grey vertical lines mark stopping checkpoints, not every diagnostic point.

This helps distinguish an improving observed incumbent from a decreasing regret
estimate caused mainly by changes in the fitted GP's uncertainty.

## Limits and useful interpretation

The script reads `regret_history.csv`, optionally `bo_settings.json`, and the
saved observation history. A checkpoint can be plotted before a run finishes.
Without settings, the patience target and stopping point are not guessed. An
empty CSV gives an explanatory error rather than an invented curve.

New runs record a full regret estimate every two successful updates from the
start (`BO_REGRET_RECORD_EVERY=2`), or every update if set to 1. Stopping is
separate: `BO_REGRET_CHECK_EVERY=10` applies after the minimum exploration budget.
For six coils the first stopping checks are 150, 160 and 170. With three passing
checks, a run ending at 170 therefore has 85 recorded points and three stopping
checkpoints. Recording continues even with early stopping disabled; set the
recording interval to 0 to disable extra records. More frequent diagnostics add
GP/UCB work but no Monte Carlo simulations.

Old sparse CSVs remain supported: without a `stopping_check` column, every row
is treated as a stopping checkpoint. Updating the plotter cannot add historical
estimates that were never recorded. No interpolation reconstructs unrecorded GP
fits; connecting lines only guide the eye between recorded markers.

The plots display the existing model-based estimates faithfully. A small or flat
bound is not a global-optimality certificate and does not validate the GP noise
model. In the earlier constant-mean, variance-floor case, c=2 and sigma=1e-5
give B approximately 4e-5; a flat curve there can reflect floor-dominated
uncertainty. Near-1e-10 values in the saved local variance trigger an informational
console note, not an automatic diagnosis of covariance failure.

The plotter does not estimate true cumulative regret: the unknown optimum is
not available. Integrating this curve would not recover actual cumulative regret.
The existing [noise experiment](NOISE_EXPERIMENT.md) remains useful for checking
whether the uncertainties underlying these regret estimates are credible.
