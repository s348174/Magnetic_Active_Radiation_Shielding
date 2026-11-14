# Common optimization energies, fresh spatial scrambles and independent validation

The optimization objective remains the total flux-weighted received-energy score.
The GP still maximizes its negative logarithm. The relative received-energy
diagnostic described below is never substituted into the objective or final ranking.

## Randomness during optimization

`input.py` defines `BO_ENERGY_SEED = SEED`. All calls to `objective_function`,
including initial Sobol evaluations, vanilla BO, TuRBO regions and retries, use
this common energy seed. For a fixed spectrum, particle count and number of bins,
the energy arrays and their unweighted and flux-weighted totals are identical
across configurations. The arrays are regenerated deterministically from the
same seed rather than retained in an additional large cache. The logarithmic
bins, inverse-CDF sampler, physical flux weights and particle-energy conversion
are unchanged. Energies are not rescaled to force a particular total.

Spatial samples are deliberately different. `next_optimization_seed()` allocates
consecutive, nonrepeated spatial seeds starting at `SEED`. The C++ sampler still
applies its existing hash-based nested Owen scramble to Sobol sphere points,
separating species, dimensions and energy-bin seeds as before. A retry consumes
a fresh spatial seed but retains the common energies. The allocator rejects
exhaustion before reaching the default validation seed range at `SEED+1000000`.

This is **energy-only common random numbers**, rather than sharing every random
input. It removes changing energy draws from comparisons while preserving fresh
randomized-QMC spatial sampling between evaluations. Energy draws remain
stratified pseudorandom draws; the complete estimator is a hybrid of energy
Monte Carlo and spatial randomized quasi-Monte Carlo, as before.

For the current N=1000, 28 species and eight nonempty bins, every optimization
evaluation has 28000 particles with 125 energies in each species/bin block.
Changing N, the spectrum, bin settings or BO_ENERGY_SEED changes the common bank.

## Final validation

`evaluate_finalists` explicitly calls `evaluate_configuration` with
`common_energy=False`. Each finalist/replicate pair uses its own seed, with
fresh energies and a fresh spatial Owen scramble. Default seeds start at
`SEED+1000000`, independently of the optimization spatial range.

NumPy SeedSequence also separates optimization and validation energy streams:
optimization uses entropy `[1, BO_ENERGY_SEED]`; validation uses `[0, seed]`.
Thus matching integer seed values still do not reuse optimization energies.
The scheme is reproducible, but its energy draws differ from older code versions.
Repeating an entire campaign with the same settings/seeds intentionally reproduces
the same seed scheme; different replicate seeds within a campaign remain distinct.

Validation still defaults to 100000 particles per species and three independent
replicates per finalist. Ranking uses arithmetic means of raw received-energy
responses, with uncertainty estimated between complete replicates. It does not
rank average percentages or feed validation observations back into the GP.

## Relative received-energy diagnostic

Let e_sbi be a sampled energy in MeV/amu, A_s the species mass in amu from
`data/particles_input.csv`, W_sb its bin's integrated physical flux and n_sb its
sample count. The sampled incident-energy estimate is

```text
E_in = sum_species sum_bins W_sb / n_sb * sum_i (A_s * 1e6 * e_sbi)
```

The displayed diagnostic is

```text
relative received = 100 * received_energy_score / E_in
```

The same flux weights and total nuclear energy units are used in numerator and
denominator. Dividing the weighted response by the unweighted energy sum of
simulated particles would be incorrect because the bins are oversampled unevenly
relative to their physical flux. `Objective.incident_energy` reads the same mass
table as the simulator, including supported multiplicative mass entries.

Every optimization or validation evaluation prints the received score, incident
estimate, percentage, energy seed/mode and spatial seed. For example:

```text
Energy diagnostic: received=..., sampled incident=..., relative received=...%
(smoothed detector score); energy seed=67 (common), Owen spatial seed=68.
```

The percentage uses the simulator's existing smooth detector response, not a
binary hit indicator or a material energy-deposition model. It is not guaranteed
to lie in [0,100], and is not clipped. Long curved paths, detector-score weighting
or integration error can produce values above 100%; that should be investigated
as a detector-model diagnostic, not hidden by normalization.

The mode and BO_ENERGY_SEED are saved in `bo_settings.json` or
`region_selection.json` by the driver. Individual validation seeds remain recorded
in `validation_runs.csv`.

## Interpretation and limitations

Common energies reduce one source of variation between configurations, but BO
now fits a response conditional on a finite energy bank. It can overfit that
bank. Fresh high-particle-count validation is therefore essential to assess
performance under the original spectrum. Repeating full searches with different
BO_ENERGY_SEED values can assess sensitivity to that choice.

The existing within-run variance and log-response delta-method estimates are
retained. They are approximate: they neither model all cross-configuration
dependence from shared energies nor measure the fixed bank's integration error
through spatial rescrambling alone. The GP simple-regret criterion is still
model-based and does not certify the optimum of the full physical energy integral.

Only Python sampling orchestration and diagnostics changed; the Boris integrator,
magnetic field and C++ Owen scrambling are unchanged. No C++ rebuild is required.

## Verification

The `multi_turbo_validation` regression suite checks equality of optimization
energy arrays across different spatial seeds, separation of validation streams,
unique spatial-seed allocation, retry behavior, weighted/nuclear-energy diagnostic
calculations, printed percentages and preservation of the raw-energy objective.
Existing final-validation tests assert fresh-energy mode and independent seeds.

```bash
ctest --test-dir bayesian_optimization/build-cloud --output-on-failure
```
