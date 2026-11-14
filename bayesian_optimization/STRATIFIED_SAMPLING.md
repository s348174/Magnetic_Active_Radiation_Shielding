# Logarithmic energy strata

The optimizer now reads `data/flux_data_clean.csv`. Energies remain in MeV/amu
and flux values retain their original physical normalization. Logarithmic bin
boundaries replace the old rescaling of energy and flux values. The existing
log-log visualization utilities and generated table remain available.

`input.py` sets `ENERGY_BINS = 8` and `N = 1000`. Eight equally spaced intervals
in log-energy cover the complete tabulated range. Every species uses exactly
`N` trajectories, divided equally among its nonempty bins, with any remainder
assigned to the first bins. Every nonempty bin requires at least two samples;
an insufficient budget raises an error rather than omitting the tail.

The physical flux is interpolated **linearly in energy** between the original
table nodes. Bin boundaries are inserted into this interpolant. Trapezoidal
integration gives each bin's exact mass for that interpolant, and a quadratic
inverse CDF samples its normalized density exactly. Zero-density intervals are
skipped without interpolating across their support gaps. No arbitrary tail
cutoff is introduced beyond the limits of the supplied table.

`sample_all_particles_stratified` returns two dictionaries:

- `samples[species]`: consecutive energy sample blocks, ordered by bin;
- `strata[species]`: rows `[sample_count, integrated_bin_flux]`.

The optional ninth argument to `simulator.launch_simulation` carries `strata`.
The original eight-argument call remains available for whole-spectrum sampling.
The C++ worker evaluates each bin separately and returns

```
response = sum_species sum_bins flux_bin * mean_score_bin
variance = sum_species sum_bins flux_bin^2 * sample_variance_bin / count_bin
```

The bin flux already contains the species abundance: it is **not** multiplied
by the total species flux again. Counts and total bin flux are checked against
the sample array and the original physical spectrum before threads start.
Invalid bin metadata or missing species raises an error in Python.

The outer Bayesian optimizer still fits the negative logarithm of the response.
The variance formula is the usual stratified sample-mean estimate, not a
calibrated randomized-QMC uncertainty estimate. Spatial Sobol points now use
seeded nested Owen scrambling. Final validation estimates uncertainty between
independent complete replicates; see `MULTI_TURBO.md`. Optimization now shares
energy draws and refreshes spatial scrambles; see [COMMON_RANDOM_ENERGIES.md](COMMON_RANDOM_ENERGIES.md).
GP noise handling is
unchanged. Historical optimization values from the rescaled spectra are
not directly comparable and should be regenerated for this sampling model.

After configuring and building `simulation_cpp`, run both suites with:

```bash
ctest --test-dir bayesian_optimization/build-cloud --output-on-failure
```

Tests cover analytic spectra, zero-density gaps, bin flux conservation and tail
coverage, unequal counts, squared-weight variance, Python/C++ integration and
invalid metadata, alongside the relativistic integration regressions.
