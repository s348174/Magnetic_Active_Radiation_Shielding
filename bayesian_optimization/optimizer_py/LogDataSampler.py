"""
Sample energies from particle flux distributions.

Each particle has a discrete (Energy, Flux) spectrum. This script treats the
flux as proportional to a probability density function (PDF) and draws random
energy samples via inverse-CDF sampling on the piecewise-linear interpolation
of that PDF.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from input import rng

# ---------------------------------------------------------------------------
# Core sampling helpers
# ---------------------------------------------------------------------------

def diagnose_sampling_input(df: pd.DataFrame) -> None:
    """
    Validate input CSV content before sampling.

    Raises ValueError with a compact diagnostic report if invalid values are
    found that can lead to skipped/incomplete sampling.
    """
    issues = []

    if "Energy" not in df.columns:
        raise ValueError("Input data must contain an 'Energy' column.")

    energy = pd.to_numeric(df["Energy"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(energy).any() or np.isinf(energy).any():
        bad = int((~np.isfinite(energy)).sum())
        issues.append(f"Energy has {bad} non-finite values (NaN/Inf)")

    finite_energy = energy[np.isfinite(energy)]
    if finite_energy.size >= 2 and np.any(np.diff(finite_energy) <= 0):
        issues.append("Energy is not strictly increasing (after removing non-finite entries)")

    particles = df.columns[1:].tolist()
    if not particles:
        issues.append("No particle columns found (columns after 'Energy')")

    for particle in particles:
        flux = pd.to_numeric(df[particle], errors="coerce").to_numpy(dtype=float)
        non_finite = int((~np.isfinite(flux)).sum())
        if non_finite > 0:
            issues.append(f"{particle}: {non_finite} non-finite flux values (NaN/Inf)")
            continue

        if np.any(flux < 0):
            neg = int((flux < 0).sum())
            issues.append(f"{particle}: {neg} negative flux values")

        if np.all(flux == 0):
            issues.append(f"{particle}: flux is all zeros")

    if issues:
        report = "\n  - " + "\n  - ".join(issues)
        raise ValueError(
            "Invalid values detected in source CSV before sampling. "
            "Fix data issues to avoid incomplete sampling:" + report
        )

def build_sampler(energy: np.ndarray, flux: np.ndarray) -> callable:
    """
    Build an inverse-CDF sampler for a single particle's energy distribution.

    The flux values are treated as a piecewise-linear PDF over the energy axis.
    We integrate it to get the CDF, normalise, then invert it so that uniform
    U(0,1) draws map to energy samples.

    Parameters:
    energy : 1-D array, shape (N,)
        Sorted energy values (any units, e.g. MeV/nucleon).
    flux   : 1-D array, shape (N,)
        Flux values (must be ≥ 0; treated as unnormalised PDF weights).

    Returns:
    sampler : callable
        sampler(n_samples) → np.ndarray of shape (n_samples,)
    """
    energy = np.asarray(energy, dtype=float)
    flux   = np.asarray(flux,   dtype=float)

    # Safety: clip any tiny negatives that might arise from floating-point noise
    flux = np.clip(flux, 0.0, None)

    # Trapezoidal cumulative integral → CDF
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(energy) * 0.5 * (flux[:-1] + flux[1:]))])
    total = cdf[-1]

    if total == 0:
        raise ValueError("Flux is identically zero — cannot define a distribution.")

    cdf_norm = cdf / total  # normalise to [0, 1]

    # Remove duplicate CDF values (needed for monotone interpolation)
    _, unique_idx = np.unique(cdf_norm, return_index=True)
    cdf_unique    = cdf_norm[unique_idx]
    energy_unique = energy[unique_idx]

    # Inverse CDF: uniform → energy
    inv_cdf = interp1d(cdf_unique, energy_unique, kind="linear",
                       bounds_error=False,
                       fill_value=(energy_unique[0], energy_unique[-1]))

    def sampler(n_samples: int = 1) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n_samples)
        return inv_cdf(u)

    return sampler


def sample_particle(df: pd.DataFrame, particle: str, n_samples: int = 1000) -> np.ndarray:
    """
    Draw n_samples energy values from the distribution of particle.

    Parameters:
    df        : DataFrame with an 'Energy' column and one column per particle.
    particle  : Column name of the particle (e.g. 'proton', 'Fe56').
    n_samples : Number of samples to draw.

    Returns:
    samples : np.ndarray, shape (n_samples,)
    """
    if particle not in df.columns:
        raise ValueError(f"Particle '{particle}' not found. "
                         f"Available: {list(df.columns[1:])}")

    sampler = build_sampler(df["Energy"].values, df[particle].values)
    return sampler(n_samples)


def sample_all_particles(df: pd.DataFrame,
                         n_samples: int = 1000) -> dict[str, np.ndarray]:
    """
    Draw n_samples energy values for every particle in df.
    Returns dict mapping particle name → np.ndarray of sampled energies.
    """
    diagnose_sampling_input(df)
    particles = df.columns[1:].tolist()   # everything except 'Energy'
    results   = {}

    for particle in particles:
        try:
            results[particle] = sample_particle(df, particle, n_samples)
        except ValueError as exc:
            print(f"  [skip] {particle}: {exc}")

    return results

# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def plot_samples(df: pd.DataFrame,
                 samples: dict[str, np.ndarray],
                 particles_to_plot: list[str] | None = None,
                 log_scale: bool = True) -> None:
    """
    For each selected particle, overlay the original flux spectrum (normalized)
    with a histogram of the sampled energies to verify the sampling is correct.
    """
    if particles_to_plot is None:
        particles_to_plot = list(samples.keys())[:2]

    n = len(particles_to_plot)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    energy = df["Energy"].values

    for ax, particle in zip(axes, particles_to_plot):
        flux = df[particle].values

        # Normalize flux to area = 1 for comparison with histogram density
        area  = np.trapezoid(flux, energy)
        flux_norm = flux / area if area > 0 else flux

        ax.plot(energy, flux_norm, label="PDF (flux, normalized)", lw=2)
        ax.hist(samples[particle], bins=100, density=True,
                alpha=0.5, label="Samples (histogram)")

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_title(particle)
        ax.set_xlabel("Energy")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Energy sampling verification", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("sampling_verification.png", dpi=150, bbox_inches="tight")
    print("Verification plot saved to sampling_verification.png")
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = "../data/log_scaled_flux_data.csv"
    N_SAMPLES = 100_000

    print(f"Loading data from '{DATA_PATH}' …")
    df = pd.read_csv(DATA_PATH)
    #energy_steps, data = read_energy_data(DATA_PATH)
    # Log scaling for sampling stability (especially for low-flux tails)
    #df = build_log_scaled_dataset(energy_steps, data)
    particles = df.columns[1:].tolist()
    print(f"  {len(particles)} particles, {len(df)} energy bins, "
          f"energy range [{df['Energy'].min():.3g}, {df['Energy'].max():.3g}]\n")

    print(f"Sampling {N_SAMPLES:,} energies per particle …")
    samples = sample_all_particles(df, n_samples=N_SAMPLES)

    # Print summary statistics
    print(f"\n{'Particle':<10}  {'Mean E':>12}  {'Median E':>12}  "
          f"{'5th pct':>10}  {'95th pct':>10}")
    print("-" * 60)
    for particle, s in samples.items():
        print(f"{particle:<10}  {np.mean(s):>12.4g}  {np.median(s):>12.4g}  "
              f"{np.percentile(s, 5):>10.4g}  {np.percentile(s, 95):>10.4g}")

    # Optional: visualise first 2 particles
    print("\nGenerating verification plots …")
    plot_samples(df, samples, particles_to_plot=particles[:2], log_scale=False)