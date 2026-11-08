# Import C++ bindings
import sys
sys.path.append('../Release')
import simulator

import numpy as np

from LogDataSampler import sample_all_particles_stratified
from input import K, N, I, R, FLUX_DATA, ENERGY_BINS

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian.
    Convention: theta = azimuth (from X-axis in XY-plane), phi = polar (from Z-axis)
    
    Args:
        r:     radial distance, shape (K,)
        theta: azimuth angle, shape (K,)
        phi:   polar angle,   shape (K,)
    Returns:
        xyz:   cartesian coordinates, shape (K, 3)
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def evaluate_configuration(seed, centers, normals, n_particles=None):
    """Return the raw flux-weighted response and within-run variance estimate.

    A supplied seed controls both energy sampling and spatial scrambling.
    n_particles is per species, split over the logarithmic energy bins.
    """
    count = N if n_particles is None else int(n_particles)
    centers_cart = spherical_to_cartesian(centers[:, 0], centers[:, 1], centers[:, 2])
    normals_cart = spherical_to_cartesian(np.ones(K), normals[:, 0], normals[:, 1])
    samples, strata = sample_all_particles_stratified(
        FLUX_DATA, n_samples=count, n_bins=ENERGY_BINS,
        generator=np.random.default_rng(int(seed)))
    mean, variance = simulator.launch_simulation(
        int(seed), count, K, I, R, centers_cart, normals_cart, samples, strata)
    if not np.isfinite([mean, variance]).all() or mean <= 0 or variance < 0:
        raise FloatingPointError("Simulator returned an invalid response or variance.")
    return float(mean), float(variance)


def objective_function(seed, centers, normals):
    """Log-response and delta-method variance used during optimization."""
    try:
        mean, variance = evaluate_configuration(seed, centers, normals)
    except FloatingPointError:
        print(f"Retrying non-finite response with seed {int(seed)+1}.")
        mean, variance = evaluate_configuration(int(seed)+1, centers, normals)
    return np.log(mean), variance / (mean**2 + 1e-6) + 1e-6
