# Import C++ bindings
import sys
from xml.parsers.expat import model
sys.path.append('../Release')
import simulator

import numpy as np
from joblib import Parallel, delayed

from LogDataSampler import sample_all_particles
from input import K, N, I, R, LOG_DATA, rng, COPY

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

def objective_function(seed, centers, normals):
    """
    Calls the C++ simulator to compute the energy hitting the detector.
    R: is coils radius
    centers: list of coil centers (spherical coordinates: r, theta, phi)
    normals: list of coil normal vectors (spherical coordinates: theta, phi)
    Returns: energy hitting the detector (lower is better)
    """
    # Change coordinates from spherical to cartesian for simulator input
    centers_cart = spherical_to_cartesian(centers[:, 0], centers[:, 1], centers[:, 2]) # shape: (K, 3)
    normals_cart = spherical_to_cartesian(np.ones(K), normals[:, 0], normals[:, 1]) # shape: (K, 3) - unit vectors for normals
    
    # Sample particle energies from the log data distribution (for all particles, but we can choose one if needed)
    samples_dict = sample_all_particles(LOG_DATA, n_samples=N)
    
    # Call the C++ simulator
    energies = Parallel(n_jobs=-1)(
        delayed(simulator.launch_simulation)(seed, N, K, I, R, centers_cart, normals_cart, samples_dict) 
        for _ in range(COPY)
    )
    expected_energy = np.mean(energies)
    var = np.var(energies, ddof=1, dtype=np.float64)/COPY # Variance of the mean estimate, can be used to adaptively adjust noise floor if desired
    print(f"Mean energy from {COPY} replicates: {expected_energy:.4f} with variance: {var:.6f}")
    # Take the log of energy to stabilize optimization and handle wide range of values
    log_energy = np.log(expected_energy)
    log_var = var / (expected_energy ** 2) + 1e-6 # Variance of log(energy) using delta method
    if np.isnan(log_energy) or np.isinf(log_energy):
        print(f"Warning: Non-finite energy encountered. Expected energy: {np.mean(energies)}. Try again with a different seed or check the simulator for issues.")
        samples_dict = sample_all_particles(LOG_DATA, n_samples=N)  # Debug: print some samples to check distribution
        seed = rng.integers(1e6)
        print(f"New seed: {seed}")
        energies = Parallel(n_jobs=-1)(
            delayed(simulator.launch_simulation)(seed, N, K, I, R, centers_cart, normals_cart, samples_dict) 
            for _ in range(COPY)
        )
        expected_energy = np.mean(energies)
        var = np.var(energies, ddof=1, dtype=np.float64)/COPY # Variance of the mean estimate, can be used to adaptively adjust noise floor if desired
        print(f"Mean energy from {COPY} replicates: {expected_energy:.4f} with variance: {var:.6f}")
        # Take the log of energy to stabilize optimization and handle wide range of values
        log_energy = np.log(expected_energy)
        log_var = var / (expected_energy ** 2) + 1e-6 # Variance of log(energy) using delta method
    return log_energy, log_var
