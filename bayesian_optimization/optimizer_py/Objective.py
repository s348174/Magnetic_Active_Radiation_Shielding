# Import C++ bindings
import sys
sys.path.append('../Release')
import simulator

import numpy as np
import csv
from itertools import count

from LogDataSampler import sample_all_particles_stratified
from input import K, N, I, R, FLUX_DATA, ENERGY_BINS, SEED, BO_ENERGY_SEED

# Disjoint from the default final-validation seed range, which starts at SEED+1e6.
_optimization_seeds = count(SEED)


def next_optimization_seed():
    """A fresh spatial scramble for each optimization evaluation, including retries."""
    seed = next(_optimization_seeds)
    if seed >= SEED+1000000:
        raise RuntimeError('Optimization spatial-seed range exhausted.')
    return seed


def incident_energy(samples, strata):
    """Flux-weighted incident nuclear energy, using the simulator's mass table."""
    total = 0.
    with open('../data/particles_input.csv',newline='') as file:
        for particle in csv.DictReader(file):
            name = particle['particleName'].strip()
            # The C++ table uses masses in amu and also accepts multiplication.
            mass_number = float(np.prod([float(term) for term in particle['m'].split('*')]))
            if not np.isfinite(mass_number) or mass_number <= 0:
                raise ValueError('Particle masses must be finite and positive.')
            offset = 0
            for count_in_bin, flux in strata[name]:
                count_in_bin = int(count_in_bin)
                block = samples[name][offset:offset+count_in_bin]
                total += float(flux)*float(block.mean())*mass_number*1e6
                offset += count_in_bin
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError('Invalid sampled incident energy.')
    return float(total)

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

def evaluate_configuration(seed, centers, normals, n_particles=None, common_energy=False):
    """Return the raw flux-weighted response and within-run variance estimate.

    Default: fresh energy and spatial seeds, as required for final validation.
    common_energy=True reuses BO_ENERGY_SEED while retaining the supplied spatial seed.
    n_particles is per species, split over the logarithmic energy bins.
    """
    count = N if n_particles is None else int(n_particles)
    centers_cart = spherical_to_cartesian(centers[:, 0], centers[:, 1], centers[:, 2])
    normals_cart = spherical_to_cartesian(np.ones(K), normals[:, 0], normals[:, 1])
    energy_seed = BO_ENERGY_SEED if common_energy else int(seed)
    # Separate search and validation energy streams, even if integer seeds coincide.
    generator = np.random.default_rng(np.random.SeedSequence([int(common_energy),int(energy_seed)]))
    samples, strata = sample_all_particles_stratified(
        FLUX_DATA, n_samples=count, n_bins=ENERGY_BINS,
        generator=generator)
    incoming = incident_energy(samples,strata)
    mean, variance = simulator.launch_simulation(
        int(seed), count, K, I, R, centers_cart, normals_cart, samples, strata)
    if not np.isfinite([mean, variance]).all() or mean <= 0 or variance < 0:
        raise FloatingPointError("Simulator returned an invalid response or variance.")
    mode = 'common' if common_energy else 'fresh'
    print(f"Energy diagnostic: received={mean:.8g}, sampled incident={incoming:.8g}, "
          f"relative received={100*mean/incoming:.6f}% (smoothed detector score); "
          f"energy seed={energy_seed} ({mode}), Owen spatial seed={int(seed)}.",flush=True)
    return float(mean), float(variance)


def objective_function(seed, centers, normals):
    """Log-response and delta-method variance used during optimization."""
    try:
        mean, variance = evaluate_configuration(seed, centers, normals,common_energy=True)
    except FloatingPointError:
        retry_seed = next_optimization_seed()
        if retry_seed == int(seed):
            retry_seed = next_optimization_seed()
        print(f"Retrying non-finite response with spatial seed {retry_seed}; common energies retained.")
        mean, variance = evaluate_configuration(retry_seed,centers,normals,common_energy=True)
    return np.log(mean), variance / (mean**2 + 1e-6) + 1e-6
