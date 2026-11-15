"""Single-run GP noise measured with fixed energies and fresh spatial scrambles."""
from numbers import Integral

import numpy as np

import input as config
from BoUtils import unpack_configuration
from Objective import evaluate_configuration, spherical_to_cartesian


def _radial_designs():
    """Near radial coils and far parallel coils, in the existing spherical convention."""
    index = np.arange(config.K)
    z = 1-2*(index+.5)/config.K
    azimuth = (np.pi*(3-np.sqrt(5))*index) % (2*np.pi)
    polar = np.arccos(z)
    # theta is azimuth in [0,pi]; phi can represent either sign of sin(phi).
    # This preserves each Cartesian direction when azimuth is larger than pi.
    other_half = azimuth > np.pi
    azimuth = np.where(other_half, azimuth-np.pi, azimuth)
    polar = np.where(other_half, 2*np.pi-polar, polar)
    near, far = config.X_BOUNDS
    physical = [
        np.concatenate((np.full(config.K, near), azimuth, polar, azimuth, polar)),
        np.concatenate((np.full(config.K, far), azimuth, polar,
                        np.zeros(config.K), np.zeros(config.K))),
    ]
    lower = config.LOWER.cpu().numpy()
    width = (config.UPPER-config.LOWER).cpu().numpy()
    return [(x-lower)/width for x in physical]


def _geometry(x):
    physical = config.LOWER.cpu().numpy()+x*(config.UPPER-config.LOWER).cpu().numpy()
    centers, normals = unpack_configuration(physical)
    cart_centers = spherical_to_cartesian(centers[:,0], centers[:,1], centers[:,2])
    cart_normals = spherical_to_cartesian(np.ones(config.K), normals[:,0], normals[:,1])
    return centers, normals, cart_centers, cart_normals


def calibrate_noise(X, y, repeats=6):
    """Pool within-configuration log variances; keep all repeats out of GP training.

    The best initial configuration and two contrasting active-field configurations
    each receive independent spatial seeds. Energies and particle count match BO.
    The returned variance describes ONE simulation, not the calibration mean.
    """
    if not isinstance(repeats, Integral) or repeats < 2:
        raise ValueError('Noise calibration requires at least two spatial scrambles.')
    if hasattr(X, 'detach'):
        X = X.detach().cpu().numpy()
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (X.ndim != 2 or X.shape[1] != config.D or y.shape != (len(X),) or not len(X)
            or not np.isfinite(X).all() or not np.isfinite(y).all()
            or np.any((X < 0)|(X > 1))):
        raise ValueError('Calibration needs finite normalized initial designs and scores.')

    candidates = [X[int(y.argmax())], *_radial_designs()]
    names = ['best_initial', 'near_radial', 'far_aligned']
    designs, means, mean_variances, variances, used_names, runs, geometries = [], [], [], [], [], [], []
    seed_start = config.SEED+2_000_000
    for case_index, (name, x) in enumerate(zip(names, candidates)):
        centers, normals, cart_centers, cart_normals = _geometry(x)
        if any(np.allclose(cart_centers, c, rtol=0, atol=1e-10)
               and np.allclose(cart_normals, n, rtol=0, atol=1e-10) for c, n in geometries):
            continue
        geometries.append((cart_centers, cart_normals))
        scores = []
        for replicate in range(repeats):
            seed = seed_start+case_index*repeats+replicate
            response, particle_variance = evaluate_configuration(
                seed, centers, normals, common_energy=True)
            if not np.isfinite(response) or response <= 0:
                raise FloatingPointError('Nonpositive or nonfinite calibration response.')
            score = -float(np.log(response))
            scores.append(score)
            runs.append(dict(configuration=name, replicate=replicate+1, spatial_seed=seed,
                             response=float(response), score=score,
                             particle_variance=float(particle_variance)))
        variance = float(np.var(scores, ddof=1))
        designs.append(x.tolist())
        means.append(float(np.mean(scores)))
        variances.append(variance)
        mean_variances.append(variance/repeats)
        used_names.append(name)
        print(f'Noise calibration {name}: single-run log variance={variance:.6g}; '
              f'repeats={repeats}.', flush=True)

    # Equal replicate counts give equal degrees of freedom for each configuration.
    pooled = max(float(np.mean(variances)), 1e-12)
    print(f'Calibrated GP single-run log variance: {pooled:.6g}.', flush=True)
    return dict(variance=pooled, X=designs, mean=means, mean_variance=mean_variances,
                single_run_variance=variances, names=used_names, repeats=int(repeats), runs=runs,
                settings=dict(K=config.K, particles_per_species=config.N, current_A=config.I,
                              coil_radius_m=config.R, energy_bins=config.ENERGY_BINS,
                              energy_seed=config.BO_ENERGY_SEED, spatial_seed_start=seed_start,
                              lower=config.LOWER.cpu().tolist(), upper=config.UPPER.cpu().tolist(),
                              numerical_variance_floor=1e-12))
