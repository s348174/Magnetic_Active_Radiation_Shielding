"""Offline coil-set GP validation on saved observations; never runs the simulator."""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

import input as config
from BoLoop import _fit_model
from BoUtils import map_coilwise
from CoilSetModel import make_coil_kernel, select_coil_scales
from PosteriorChecks import check_kernel, raw_mean_variance


def load_run(run_dir):
    """Reject physical-setting mismatches rather than reinterpreting saved designs."""
    directory = Path(run_dir).expanduser().resolve()
    with open(directory/'noise_calibration.json') as file:
        calibration = json.load(file)
    settings = calibration['settings']
    expected = dict(K=config.K, particles_per_species=config.N, current_A=config.I,
                    coil_radius_m=config.R, energy_bins=config.ENERGY_BINS,
                    energy_seed=config.BO_ENERGY_SEED)
    mismatches = []
    for name, value in expected.items():
        saved = settings.get(name)
        if (not isinstance(saved, (int, float))
                or not math.isclose(saved, value, rel_tol=1e-12, abs_tol=1e-12)):
            mismatches.append(name)
    for name, value in (('lower',config.LOWER), ('upper',config.UPPER)):
        saved = np.asarray(settings.get(name, []), dtype=float)
        current = value.cpu().numpy()
        if saved.shape != current.shape or not np.allclose(saved,current,rtol=1e-12,atol=1e-12):
            mismatches.append(name)
    if mismatches:
        raise ValueError('Saved run differs from input.py: '+', '.join(mismatches)+
                         '. Restore the original physical settings before validating this run.')
    noise = float(calibration['variance'])
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError('Saved single-run log variance must be finite and positive.')
    source = next((directory/name for name in ('bo_history.npz','bo_checkpoint.npz')
                   if (directory/name).is_file()), None)
    if source is None:
        raise ValueError('Run directory needs bo_history.npz or bo_checkpoint.npz.')
    with np.load(source,allow_pickle=False) as stored:
        X, y = np.asarray(stored['X'],dtype=float), np.asarray(stored['y'],dtype=float)
    if (X.ndim != 2 or X.shape[1] != config.D or y.shape != (len(X),)
            or len(X) < 10 or not np.isfinite(X).all() or not np.isfinite(y).all()
            or np.any((X < 0)|(X > 1))):
        raise ValueError('Need at least ten finite normalized saved designs with matching scores.')
    return X, y, noise, settings, source


def validate_surrogate(run_dir, output_dir=None, seed=67):
    """Select geometry scales without outer-test labels, then assess the refitted GP."""
    X, y, noise, settings, source = load_run(run_dir)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Validation seed must be a nonnegative integer.')
    order = np.random.default_rng(seed).permutation(len(X))
    test_count = max(2, int(math.ceil(.2*len(X))))
    test, train = order[:test_count], order[test_count:]
    X_train = torch.as_tensor(X[train],dtype=torch.double,device=config.device)
    X_test = torch.as_tensor(X[test],dtype=torch.double,device=config.device)
    variance = torch.full((len(train),1),noise,dtype=torch.double,device=config.device)
    # The complete outer test fold is excluded from every scale-selection fit.
    ell, kappa, selection = select_coil_scales(
        map_coilwise(X_train),y[train],variance,n_coils=config.K,seed=seed)
    gp = _fit_model(X_train,y[train],variance,map_coilwise,
                    make_coil_kernel(config.K,ell,kappa).to(X_train))
    kernel_report = check_kernel(gp,map_coilwise(X_train))
    with torch.no_grad():
        mean, latent = raw_mean_variance(gp,map_coilwise(X_test))
    prediction = mean.reshape(-1).detach().cpu().numpy()
    latent = latent.reshape(-1).detach().cpu().numpy()
    if (prediction.shape != y[test].shape or latent.shape != y[test].shape
            or not np.isfinite(prediction).all() or not np.isfinite(latent).all()
            or np.any(latent < 0)):
        raise RuntimeError('Invalid held-out GP mean or latent variance.')
    baseline_mean = float(y[train].mean())
    mse = float(np.mean((prediction-y[test])**2))
    baseline_mse = float(np.mean((baseline_mean-y[test])**2))
    correlation = (float(np.corrcoef(y[test],prediction)[0,1])
                   if np.ptp(y[test]) > 0 and np.ptp(prediction) > 0 else None)
    # These are individual noisy observations, not repeat means.
    predictive_variance = latent+noise
    half_width = 1.959963984540054*np.sqrt(predictive_variance)
    covered = np.abs(prediction-y[test]) <= half_width
    selection = dict(selection)
    for name in ('train_indices','validation_indices'):
        if name in selection:
            selection['global_'+name] = train[np.asarray(selection[name],dtype=int)].tolist()
    report = dict(source=str(source), target='negative_log_received_energy', split_seed=seed,
                  samples=len(X),train_count=len(train),test_count=len(test),
                  train_indices=train.tolist(),test_indices=test.tolist(),
                  outer_test_excluded_from_selection=True,
                  validation_scope='Offline holdout of historical BO observations; '
                                   'not fresh independent simulator validation.',
                  noise_variance=noise,physical_settings=settings,
                  selected_lengthscale=float(ell),selected_kappa=float(kappa),
                  selection=selection,kernel_check=kernel_report,
                  log_rmse=math.sqrt(mse),constant_mean=baseline_mean,
                  constant_log_rmse=math.sqrt(baseline_mse),correlation=correlation,
                  skill=1-mse/baseline_mse if baseline_mse > 0 else None,
                  coverage95=float(covered.mean()),coverage_includes_observation_noise=True,
                  minimum_raw_variance=float(latent.min()),
                  outputscale=float(gp.covar_module.outputscale.detach().cpu()))
    if output_dir is None:
        Path('results').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='surrogate_validation_',dir='results'))
    else:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True,exist_ok=False)
    report['output_dir'] = str(directory.resolve())
    with open(directory/'validation_report.json','w') as file:
        json.dump(report,file,indent=2,allow_nan=False)
    with open(directory/'held_out_predictions.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index','observed_score','predicted_score','constant_prediction',
                         'latent_variance','observation_noise_variance','predictive_variance',
                         'lower95','upper95','covered95'])
        writer.writerows(zip(test,y[test],prediction,np.full(len(test),baseline_mean),
                             latent,np.full(len(test),noise),predictive_variance,
                             prediction-half_width,prediction+half_width,covered))
    print(f'Outer holdout: {len(train)} training / {len(test)} test observations. '
          f'RMSE={report["log_rmse"]:.6g}; constant={report["constant_log_rmse"]:.6g}; '
          f'skill={report["skill"]}; coverage95={report["coverage95"]:.3f}.')
    print(f'Offline surrogate report: {directory.resolve()}.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path,help='Saved BO results directory')
    parser.add_argument('--output-dir',type=Path,help='New report directory; default: results/surrogate_validation_*')
    parser.add_argument('--seed',type=int,default=67,help='Local outer split seed; default: 67')
    args = parser.parse_args(argv)
    try:
        return validate_surrogate(args.run_dir,args.output_dir,args.seed)
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        parser.exit(1,f'Offline surrogate validation failed: {error}\n')


if __name__ == '__main__':
    main()
