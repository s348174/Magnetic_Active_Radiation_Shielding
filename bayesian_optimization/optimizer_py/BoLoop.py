"""Single-loop Bayesian optimization with Matérn, RBF and coil-set kernels."""
import warnings
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
from RobustGPFitting import fit_gpytorch_mll_robust as fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement, PosteriorMean, qKnowledgeGradient
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from AcquisitionOptimizer import optimize_acqf
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.settings import min_fixed_noise
from Kernels_v3 import CoilSetKernel

from BoUtils import (
    denormalize, unpack_configuration, map_fn, map_set, map_coilwise,
    InputMappedAcquisition, InputMappedModel, sobol_sample,
    duplicate_safe_candidate,
)
from RegretStopping import SimpleRegretStopping, lower_bound_incumbent
from Objective import objective_function, next_optimization_seed
from NoiseCalibration import calibrate_noise
from ModelDiagnostics import assess_model
from input import device, MAX_ITER, unit_bounds, Q, BO_NOISE_REPEATS, K


def _fit_model(X_train, y_train, train_yvar, mapping, covar_module):
    # Standardize transforms BOTH targets and their variances consistently.
    with warnings.catch_warnings(), min_fixed_noise(double_value=1e-12):
        warnings.filterwarnings("ignore", category=InputDataWarning)
        gp = SingleTaskGP(
            mapping(X_train).detach().clone(),
            torch.as_tensor(y_train, dtype=torch.double, device=device).unsqueeze(-1),
            train_Yvar=train_yvar,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def _best_posterior_mean(gp, mapping):
    _, value = optimize_acqf(
        acq_function=InputMappedAcquisition(PosteriorMean(gp), mapping),
        bounds=unit_bounds,
        q=1,
        num_restarts=30,
        raw_samples=256,
    )
    return value.detach().item()


def _bo_loop(kernel, mapping, max_iter=None, initial_data=None, nu=2.5, return_diagnostics=False,
             checkpoint=None, calibration=None, model_callback=None):
    iterations = MAX_ITER if max_iter is None else int(max_iter)
    if kernel not in ('matern','rbf','coil-set'):
        raise ValueError(f'Unknown vanilla BO kernel: {kernel}')
    if iterations < 1 or Q != 1:
        raise ValueError("Use a positive iteration budget and Q=1 for these single-point loops.")
    stopping = SimpleRegretStopping()

    # Initial observations may be shared with the driver without simulating twice.
    X_train, y_train, train_yvar = sobol_sample() if initial_data is None else initial_data
    X_train = torch.as_tensor(X_train, dtype=torch.double, device=device).detach().clone()
    y_train = np.asarray(y_train, dtype=float).copy()
    # Old particle-based variance estimates are replaced, including for saved data.
    valid = np.isfinite(y_train)
    valid &= torch.isfinite(X_train).all(dim=-1).cpu().numpy()
    X_train, y_train = X_train[valid], y_train[valid]
    if len(y_train) < 2:
        raise ValueError("At least two finite initial observations are required.")
    if calibration is None:
        calibration = calibrate_noise(X_train, y_train, repeats=BO_NOISE_REPEATS)
    noise = float(calibration['variance'])
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError('Calibrated single-run log variance must be finite and positive.')
    train_yvar = torch.full((len(y_train),1),noise,dtype=torch.double,device=device)
    initial_evaluations = len(y_train)
    acquisition_hist, best_mean_hist, variance_hist = [], [], []
    evaluations, fitted_evaluations, attempted_iteration = 0, 0, 0

    def save_checkpoint(status, error=''):
        if checkpoint is not None:
            checkpoint({'X':X_train.cpu().numpy(), 'y':y_train,
                        'variance':train_yvar.squeeze(-1).cpu().numpy(),
                        'acquisition':np.asarray(acquisition_hist),
                        'posterior_mean':np.asarray(best_mean_hist),
                        'posterior_variance':np.asarray(variance_hist),
                        'status':np.asarray(status), 'error':np.asarray(error),
                        'evaluations':np.asarray(evaluations),
                        'fitted_evaluations':np.asarray(fitted_evaluations),
                        'iteration':np.asarray(attempted_iteration),
                        'initial_evaluations':np.asarray(initial_evaluations)},stopping.history)

    if kernel == 'coil-set':
        base_kernel = CoilSetKernel(n_coils=K,normalize=True,chunk_size=200)
    else:
        n_features = mapping(X_train).shape[-1]
        base_kernel = (MaternKernel(nu=nu, ard_num_dims=n_features) if kernel == 'matern'
                       else RBFKernel(ard_num_dims=n_features))
    covar_module = ScaleKernel(base_kernel) # Fresh parameters at the start of every run.
    save_checkpoint('fit_pending')
    try:
        gp = _fit_model(X_train, y_train, train_yvar, mapping, covar_module)
    except ModelFittingError as error:
        save_checkpoint('model_fitting_failed',str(error))
        raise
    save_checkpoint('ready')
    model_checks = []

    def check_model(evaluations):
        report = assess_model(gp, mapping, calibration)
        report['evaluations'] = evaluations
        model_checks.append(report)
        print(f"GP check: model_ok={report['model_ok']}; "
              f"outputscale={report.get('outputscale')}; single-run noise={noise:.6g}; "
              f"predicted calibration range={report.get('predicted_mean_range')}.")
        if not report['model_ok']:
            print('GP check failed: ' + ' '.join(report['reasons']))
            stopping.consecutive = 0
        if model_callback is not None:
            model_callback(model_checks)
        return report['model_ok']

    check_model(0)

    warnings_counter = 0
    stop_reason = 'iteration_budget'
    # Matérn retains the original KG acquisition. Subtract the current optimum
    # so its reported value is an improvement, rather than a future value level.
    current_best_mean = _best_posterior_mean(gp, mapping) if kernel == 'matern' else None
    for iteration in range(iterations):
        attempted_iteration = iteration+1
        model = InputMappedModel(gp, mapping)
        if kernel == 'matern':
            acquisition = qKnowledgeGradient(
                model=model, num_fantasies=2,
                current_value=torch.tensor(current_best_mean, dtype=torch.double, device=device),
            )
            candidate, acquisition_value = optimize_acqf(
                acq_function=acquisition, bounds=unit_bounds, q=Q,
                num_restarts=40, raw_samples=512,
                ic_generator=gen_one_shot_kg_initial_conditions,
            )
        else:
            # Keep the actual GP visible to BoTorch so baseline Cholesky caching works.
            acquisition = InputMappedAcquisition(qLogNoisyExpectedImprovement(
                model=gp, X_baseline=mapping(X_train), cache_root=True),mapping)
            candidate, acquisition_value = optimize_acqf(
                acq_function=acquisition, bounds=unit_bounds, q=Q,
                num_restarts=60, raw_samples=1024,
            )
        # Retain the acquisition value BEFORE fitting the updated GP.
        acquisition_scalar = acquisition_value.detach().item()
        cand = candidate[:1].detach().clone()
        del acquisition, model, candidate, acquisition_value
        if duplicate_safe_candidate(X_train, cand):
            print("Duplicate candidate returned by optimizer — skipping this iteration")
            continue

        physical = denormalize(cand).detach().cpu().numpy()
        centers_next, normals_next = unpack_configuration(physical[0])
        energy_next, var_next = objective_function(next_optimization_seed(), centers_next, normals_next,
                                                 log_variance=noise)
        if not np.isfinite([energy_next, var_next]).all() or var_next < 0:
            warnings.warn(f"Invalid response or variance at iteration {iteration+1}. Skipping.")
            warnings_counter += 1
            if warnings_counter > iterations // 3:
                raise ValueError("Too many invalid simulator observations encountered.")
            continue

        # Append inputs, negative log-responses and variances together, only if valid.
        X_train = torch.vstack((X_train, cand))
        y_train = np.append(y_train, -float(energy_next))
        train_yvar = torch.cat((train_yvar, torch.tensor(
            [[noise]], dtype=torch.double, device=device)))
        evaluations += 1
        # Save the expensive observation before fitting; diagnostics are pending.
        acquisition_hist.append(acquisition_scalar)
        best_mean_hist.append(float('nan'))
        variance_hist.append(float('nan'))
        save_checkpoint('fit_pending')
        try:
            gp = _fit_model(X_train, y_train, train_yvar, mapping, covar_module)
        except ModelFittingError as error:
            save_checkpoint('model_fitting_failed',str(error))
            raise
        fitted_evaluations = evaluations

        # Acquisition values stay in their native units: KG or log-NEI.
        current_best_mean = _best_posterior_mean(gp, mapping)
        best_mean_hist[-1] = current_best_mean
        top_idx = torch.as_tensor(np.argsort(y_train)[-min(5, len(y_train)):], device=device)
        with torch.no_grad():
            posterior_var = gp.posterior(mapping(X_train[top_idx])).variance.mean().item()
        variance_hist[-1] = posterior_var
        model_ok = check_model(evaluations)
        should_stop = stopping.check(gp, mapping, X_train, evaluations, iteration+1, model_ok=model_ok)
        if should_stop:
            stop_reason = 'estimated_simple_regret'
        save_checkpoint('ready')
        if should_stop:
            print(f"Estimated simple-regret criterion reached after {evaluations} new evaluations.")
            break
        print(f"Iteration {iteration+1}/{iterations}, Energy: {np.exp(energy_next):.4f}")
    print(f"BO loop completed with {warnings_counter} invalid-observation warnings.")
    result = (X_train.cpu().numpy(), y_train, np.asarray(acquisition_hist),
              best_mean_hist, variance_hist)
    if return_diagnostics:
        incumbent, _ = lower_bound_incumbent(gp, mapping, X_train, stopping.confidence)
        diagnostics = {'settings':stopping.settings(), 'checks':stopping.history,
                       'stop_reason':stop_reason, 'successful_evaluations':evaluations,
                       'attempted_iterations':iteration+1, 'incumbent_index':incumbent,
                       'model_checks':model_checks, 'noise_calibration':calibration}
        return (*result, diagnostics)
    return result


def bo_matern_kernel(nu=2.5, max_iter=None, initial_data=None, return_diagnostics=False, checkpoint=None,
                     calibration=None, model_callback=None):
    """Matérn ARD GP on map_fn features, with knowledge gradient (two fantasies)."""
    return _bo_loop('matern', map_fn, max_iter, initial_data, nu, return_diagnostics, checkpoint,
                    calibration, model_callback)


def bo_rbf_kernel(max_iter=None, initial_data=None, return_diagnostics=False, checkpoint=None,
                  calibration=None, model_callback=None):
    """RBF ARD GP on pooled Cartesian coil features, with log noisy EI."""
    return _bo_loop('rbf', map_set, max_iter, initial_data, return_diagnostics=return_diagnostics,
                    checkpoint=checkpoint,calibration=calibration,model_callback=model_callback)


def bo_rbf_old(max_iter=None, initial_data=None, return_diagnostics=False, checkpoint=None,
               calibration=None, model_callback=None):
    """RBF ARD GP on map_fn features, with log noisy EI (legacy function name)."""
    return _bo_loop('rbf', map_fn, max_iter, initial_data, return_diagnostics=return_diagnostics,
                    checkpoint=checkpoint,calibration=calibration,model_callback=model_callback)


def bo_coil_set_kernel(max_iter=None, initial_data=None, return_diagnostics=False, checkpoint=None,
                       calibration=None, model_callback=None):
    """Exact Kernels_v3 coil-set GP on Cartesian coils, with log noisy EI.

    map_coilwise converts physical geometry; USE_FEATURE_MAPPING does not apply.
    """
    return _bo_loop('coil-set', map_coilwise, max_iter, initial_data,
                    return_diagnostics=return_diagnostics,checkpoint=checkpoint,
                    calibration=calibration,model_callback=model_callback)
