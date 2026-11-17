"""Single-loop Bayesian optimization with Matérn, RBF and coil-set kernels."""
import warnings
import json
from time import perf_counter
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
from RobustGPFitting import fit_gpytorch_mll_robust as fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement, PosteriorMean, qKnowledgeGradient
from KGInitialization import gen_one_shot_kg_initial_conditions
from AcquisitionOptimizer import optimize_acqf
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.settings import min_fixed_noise
from AcquisitionStarts import optimize_acquisition, observed_local_points, new_point_mask
from CoilSetModel import select_coil_scales, make_coil_kernel
from PosteriorChecks import raw_mean_variance, check_kernel

from BoUtils import (
    denormalize, unpack_configuration, map_fn, map_set, map_coilwise,
    InputMappedAcquisition, InputMappedModel, sobol_sample,
    duplicate_safe_candidate,
)
from RegretStopping import SimpleRegretStopping, lower_bound_incumbent
from Objective import objective_function, next_optimization_seed, optimization_seed_state
from NoiseCalibration import calibrate_noise
from ModelDiagnostics import assess_model
from GPConditioning import condition_on_observations
from FastCoilAcquisition import CheckedLogExpectedImprovement
from input import (device, MAX_ITER, unit_bounds, Q, BO_NOISE_REPEATS, K, SEED,
                   BO_COIL_LENGTHSCALES, BO_COIL_KAPPAS, BO_COIL_SELECT_EVERY,
                   BO_COIL_MIN_SELECTION, BO_POSTERIOR_BATCH_LIMIT,
                   BO_OPT_BATCH_LIMIT, BO_OPT_INIT_BATCH_LIMIT,
                   BO_COIL_REFIT_EVERY, BO_COIL_ACQUISITION, BO_COIL_RESTARTS,
                   BO_COIL_RAW_SAMPLES, BO_COIL_OPT_MAXITER)


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
    check_kernel(gp, gp.train_inputs[0])
    return gp


def _observed_means(gp, mapping, X):
    with torch.no_grad():
        return torch.cat([raw_mean_variance(gp,mapping(batch))[0].reshape(-1)
                          for batch in X.split(BO_POSTERIOR_BATCH_LIMIT)])


def _best_posterior_mean(gp, mapping, X):
    ranking = _observed_means(gp,mapping,X)
    _, value = optimize_acquisition(
        acq_function=InputMappedAcquisition(PosteriorMean(gp),mapping),
        bounds=unit_bounds,X=X,ranking=ranking,num_restarts=30,raw_samples=256,
        optimizer=optimize_acqf,allow_observed=True)
    return value.detach().item()


def _kg_candidate(acquisition, gp, mapping, X):
    """Keep fantasy coordinates intact and return a distinct real candidate."""
    starts = gen_one_shot_kg_initial_conditions(
        acq_function=acquisition,bounds=unit_bounds,q=Q,num_restarts=40,raw_samples=512,
        options={'batch_limit':BO_OPT_BATCH_LIMIT,'init_batch_limit':BO_OPT_INIT_BATCH_LIMIT,
                 'num_inner_restarts':2,'raw_inner_samples':16})
    starts[:20,:Q,:] = observed_local_points(
        unit_bounds,X,_observed_means(gp,mapping,X),20)
    candidate,value = optimize_acqf(acq_function=acquisition,bounds=unit_bounds,q=Q,
        num_restarts=40,raw_samples=512,batch_initial_conditions=starts)
    if (candidate.shape == (1,X.shape[-1]) and torch.isfinite(candidate).all()
            and (candidate >= unit_bounds[0]).all() and (candidate <= unit_bounds[1]).all()
            and not duplicate_safe_candidate(X,candidate)):
        return candidate,value

    # Evaluate full one-shot trees: scoring extracted real points alone is not KG.
    eligible = new_point_mask(starts[:,0,:],X)
    trees = starts[eligible]
    if len(trees):
        with torch.no_grad():
            values = torch.cat([acquisition(batch).reshape(-1)
                                for batch in trees.split(BO_OPT_INIT_BATCH_LIMIT)])
        finite = torch.isfinite(values)
        if finite.any():
            best = int(values.masked_fill(~finite,-torch.inf).argmax().item())
            return trees[best,:Q,:].detach(),values[best].detach()
    raise RuntimeError('KG optimization and initial trees contain no finite unevaluated candidate.')


def _bo_loop(kernel, mapping, max_iter=None, initial_data=None, nu=2.5, return_diagnostics=False,
             checkpoint=None, calibration=None, model_callback=None):
    iterations = MAX_ITER if max_iter is None else int(max_iter)
    if kernel not in ('matern','rbf','coil-set'):
        raise ValueError(f'Unknown vanilla BO kernel: {kernel}')
    if iterations < 1 or Q != 1:
        raise ValueError("Use a positive iteration budget and Q=1 for these single-point loops.")
    if BO_COIL_SELECT_EVERY < 1:
        raise ValueError('BO_COIL_SELECT_EVERY must be positive.')
    if kernel == 'coil-set' and (min(BO_COIL_REFIT_EVERY,BO_COIL_RESTARTS,
            BO_COIL_RAW_SAMPLES,BO_COIL_OPT_MAXITER) < 1
            or BO_COIL_ACQUISITION not in ('log-ei','log-nei')):
        raise ValueError('Coil CPU budgets must be positive; acquisition must be log-ei or log-nei.')
    stopping = SimpleRegretStopping(fast_record=kernel == 'coil-set')

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
    gp = None
    scale_selection = None
    selection_evaluation = 0
    hyperparameter_evaluations = 0
    timings = []

    def save_checkpoint(status, error=''):
        if checkpoint is not None:
            state = {'X':X_train.cpu().numpy(), 'y':y_train,
                        'variance':train_yvar.squeeze(-1).cpu().numpy(),
                        'acquisition':np.asarray(acquisition_hist),
                        'posterior_mean':np.asarray(best_mean_hist),
                        'posterior_variance':np.asarray(variance_hist),
                        'status':np.asarray(status), 'error':np.asarray(error),
                        'evaluations':np.asarray(evaluations),
                        'fitted_evaluations':np.asarray(fitted_evaluations),
                        'iteration':np.asarray(attempted_iteration),
                        'initial_evaluations':np.asarray(initial_evaluations),
                        'kernel':np.asarray(kernel)}
            state['next_spatial_seed'] = np.asarray(optimization_seed_state())
            state['hyperparameter_evaluations'] = np.asarray(hyperparameter_evaluations)
            state['timings_json'] = np.asarray(json.dumps(timings,allow_nan=False))
            if scale_selection is not None:
                state['scale_selection_json'] = np.asarray(json.dumps(scale_selection,allow_nan=False))
            if gp is not None:
                # Pure arrays preserve the exact last successful fit without pickle.
                state.update({'gp_state__'+name:value.detach().cpu().numpy()
                              for name,value in gp.state_dict().items() if torch.is_tensor(value)})
            checkpoint(state,stopping.history)

    if kernel == 'coil-set':
        covar_module = make_coil_kernel(K)
    else:
        n_features = mapping(X_train).shape[-1]
        base_kernel = (MaternKernel(nu=nu, ard_num_dims=n_features) if kernel == 'matern'
                       else RBFKernel(ard_num_dims=n_features))
        covar_module = ScaleKernel(base_kernel)

    def refit():
        nonlocal covar_module, scale_selection, selection_evaluation
        if kernel == 'coil-set' and (scale_selection is None
                or evaluations-selection_evaluation >= BO_COIL_SELECT_EVERY):
            ell,kappa,scale_selection = select_coil_scales(
                mapping(X_train),y_train,train_yvar,K,lengthscales=BO_COIL_LENGTHSCALES,
                kappas=BO_COIL_KAPPAS,seed=SEED,min_points=BO_COIL_MIN_SELECTION)
            selection_evaluation = evaluations
            if scale_selection['predictive_ok'] is False:
                raise ModelFittingError('No coil-set scale candidate improves the constant-mean '
                                        'validation baseline. Observations are saved; revise the model '
                                        'before spending further simulation evaluations.')
            covar_module = make_coil_kernel(K,ell,kappa)
        return _fit_model(X_train,y_train,train_yvar,mapping,covar_module)

    save_checkpoint('fit_pending')
    initial_fit_start = perf_counter()
    try:
        gp = refit()
    except ModelFittingError as error:
        save_checkpoint('model_fitting_failed',str(error))
        raise
    initial_fit_seconds = perf_counter()-initial_fit_start
    print(f'Initial GP fit and scale selection: {initial_fit_seconds:.2f}s.')
    save_checkpoint('ready')
    model_checks = []

    def check_model(evaluations):
        report = assess_model(gp, mapping, calibration)
        report['evaluations'] = evaluations
        base = gp.covar_module.base_kernel
        report['lengthscale'] = base.lengthscale.detach().cpu().reshape(-1).tolist()
        if hasattr(base,'kappa'):
            report['kappa'] = float(base.kappa.item())
        if scale_selection is not None:
            report['selection_skill'] = scale_selection['selection_skill']
            report['selection_evaluation'] = selection_evaluation
            report['predictive_ok'] = scale_selection['predictive_ok']
            if evaluations == selection_evaluation:
                report['scale_selection'] = scale_selection
            if scale_selection['predictive_ok'] is not True:
                report['model_ok'] = False
                report['reasons'].append('Insufficient observations to establish predictive skill.')
        model_checks.append(report)
        print(f"GP check: model_ok={report['model_ok']}; "
              f"outputscale={report.get('outputscale')}; single-run noise={noise:.6g}; "
              f"lengthscale={report['lengthscale']}; kappa={report.get('kappa')}; "
              f"selection skill={report.get('selection_skill')}; "
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
    current_best_mean = _best_posterior_mean(gp,mapping,X_train) if kernel == 'matern' else None
    for iteration in range(iterations):
        iteration_start = perf_counter()
        acquisition_start = iteration_start
        attempted_iteration = iteration+1
        model = InputMappedModel(gp, mapping)
        if kernel == 'matern':
            acquisition = qKnowledgeGradient(
                model=model, num_fantasies=2,
                current_value=torch.tensor(current_best_mean, dtype=torch.double, device=device),
            )
            candidate, acquisition_value = _kg_candidate(acquisition,gp,mapping,X_train)
        else:
            ranking = _observed_means(gp,mapping,X_train)
            if kernel == 'coil-set' and BO_COIL_ACQUISITION == 'log-ei':
                # The incumbent is a denoised plug-in estimate, not the noisiest
                # observed score. Exact latent uncertainty still drives exploration.
                base_acquisition = CheckedLogExpectedImprovement(gp,best_f=ranking.max())
            else:
                # Keep the actual GP visible for baseline Cholesky caching.
                base_acquisition = qLogNoisyExpectedImprovement(
                    model=gp,X_baseline=mapping(X_train),cache_root=True)
            acquisition = InputMappedAcquisition(base_acquisition,mapping)
            budget = (dict(num_restarts=BO_COIL_RESTARTS,raw_samples=BO_COIL_RAW_SAMPLES,
                           options={'maxiter':BO_COIL_OPT_MAXITER}) if kernel == 'coil-set'
                      else dict(num_restarts=60,raw_samples=1024))
            candidate, acquisition_value = optimize_acquisition(
                acq_function=acquisition, bounds=unit_bounds, q=Q,
                X=X_train,ranking=ranking,**budget,optimizer=optimize_acqf,allow_observed=False,
            )
        # Retain the acquisition value BEFORE fitting the updated GP.
        acquisition_scalar = acquisition_value.detach().item()
        cand = candidate[:1].detach().clone()
        del acquisition, model, candidate, acquisition_value
        if kernel != 'matern':
            del base_acquisition
        if duplicate_safe_candidate(X_train, cand):
            print("Duplicate candidate returned by optimizer — skipping this iteration")
            continue

        acquisition_seconds = perf_counter()-acquisition_start
        simulation_start = perf_counter()
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
        simulation_seconds = perf_counter()-simulation_start

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
        update_start = perf_counter()
        full_refit = (kernel != 'coil-set'
                      or evaluations-hyperparameter_evaluations >= BO_COIL_REFIT_EVERY
                      or evaluations-selection_evaluation >= BO_COIL_SELECT_EVERY)
        try:
            if full_refit:
                gp = refit()
            else:
                try:
                    gp = condition_on_observations(gp,mapping(X_train),y_train,train_yvar)
                except (ModelFittingError,RuntimeError,ValueError) as error:
                    print(f'Conditioning failed; refitting on all observations: {error}')
                    gp = refit()
                    full_refit = True
        except ModelFittingError as error:
            save_checkpoint('model_fitting_failed',str(error))
            raise
        fitted_evaluations = evaluations
        if full_refit:
            hyperparameter_evaluations = evaluations
        update_seconds = perf_counter()-update_start
        diagnostic_start = perf_counter()

        # Acquisition values stay in their native units: KG, log-EI or log-NEI.
        # Coil diagnostics report the best evaluated posterior mean. A global
        # continuous optimization just to draw this plot can dominate the run.
        current_best_mean = (float(_observed_means(gp,mapping,X_train).max())
                             if kernel == 'coil-set' else _best_posterior_mean(gp,mapping,X_train))
        best_mean_hist[-1] = current_best_mean
        top_idx = torch.as_tensor(np.argsort(y_train)[-min(5, len(y_train)):], device=device)
        with torch.no_grad():
            posterior_var = raw_mean_variance(gp,mapping(X_train[top_idx]))[1].mean().item()
        variance_hist[-1] = posterior_var
        model_ok = check_model(evaluations)
        should_stop = stopping.check(gp, mapping, X_train, evaluations, iteration+1, model_ok=model_ok)
        if should_stop:
            stop_reason = 'estimated_simple_regret'
        timing = dict(evaluations=evaluations,acquisition_seconds=acquisition_seconds,
                      simulation_seconds=simulation_seconds,update_seconds=update_seconds,
                      diagnostic_seconds=perf_counter()-diagnostic_start,full_refit=bool(full_refit),
                      total_seconds=perf_counter()-iteration_start)
        timings.append(timing)
        print(f"CPU timing: acquisition={timing['acquisition_seconds']:.2f}s; "
              f"model {'refit' if full_refit else 'conditioning'}={update_seconds:.2f}s; "
              f"diagnostics={timing['diagnostic_seconds']:.2f}s; simulation={simulation_seconds:.2f}s.")
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
                       'model_checks':model_checks, 'noise_calibration':calibration,
                       'iteration_timings':timings,
                       'initial_fit_seconds':initial_fit_seconds,
                       'acquisition':BO_COIL_ACQUISITION if kernel == 'coil-set' else
                                     'kg' if kernel == 'matern' else 'log-nei',
                       'hyperparameter_refit_every':BO_COIL_REFIT_EVERY if kernel == 'coil-set' else 1,
                       'posterior_mean_scope':'evaluated_points' if kernel == 'coil-set' else 'continuous_search'}
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
    """Exact coil-set GP with fast q=1 acquisition and periodic hyperparameter fits.

    map_coilwise converts physical geometry; USE_FEATURE_MAPPING does not apply.
    """
    return _bo_loop('coil-set', map_coilwise, max_iter, initial_data,
                    return_diagnostics=return_diagnostics,checkpoint=checkpoint,
                    calibration=calibration,model_callback=model_callback)
