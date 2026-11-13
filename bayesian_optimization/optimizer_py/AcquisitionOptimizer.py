"""Memory-limited acquisition optimization with bounded numerical recovery."""
import warnings
import torch
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from botorch.acquisition import OneShotAcquisitionFunction
from botorch.exceptions.warnings import OptimizationWarning

from input import BO_OPT_BATCH_LIMIT, BO_OPT_INIT_BATCH_LIMIT


def _optimizer_warning(warning):
    return (issubclass(warning.category, OptimizationWarning)
            or (issubclass(warning.category, RuntimeWarning)
                and 'Optimization failed' in str(warning.message)))


def _finite_result(result, bounds):
    candidate, value = result
    return (torch.isfinite(candidate).all().item() and torch.isfinite(value).all().item()
            and (candidate >= bounds[0]-1e-7).all().item()
            and (candidate <= bounds[1]+1e-7).all().item())


def optimize_acqf(*, require_convergence=False, **kwargs):
    """Keep every requested restart, processing small batches.

    Retry warned L-BFGS-B runs once with SLSQP. Ordinary BO may use the best
    finite solution/grid fallback; regret UCB checks require a clean optimizer.
    """
    if min(BO_OPT_BATCH_LIMIT, BO_OPT_INIT_BATCH_LIMIT) < 1:
        raise ValueError('Acquisition batch limits must be positive.')
    options = dict(kwargs.pop('options', None) or {})
    options['batch_limit'] = min(options.get('batch_limit',BO_OPT_BATCH_LIMIT),BO_OPT_BATCH_LIMIT)
    options['init_batch_limit'] = min(options.get('init_batch_limit',BO_OPT_INIT_BATCH_LIMIT),BO_OPT_INIT_BATCH_LIMIT)
    options.setdefault('maxiter',200)
    options.setdefault('maxls',50)
    # The KG initializer launches an internal optimizer without forwarding batch
    # limits. Bound that internal problem as well; outer restart counts are unchanged.
    if isinstance(kwargs['acq_function'],OneShotAcquisitionFunction):
        options['num_inner_restarts'] = min(options.get('num_inner_restarts',2),2)
        options['raw_inner_samples'] = min(options.get('raw_inner_samples',16),16)
    kwargs['retry_on_optimization_warning'] = False
    best = None
    name = type(getattr(kwargs['acq_function'],'base_acqf',kwargs['acq_function'])).__name__
    for attempt in range(2):
        attempt_options = dict(options)
        if attempt:
            attempt_options.update(method='SLSQP',batch_limit=1,
                                   init_batch_limit=min(options['init_batch_limit'],8),ftol=1e-8)
            for key in ('maxls','maxcor','gtol'):
                attempt_options.pop(key,None)
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always',OptimizationWarning)
                warnings.simplefilter('always',RuntimeWarning)
                result = botorch_optimize_acqf(**kwargs,options=attempt_options)
            warned = any(_optimizer_warning(warning) for warning in captured)
            for warning in captured:
                if not _optimizer_warning(warning):
                    warnings.warn_explicit(warning.message,warning.category,warning.filename,warning.lineno)
            valid = _finite_result(result,kwargs['bounds'])
            if valid:
                result = tuple(value.detach() for value in result)
                if best is None or result[1].max() > best[1].max():
                    best = result
                if not warned:
                    return result if require_convergence else best
        except (RuntimeError,ValueError) as error:
            print(f'{name}: numerical optimizer failure: {error}')
        if not attempt:
            print(f'{name}: retrying with SLSQP in batches of one.')
    if require_convergence:
        raise RuntimeError(f'{name}: both optimizer attempts failed or warned; regret check is inconclusive.')
    print(f'{name}: using a finite candidate/grid fallback after optimizer failures.')
    # This is a search fallback, never evidence for regret-based stopping.
    acquisition = kwargs['acq_function']
    q = kwargs['q']
    one_shot = isinstance(acquisition,OneShotAcquisitionFunction)
    augmented_q = acquisition.get_augmented_q_batch_size(q) if one_shot else q
    bounds = kwargs['bounds']
    sobol = torch.quasirandom.SobolEngine(augmented_q*bounds.shape[-1],scramble=True)
    points = sobol.draw(128).to(bounds).reshape(128,augmented_q,-1)
    points = bounds[0] + (bounds[1]-bounds[0])*points
    with torch.no_grad():
        for batch in points.split(BO_OPT_INIT_BATCH_LIMIT):
            values = acquisition(batch).reshape(-1)
            finite = torch.isfinite(values)
            if finite.any():
                index = values.masked_fill(~finite,-torch.inf).argmax()
                point = batch[index]
                if one_shot:
                    point = acquisition.extract_candidates(point)
                result = point.detach(), values[index].detach()
                if best is None or result[1] > best[1].max():
                    best = result
    if best is None:
        raise RuntimeError(f'{name}: no finite feasible candidate found.')
    return best
