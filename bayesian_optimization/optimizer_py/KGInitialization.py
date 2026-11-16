"""Memory-bounded initialization for the vanilla single-output one-shot KG."""
from botorch.acquisition import PosteriorMean
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.sampling import boltzmann_sample

from AcquisitionOptimizer import optimize_acqf
from input import BO_OPT_BATCH_LIMIT, BO_OPT_INIT_BATCH_LIMIT


def gen_one_shot_kg_initial_conditions(acq_function, bounds, q, num_restarts,
                                       raw_samples, options=None):
    """Use BoTorch's KG initialization scheme with bounded inner optimization.

    BoTorch's standard initializer does not forward batch options to its inner
    fantasy-point optimizer. This single-output version bounds both stages.
    """
    if getattr(acq_function,'objective',None) is not None or getattr(acq_function,'project',None) is not None:
        raise ValueError('This initializer supports the vanilla single-output KG only.')
    if min(BO_OPT_BATCH_LIMIT,BO_OPT_INIT_BATCH_LIMIT) < 1:
        raise ValueError('KG initialization batch limits must be positive.')
    options = dict(options or {})
    options['batch_limit'] = min(options.get('batch_limit',BO_OPT_BATCH_LIMIT),BO_OPT_BATCH_LIMIT)
    options['init_batch_limit'] = min(options.get('init_batch_limit',BO_OPT_INIT_BATCH_LIMIT),BO_OPT_INIT_BATCH_LIMIT)
    fraction = options.get('frac_random',.1)
    if not 0 < fraction < 1:
        raise ValueError('KG frac_random must be between zero and one.')
    augmented_q = acq_function.get_augmented_q_batch_size(q=q)
    starts = gen_batch_initial_conditions(acq_function=acq_function,bounds=bounds,
        q=augmented_q,num_restarts=num_restarts,raw_samples=raw_samples,options=options)
    if starts is None:
        raise RuntimeError('KG initialization produced no restart points.')
    count = int((1-fraction)*(augmented_q-q))
    if count == 0:
        return starts

    value_function = PosteriorMean(acq_function.model,
        posterior_transform=getattr(acq_function,'posterior_transform',None))
    # Keep the requested counts while limiting their simultaneous processing.
    candidates,values = optimize_acqf(acq_function=value_function,bounds=bounds,q=1,
        num_restarts=options.get('num_inner_restarts',2),
        raw_samples=options.get('raw_inner_samples',16),return_best_only=False,
        options={name:options[name] for name in ('batch_limit','init_batch_limit')})
    candidates = candidates.reshape(-1,bounds.shape[-1])
    values = values.reshape(-1)
    if len(candidates) != len(values):
        raise RuntimeError('KG fantasy candidates and values have inconsistent shapes.')
    indices = boltzmann_sample(function_values=values,num_samples=num_restarts*count,
                               eta=options.get('eta',2.),replacement=True)
    starts[..., -count:, :] = candidates[indices].reshape(num_restarts,count,bounds.shape[-1])
    return starts
