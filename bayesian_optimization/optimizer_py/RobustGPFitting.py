"""Bounded GP fitting retries with a line-search-free fallback."""
from copy import deepcopy
from functools import partial
import math

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.optim.core import OptimizationStatus
from botorch.optim.fit import fit_gpytorch_mll_torch


def _fresh_parameters(mll):
    """Reset in place: the BO loop retains a reference to this kernel."""
    mll.train()
    mll.model.covar_module.initialize(outputscale=1.)
    mll.model.covar_module.base_kernel.initialize(lengthscale=1.)
    mll.model.mean_module.initialize(constant=0.)


def _validate_fit(mll):
    if any(not torch.isfinite(parameter).all() for parameter in mll.parameters()):
        raise ModelFittingError('GP fitting produced non-finite parameters.')
    # Evaluate the training likelihood, not a cached posterior from a previous fit.
    mll.train()
    with torch.no_grad():
        output = mll.model(*mll.model.train_inputs)
        value = mll(output, mll.model.train_targets)
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise ModelFittingError('GP fitting produced a non-finite marginal likelihood.')
    mll.eval()


def _fit_adam(mll, **kwargs):
    result = fit_gpytorch_mll_torch(
        mll, step_limit=300, optimizer=partial(torch.optim.Adam, lr=.05), **kwargs)
    if (result.status not in (OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED)
            or not math.isfinite(result.fval)):
        raise ModelFittingError('Adam GP fitting failed or produced a non-finite loss.')
    # A bounded Adam run may stop at its budget; this is not a convergence claim.
    print(f'Adam GP fitting finished after {result.step} steps: {result.message}')
    return result


def fit_gpytorch_mll_robust(mll):
    """Try warm L-BFGS-B, a fresh start, then bounded Adam; never return a bad fit."""
    original = deepcopy(mll.state_dict())
    original_training = mll.training
    attempts = ('warm-start L-BFGS-B', 'fresh-start L-BFGS-B', 'fresh-start Adam')
    last_error = None
    for attempt, name in enumerate(attempts):
        try:
            # Discard all failed updates before trying another initialization.
            mll.load_state_dict(original)
            mll.train()
            if attempt:
                _fresh_parameters(mll)
            if attempt < 2:
                fit_gpytorch_mll(mll, max_attempts=1,
                                 optimizer_kwargs={'options': {'maxiter':300, 'maxls':50}})
            else:
                fit_gpytorch_mll(mll, max_attempts=1, optimizer=_fit_adam)
            _validate_fit(mll)
            return mll
        except (ModelFittingError, RuntimeError, np.linalg.LinAlgError) as error:
            last_error = error
            print(f'GP fitting failed ({name}): {error}')
    mll.load_state_dict(original)
    mll.train(original_training)
    raise ModelFittingError(
        'GP fitting failed with warm-start L-BFGS-B, fresh-start L-BFGS-B and Adam. '
        'No failed parameters were accepted.'
    ) from last_error
