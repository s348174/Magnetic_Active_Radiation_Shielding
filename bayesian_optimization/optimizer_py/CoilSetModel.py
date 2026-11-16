"""Choose fixed physical coil-set scales by predictive validation, not likelihood alone."""
import math
import warnings

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import min_fixed_noise

from Kernels_v3 import CoilSetKernel
from PosteriorChecks import raw_mean_variance
from RobustGPFitting import fit_gpytorch_mll_robust


def make_coil_kernel(n_coils, lengthscale=.5, kappa=1.):
    """Fixed geometry scales; outputscale and GP constant mean remain trainable."""
    kernel = ScaleKernel(CoilSetKernel(n_coils=n_coils, normalize=True, chunk_size=200)).double()
    kernel.base_kernel.initialize(lengthscale=lengthscale, kappa=kappa)
    kernel.base_kernel.raw_lengthscale.requires_grad_(False)
    kernel.base_kernel.raw_kappa.requires_grad_(False)
    return kernel


def _fit_candidate(X, y, variance, n_coils, lengthscale, kappa):
    with warnings.catch_warnings(), min_fixed_noise(double_value=1e-12):
        warnings.filterwarnings('ignore', category=InputDataWarning)
        gp = SingleTaskGP(X, y, train_Yvar=variance,
                          covar_module=make_coil_kernel(n_coils, lengthscale, kappa).to(X),
                          outcome_transform=Standardize(m=1))
    fit_gpytorch_mll_robust(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def select_coil_scales(X, y, train_yvar, n_coils, lengthscales=(.25,.5,1.,2.),
                       kappas=(.25,1.,4.), seed=67, min_points=25):
    """Fit each fixed-scale candidate on 80%, choose predictive RMSE on 20%.

    X already contains map_coilwise features. Only amplitude and mean are fitted.
    The selection fold tunes scales and is NOT independent evidence of accuracy;
    keep any final outer validation observations out of this function entirely.
    """
    X = torch.as_tensor(X, dtype=torch.double).detach()
    y = torch.as_tensor(y, dtype=X.dtype, device=X.device).reshape(-1,1).detach()
    variance = torch.as_tensor(train_yvar, dtype=X.dtype, device=X.device).detach()
    if variance.numel() == 1:
        variance = variance.expand(len(X)).reshape(-1,1)
    else:
        variance = variance.reshape(-1,1)
    if (X.ndim != 2 or X.shape[1] != 6*n_coils or y.shape != (len(X),1)
            or variance.shape != y.shape or len(X) < 2 or not torch.isfinite(X).all()
            or not torch.isfinite(y).all() or not torch.isfinite(variance).all()
            or (variance <= 0).any()):
        raise ValueError('Coil scale selection needs finite coilwise features, scores and positive variances.')
    if (not lengthscales or not kappas
            or any(not math.isfinite(v) or v <= 0 for v in (*lengthscales,*kappas))):
        raise ValueError('Coil geometry scale candidates must be positive and finite.')
    report = {'method':'fixed_geometry_scale_validation', 'split_seed':int(seed),
              'samples':len(X), 'train_count':len(X), 'validation_count':0,
              'baseline_rmse':None, 'best_rmse':None, 'selection_skill':None,
              'predictive_ok':None, 'candidates':[],
              'independent_validation':False}
    if len(X) < min_points:
        lengthscale = min(lengthscales, key=lambda value: abs(value-.5))
        kappa = min(kappas, key=lambda value: abs(value-1.))
        report.update(lengthscale=float(lengthscale), kappa=float(kappa),
                      reason=f'Fewer than {min_points} observations; predictive selection unavailable.')
        return float(lengthscale), float(kappa), report

    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    validation_count = max(5, int(math.ceil(.2*len(X))))
    validation = order[:validation_count].to(X.device)
    training = order[validation_count:].to(X.device)
    baseline = y[training].mean()
    baseline_mse = float((y[validation]-baseline).square().mean())
    report.update(train_count=len(training), validation_count=len(validation),
                  train_indices=training.cpu().tolist(), validation_indices=validation.cpu().tolist(),
                  baseline_rmse=math.sqrt(baseline_mse))
    best = None
    for lengthscale in lengthscales:
        for kappa in kappas:
            row = {'lengthscale':float(lengthscale), 'kappa':float(kappa),
                   'rmse':None, 'valid':False}
            try:
                gp = _fit_candidate(X[training], y[training], variance[training], n_coils,
                                    lengthscale, kappa)
                with torch.no_grad():
                    prediction, _ = raw_mean_variance(gp, X[validation])
                    mse = float((prediction-y[validation]).square().mean())
                if not math.isfinite(mse):
                    raise ModelFittingError('Non-finite coil validation error.')
                row.update(rmse=math.sqrt(mse), valid=True)
                if best is None or mse < best[0]:
                    best = (mse, float(lengthscale), float(kappa))
            except (ModelFittingError, RuntimeError, np.linalg.LinAlgError) as error:
                row['error'] = str(error)
            report['candidates'].append(row)
    if best is None:
        raise ModelFittingError('All fixed-scale coil GP candidates failed predictive validation.')
    mse, lengthscale, kappa = best
    skill = 1-mse/baseline_mse if baseline_mse > 0 else None
    report.update(lengthscale=lengthscale, kappa=kappa, best_rmse=math.sqrt(mse),
                  selection_skill=skill, predictive_ok=bool(skill is not None and skill > 0))
    print(f'Coil scale selection: ell={lengthscale:g} m; kappa={kappa:g}; '
          f'RMSE={math.sqrt(mse):.6g}; constant baseline={math.sqrt(baseline_mse):.6g}; '
          f'predictive_ok={report["predictive_ok"]}.')
    return lengthscale, kappa, report
