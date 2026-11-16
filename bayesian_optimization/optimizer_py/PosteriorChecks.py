"""Check GP covariance before using its uncertainty for fitting or stopping."""
import torch


def _tolerance(scale):
    """Relative roundoff allowance, not an uncertainty floor."""
    finfo = torch.finfo(scale.dtype)
    return (1e-10+64*finfo.eps)*scale.abs().clamp_min(finfo.tiny)


def _check_matrix(matrix, name):
    if not torch.isfinite(matrix).all():
        raise RuntimeError(f'{name} contains non-finite entries.')
    scale = matrix.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1)
    tolerance = _tolerance(scale)
    asymmetry = (matrix-matrix.transpose(-2,-1)).abs().amax(dim=(-2,-1))
    if (asymmetry > tolerance).any():
        raise RuntimeError(f'{name} is materially asymmetric.')
    # Only small joint predictions / representative kernel subsets enter here.
    symmetric = .5*(matrix+matrix.transpose(-2,-1))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    if (eigenvalues[...,0] < -tolerance).any():
        raise RuntimeError(f'{name} is not positive semidefinite.')
    return eigenvalues[...,0]


def _prior_variance(gp, X):
    transformed = gp.transform_inputs(X) if hasattr(gp, 'transform_inputs') else X
    prior = gp.covar_module(transformed, transformed, diag=True).unsqueeze(-1)
    transform = getattr(gp, 'outcome_transform', None)
    if transform is not None:
        if not hasattr(transform, 'stdvs'):
            raise RuntimeError('Posterior checks require an affine outcome transform.')
        prior = prior*transform.stdvs.square()
    if not torch.isfinite(prior).all() or (prior < 0).any():
        raise RuntimeError('GP prior variances are non-finite or negative.')
    return prior


def checked_posterior(gp, X, full_covariance=False):
    """Return (mean, raw variance, optional covariance), preserving gradients.

    Shapes match the single-output BoTorch posterior: (..., q, 1) for mean and
    variance; (..., q, q) for covariance. Material covariance errors raise rather
    than being disguised by GPyTorch's variance accessor floor. Tiny negative
    roundoff is clamped to zero; positive small variances are never raised.
    """
    posterior = gp.posterior(X)
    mean = posterior.mean
    raw = getattr(posterior.distribution, 'lazy_covariance_matrix', None)
    if raw is None:
        raw = posterior.distribution.covariance_matrix
    variance = raw.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    prior = _prior_variance(gp, X)
    tolerance = _tolerance(prior)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise RuntimeError('GP posterior contains non-finite means or variances.')
    if (variance < -tolerance).any():
        raise RuntimeError('GP posterior has materially negative latent variance.')
    if (variance > prior+tolerance).any():
        raise RuntimeError('GP posterior variance materially exceeds its prior variance.')
    variance = variance.clamp_min(0.)
    covariance = None
    if full_covariance:
        covariance = raw.to_dense()
        _check_matrix(covariance, 'GP posterior covariance')
    return mean, variance, covariance


def raw_mean_variance(gp, X):
    """Checked latent mean and variance without GPyTorch's reporting floor."""
    mean, variance, _ = checked_posterior(gp, X)
    return mean, variance


def check_kernel(gp, mapped_X, max_points=64):
    """Check a small kernel subset; never claim this proves global validity."""
    if max_points < 1:
        raise ValueError('Kernel check subset size must be positive.')
    X = mapped_X[..., :max_points, :]
    if X.shape[-2] < 1:
        raise RuntimeError('Kernel check needs at least one point.')
    with torch.no_grad():
        full = gp.covar_module(X, X).to_dense()
        minimum_eigenvalue = _check_matrix(full, 'GP kernel matrix')
        diagonal = gp.covar_module(X, X, diag=True)
        tolerance = _tolerance(diagonal)
        if not torch.isfinite(diagonal).all() or (diagonal < 0).any():
            raise RuntimeError('GP kernel diagonal is non-finite or negative.')
        if ((full.diagonal(dim1=-2,dim2=-1)-diagonal).abs() > tolerance).any():
            raise RuntimeError('GP kernel full matrix and diagonal disagree.')
        base = getattr(gp.covar_module, 'base_kernel', gp.covar_module)
        if getattr(base, 'normalize', False):
            base_full = base(X, X).to_dense()
            base_diagonal = base(X, X, diag=True)
            if ((base_diagonal-1.).abs() > _tolerance(torch.ones_like(base_diagonal))).any():
                raise RuntimeError('Normalized kernel does not have a unit diagonal.')
            if ((base_full.diagonal(dim1=-2,dim2=-1)-1.).abs()
                    > _tolerance(torch.ones_like(base_diagonal))).any():
                raise RuntimeError('Normalized full kernel does not have a unit diagonal.')
    return {'points': X.shape[-2], 'minimum_eigenvalue': float(minimum_eigenvalue.min()),
            'maximum_diagonal_error': float((full.diagonal(dim1=-2,dim2=-1)-diagonal).abs().max())}
