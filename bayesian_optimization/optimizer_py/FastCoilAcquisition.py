"""Checked analytic q=1 improvement for fast CPU coil-set optimization."""
import math

from botorch.acquisition import LogExpectedImprovement

from PosteriorChecks import raw_mean_variance


class CheckedLogExpectedImprovement(LogExpectedImprovement):
    """Analytic EI with a plug-in incumbent and checked latent GP variance.

    Pass the largest posterior mean at evaluated designs as ``best_f``. With
    noisy observations this treats that estimated incumbent value as fixed;
    unlike qLogNEI, it does not integrate its posterior uncertainty. The GP
    itself still uses the calibrated observation noise.

    BoTorch supplies the stable log-EI formula. Its small variance clamp is
    applied ONLY after raw covariance validation, to regularize acquisition
    gradients at zero uncertainty. It never changes the GP, recorded variances,
    or regret checks. Only the vanilla single-output model is supported.
    """

    def _mean_and_sigma(self, X, compute_sigma=True, min_var=1e-12):
        if self.posterior_transform is not None:
            raise ValueError('Checked log-EI supports a single output without a posterior transform.')
        if not math.isfinite(min_var) or min_var <= 0:
            raise ValueError('Acquisition variance regularization must be finite and positive.')
        self.to(X)
        mean, variance = raw_mean_variance(self.model, X)
        mean = mean.squeeze(-2)
        sigma = variance.clamp_min(min_var).sqrt().view(mean.shape) if compute_sigma else None
        return mean, sigma
