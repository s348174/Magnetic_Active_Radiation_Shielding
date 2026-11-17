"""Update exact GP conditioning data between likelihood optimizations."""
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


def condition_on_observations(gp, X, y, variance):
    """Condition in place on ALL mapped observations, keeping fitted parameters.

    Inputs are already mapped; y and variance are in original objective units.
    The last fit's Standardize statistics stay fixed, including its noise units.
    This supports single-output fixed-noise GPs with unchanged model batch shape.
    set_train_data clears prediction caches, so subsequent predictions include
    every observation and rebuild their exact Cholesky factor. No optimizer runs.
    """
    if (gp.num_outputs != 1 or not isinstance(gp.likelihood,FixedNoiseGaussianLikelihood)
            or getattr(gp.likelihood,'second_noise_covar',None) is not None
            or getattr(gp,'input_transform',None) is not None):
        raise ValueError('Conditioning requires a single-output fixed-noise GP without an input transform.')
    transform = getattr(gp,'outcome_transform',None)
    if transform is not None and not isinstance(transform,Standardize):
        raise ValueError('Conditioning supports Standardize or no outcome transform.')
    reference = gp.train_inputs[0]
    X = torch.as_tensor(X,dtype=reference.dtype,device=reference.device).detach().clone()
    y = torch.as_tensor(y,dtype=reference.dtype,device=reference.device).detach().clone()
    variance = torch.as_tensor(variance,dtype=reference.dtype,device=reference.device).detach().clone()
    if y.shape == X.shape[:-1]:
        y = y.unsqueeze(-1)
    if variance.numel() == 1:
        variance = variance.expand_as(y).clone()
    elif variance.shape == X.shape[:-1]:
        variance = variance.unsqueeze(-1)
    if (X.ndim != reference.ndim or X.shape[:-2] != reference.shape[:-2]
            or X.shape[-1] != reference.shape[-1] or X.shape[-2] < reference.shape[-2]
            or y.shape != X.shape[:-1]+(1,) or variance.shape != y.shape
            or not torch.isfinite(X).all() or not torch.isfinite(y).all()
            or not torch.isfinite(variance).all() or (variance <= 0).any()):
        raise ValueError('Conditioning needs all finite observations and positive raw variances with matching shapes.')

    # Prepare everything before replacing the model's existing conditioning data.
    if transform is not None:
        previous_transform_mode = transform.training
        try:
            transform.eval()
            with torch.no_grad():
                y,variance = transform(y,variance,X=X)
        finally:
            transform.train(previous_transform_mode)
    if (not torch.isfinite(y).all() or not torch.isfinite(variance).all()
            or (variance <= 0).any()):
        raise ValueError('The fitted outcome transform produced invalid conditioning data.')
    old_inputs,old_targets = gp.train_inputs,gp.train_targets
    old_noise,old_prediction = gp.likelihood.noise_covar.noise,gp.prediction_strategy
    old_training = gp.training
    try:
        gp.eval()
        gp.set_train_data(inputs=X,targets=y.squeeze(-1),strict=False)
        gp.likelihood.noise = variance.squeeze(-1)
    except (RuntimeError,ValueError) as error:
        gp.train(old_training)
        gp.train_inputs,gp.train_targets = old_inputs,old_targets
        gp.likelihood.noise_covar.noise = old_noise
        gp.prediction_strategy = old_prediction
        raise ModelFittingError('Updating GP conditioning data failed; the previous state was restored.') from error
    return gp
