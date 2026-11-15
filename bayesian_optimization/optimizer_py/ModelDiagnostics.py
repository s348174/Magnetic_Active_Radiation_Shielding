"""Small repeat-based check of a GP; this is not an optimality certificate."""
import numpy as np
import torch
from scipy.stats import t


def assess_model(gp, mapping, calibration):
    """Compare latent predictions with independent means of negative log responses.

    Calibration repeats remain outside GP training. Their mean variances, rather
    than the single-simulation noise variance, enter this diagnostic. Bonferroni
    intervals account for checking several means and pairwise contrasts together.
    Passing these checks is necessary for stopping, not a global coverage claim.
    """
    means = np.asarray(calibration['mean'], dtype=float)
    mean_variance = np.asarray(calibration['mean_variance'], dtype=float)
    names = calibration['names']
    count = len(means)
    repeats = int(calibration['repeats'])
    if (count < 2 or repeats < 2 or mean_variance.shape != means.shape
            or len(names) != count or not np.isfinite(means).all()
            or not np.isfinite(mean_variance).all() or np.any(mean_variance < 0)):
        raise ValueError('Model checks require at least two finite repeated configurations.')
    total_checks = count + count*(count-1)//2
    critical = float(t.ppf(1-.05/(2*total_checks), repeats-1))
    result = {'model_ok': False, 'supplied_noise': float(calibration['variance']),
              'critical_value': critical, 'calibration_rows': [], 'contrast_rows': [],
              'reasons': []}
    reference = gp.train_inputs[0]
    X = torch.as_tensor(calibration['X'], dtype=reference.dtype, device=reference.device)
    try:
        with torch.no_grad():
            posterior = gp.posterior(mapping(X))
            predicted = posterior.mean.reshape(-1).detach().cpu().numpy()
            # The .variance accessor applies a floor that could hide a collapsed GP.
            covariance = posterior.distribution.covariance_matrix.detach().cpu().numpy()
            scale = float(gp.covar_module.outputscale.item())
            transform = getattr(gp, 'outcome_transform', None)
            outcome_std = float(transform.stdvs.item()) if transform is not None else 1.
        if (predicted.shape != means.shape or covariance.shape != (count, count)
                or not np.isfinite(predicted).all() or not np.isfinite(covariance).all()
                or not np.isfinite([scale, outcome_std]).all()
                or np.any(np.diag(covariance) < -1e-12)):
            raise ValueError('Non-finite or invalid GP predictions in repeat check.')
        result.update(outputscale=scale, signal_variance=scale*outcome_std**2,
                      predicted_mean_range=float(np.ptp(predicted)))
        for i, name in enumerate(names):
            latent = float(max(0., covariance[i, i]))
            width = critical*np.sqrt(latent+mean_variance[i])
            residual = float(predicted[i]-means[i])
            passed = bool(abs(residual) <= width)
            result['calibration_rows'].append({
                'configuration': name, 'observed_mean': float(means[i]),
                'predicted_mean': float(predicted[i]), 'latent_variance': latent,
                'repeat_mean_variance': float(mean_variance[i]),
                'residual': residual, 'allowed_error': float(width), 'passed': passed})
            if not passed:
                result['reasons'].append(f'Prediction misses repeated mean: {name}.')
        for i in range(count):
            for j in range(i):
                observed = float(means[i]-means[j])
                predicted_difference = float(predicted[i]-predicted[j])
                repeat_variance = mean_variance[i]+mean_variance[j]
                latent = max(0., covariance[i, i]+covariance[j, j]-2*covariance[i, j])
                significant = bool(abs(observed) > critical*np.sqrt(repeat_variance))
                width = critical*np.sqrt(latent+repeat_variance)
                passed = bool(not significant or abs(predicted_difference-observed) <= width)
                result['contrast_rows'].append({
                    'configuration_a': names[i], 'configuration_b': names[j],
                    'observed_difference': observed, 'predicted_difference': predicted_difference,
                    'measured_contrast_significant': significant,
                    'allowed_error': float(width), 'passed': passed})
                if not passed:
                    result['reasons'].append(f'Prediction misses measured contrast: {names[i]} / {names[j]}.')
        result['model_ok'] = not result['reasons']
    except (RuntimeError, ValueError) as error:
        result['reasons'].append(str(error))
    return result
