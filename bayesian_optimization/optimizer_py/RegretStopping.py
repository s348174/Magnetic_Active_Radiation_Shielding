"""Approximate GP simple-regret checks in the original normalized search space."""
from dataclasses import dataclass, field
import math
from numbers import Integral
import warnings

import torch
from botorch.acquisition import AcquisitionFunction
from AcquisitionOptimizer import optimize_acqf

from BoUtils import InputMappedAcquisition
from PosteriorChecks import raw_mean_variance
from input import (
    unit_bounds, device, SEED, BO_REGRET_ENABLED, BO_REGRET_REL_TOL,
    BO_REGRET_CONFIDENCE, BO_REGRET_MIN_EVALS, BO_REGRET_CHECK_EVERY,
    BO_REGRET_PATIENCE, BO_REGRET_SAMPLES, BO_POSTERIOR_BATCH_LIMIT, BO_REGRET_RECORD_EVERY,
)


def _sigma(variance):
    # Preserve genuine zero variance without a sqrt(0) derivative or a variance floor.
    positive = variance > 0
    safe = torch.where(positive,variance,torch.ones_like(variance))
    return torch.where(positive,safe.sqrt(),torch.zeros_like(variance))


class RawUpperConfidenceBound(AcquisitionFunction):
    """UCB using validated raw covariance, before GPyTorch's accessor floor."""
    def __init__(self, model, confidence):
        super().__init__(model=model)
        self.confidence = confidence

    def forward(self, X):
        if X.shape[-2] != 1:
            raise ValueError('Regret UCB supports q=1 only.')
        mean, variance = raw_mean_variance(self.model,X)
        return (mean+self.confidence*_sigma(variance)).squeeze(-1).squeeze(-1)


def _observed_moments(gp, mapping, X):
    with torch.no_grad():
        moments = [raw_mean_variance(gp,mapping(batch).unsqueeze(-2))
                   for batch in X.split(BO_POSTERIOR_BATCH_LIMIT)]
        mean = torch.cat([m.reshape(-1) for m,v in moments])
        sigma = torch.cat([_sigma(v).reshape(-1) for m,v in moments])
    return mean,sigma


def lower_bound_incumbent(gp, mapping, X, confidence):
    """Evaluated point's index and LCB using validated raw log-response moments."""
    mean,sigma = _observed_moments(gp,mapping,X)
    lower = mean-confidence*sigma
    index = int(lower.argmax().item())
    return index,float(lower[index].item())


@dataclass
class SimpleRegretStopping:
    enabled: bool = BO_REGRET_ENABLED
    relative_tolerance: float = BO_REGRET_REL_TOL
    confidence: float = BO_REGRET_CONFIDENCE
    min_evaluations: int = BO_REGRET_MIN_EVALS
    check_every: int = BO_REGRET_CHECK_EVERY
    patience: int = BO_REGRET_PATIENCE
    grid_size: int = BO_REGRET_SAMPLES
    record_every: int = BO_REGRET_RECORD_EVERY
    consecutive: int = field(default=0, init=False)
    history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if (not math.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0
                or not math.isfinite(self.confidence) or self.confidence <= 0
                or any(not isinstance(value,Integral) or value < 1 for value in
                       (self.min_evaluations,self.check_every,self.patience,self.grid_size))
                or not isinstance(self.record_every,Integral) or self.record_every < 0):
            raise ValueError('Regret tolerances and budgets must be positive; record_every can also be zero.')
        self.delta = math.log1p(self.relative_tolerance)
        sobol = torch.quasirandom.SobolEngine(unit_bounds.shape[-1], scramble=True, seed=SEED+271828)
        grid = sobol.draw(self.grid_size).to(dtype=torch.double, device=device)
        self.grid = unit_bounds[0] + (unit_bounds[1]-unit_bounds[0]) * grid

    def estimate(self, gp, mapping, X):
        mean,sigma = _observed_moments(gp,mapping,X)
        lower_values = mean-self.confidence*sigma
        incumbent = int(lower_values.argmax().item())
        lower = float(lower_values[incumbent].item())
        acquisition = InputMappedAcquisition(RawUpperConfidenceBound(gp,self.confidence),mapping)
        # Every evaluated point remains eligible, even if the global optimizer misses it.
        points = torch.cat((self.grid,X.detach()))
        with torch.no_grad():
            values = torch.cat([acquisition(batch.unsqueeze(-2)).reshape(-1)
                                for batch in points.split(BO_POSTERIOR_BATCH_LIMIT)])
        if not torch.isfinite(values).all():
            raise RuntimeError('Non-finite UCB grid values.')
        restarts = min(20,len(points))
        starts = points[values.topk(restarts).indices].unsqueeze(-2)
        candidate,_ = optimize_acqf(acq_function=acquisition,bounds=unit_bounds,q=1,
            num_restarts=restarts,batch_initial_conditions=starts,require_convergence=True)
        with torch.no_grad():
            optimized = acquisition(candidate.detach()).reshape(-1)
        if not torch.isfinite(candidate).all() or not torch.isfinite(optimized).all():
            raise RuntimeError('Non-finite UCB optimization result.')
        upper_point = (candidate.detach() if optimized.max() >= values.max()
                       else points[values.argmax()].unsqueeze(-2))
        with torch.no_grad():
            upper_mean,upper_variance = raw_mean_variance(gp,mapping(upper_point))
        upper_mean = float(upper_mean.item())
        upper_sigma = float(_sigma(upper_variance).item())
        incumbent_mean = float(mean[incumbent].item())
        incumbent_sigma = float(sigma[incumbent].item())
        upper = upper_mean+self.confidence*upper_sigma
        mean_gap = upper_mean-incumbent_mean
        uncertainty_gap = self.confidence*(upper_sigma+incumbent_sigma)
        return {'max_ucb':upper,'incumbent_lcb':lower,'incumbent_index':incumbent,
                'estimated_regret':max(0.,mean_gap+uncertainty_gap),
                'upper_mean':upper_mean,'upper_sigma':upper_sigma,
                'incumbent_mean':incumbent_mean,'incumbent_sigma':incumbent_sigma,
                'mean_gap':mean_gap,'uncertainty_gap':uncertainty_gap}

    def check(self, gp, mapping, X, evaluations, iteration, model_ok=True):
        # A failed model check invalidates previous passes, even between checks.
        if not model_ok:
            self.consecutive = 0
        stopping_check = (self.enabled and evaluations >= self.min_evaluations
                          and (evaluations-self.min_evaluations) % self.check_every == 0)
        record_due = self.record_every > 0 and evaluations > 0 and evaluations % self.record_every == 0
        if not (stopping_check or record_due):
            return False
        record = {'evaluations':evaluations, 'iteration':iteration, 'delta':self.delta,
                  'stopping_check':bool(stopping_check), 'model_ok':bool(model_ok)}
        try:
            # Diagnostic work must not consume the acquisition sampler's RNG state.
            devices = [X.device.index] if torch.is_tensor(X) and X.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                result = self.estimate(gp, mapping, X)
            if not math.isfinite(result['estimated_regret']):
                raise RuntimeError('Non-finite estimated regret.')
            record.update(result, search_ok=True)
            below = result['estimated_regret'] < self.delta
        except (RuntimeError, ValueError) as error:
            # A failed maximization must never count as evidence for stopping.
            warnings.warn(f'Regret check failed; continuing optimization: {error}')
            record.update(max_ucb=float('inf'), incumbent_lcb=float('-inf'),
                          incumbent_index=-1, estimated_regret=float('inf'), search_ok=False)
            below = False
        if stopping_check:
            self.consecutive = self.consecutive+1 if below and model_ok else 0
        record.update(below_threshold=below, consecutive_passes=self.consecutive)
        self.history.append(record)
        purpose = 'stopping check' if stopping_check else 'record only'
        print(f"Estimated simple regret ({purpose}): {record['estimated_regret']:.6g}; "
              f"tolerance={self.delta:.6g}; passes={self.consecutive}/{self.patience} "
              f"after {evaluations} new evaluations.")
        if 'mean_gap' in record:
            print(f"Regret components: mean gap={record['mean_gap']:.6g}; "
                  f"uncertainty gap={record['uncertainty_gap']:.6g}.")
        if not model_ok:
            print('Model validation failed: this regret estimate cannot justify stopping.')
        return bool(stopping_check and self.consecutive >= self.patience)

    def settings(self):
        return {'enabled':self.enabled, 'relative_tolerance':self.relative_tolerance,
                'delta':self.delta, 'confidence':self.confidence,
                'min_evaluations':self.min_evaluations, 'check_every':self.check_every,
                'patience':self.patience, 'grid_size':self.grid_size,'record_every':self.record_every}
