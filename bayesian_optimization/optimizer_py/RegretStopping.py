"""Approximate GP simple-regret checks in the original normalized search space."""
from dataclasses import dataclass, field
import math
from numbers import Integral
import warnings

import torch
from botorch.acquisition import UpperConfidenceBound
from AcquisitionOptimizer import optimize_acqf

from BoUtils import InputMappedAcquisition
from input import (
    unit_bounds, device, SEED, BO_REGRET_ENABLED, BO_REGRET_REL_TOL,
    BO_REGRET_CONFIDENCE, BO_REGRET_MIN_EVALS, BO_REGRET_CHECK_EVERY,
    BO_REGRET_PATIENCE, BO_REGRET_SAMPLES, BO_POSTERIOR_BATCH_LIMIT, BO_REGRET_RECORD_EVERY,
)


def lower_bound_incumbent(gp, mapping, X, confidence):
    """Return an evaluated point's index and LCB, in original log-response units."""
    with torch.no_grad():
        lower = []
        for batch in X.split(BO_POSTERIOR_BATCH_LIMIT):
            posterior = gp.posterior(mapping(batch).unsqueeze(-2))
            lower.append((posterior.mean-confidence*posterior.variance.clamp_min(0).sqrt()).reshape(-1))
        lower = torch.cat(lower)
    if not torch.isfinite(lower).all():
        raise RuntimeError('Non-finite posterior lower confidence bounds.')
    index = int(lower.argmax().item())
    return index, float(lower[index].item())


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
        incumbent, lower = lower_bound_incumbent(gp, mapping, X, self.confidence)
        acquisition = InputMappedAcquisition(
            UpperConfidenceBound(gp, beta=self.confidence**2), mapping)
        # Include all evaluated points, so the search is never worse than their UCBs.
        points = torch.cat((self.grid, X.detach()))
        with torch.no_grad():
            values = torch.cat([acquisition(batch.unsqueeze(-2)).reshape(-1)
                                for batch in points.split(BO_POSTERIOR_BATCH_LIMIT)])
        if not torch.isfinite(values).all():
            raise RuntimeError('Non-finite UCB grid values.')
        restarts = min(20, len(points))
        starts = points[values.topk(restarts).indices].unsqueeze(-2)
        candidate, _ = optimize_acqf(
            acq_function=acquisition, bounds=unit_bounds, q=1, num_restarts=restarts,
            batch_initial_conditions=starts, require_convergence=True,
        )
        with torch.no_grad():
            optimized = acquisition(candidate.detach()).reshape(-1)
        if not torch.isfinite(candidate).all() or not torch.isfinite(optimized).all():
            raise RuntimeError('Non-finite UCB optimization result.')
        upper = max(values.max().item(), optimized.max().item())
        return {'max_ucb':upper, 'incumbent_lcb':lower, 'incumbent_index':incumbent,
                'estimated_regret':max(0., upper-lower)}

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
        if not model_ok:
            print('Model validation failed: this regret estimate cannot justify stopping.')
        return bool(stopping_check and self.consecutive >= self.patience)

    def settings(self):
        return {'enabled':self.enabled, 'relative_tolerance':self.relative_tolerance,
                'delta':self.delta, 'confidence':self.confidence,
                'min_evaluations':self.min_evaluations, 'check_every':self.check_every,
                'patience':self.patience, 'grid_size':self.grid_size,'record_every':self.record_every}
