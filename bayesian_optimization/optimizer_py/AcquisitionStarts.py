"""Small acquisition restart pools containing known, nearby and global designs."""
import torch
from botorch.acquisition import OneShotAcquisitionFunction

from AcquisitionOptimizer import optimize_acqf
from input import BO_OPT_INIT_BATCH_LIMIT


def _values(acquisition, points):
    """Evaluate q=1 designs without making the cached posterior batch large."""
    with torch.no_grad():
        values = [acquisition(batch.unsqueeze(-2)).reshape(-1)
                  for batch in points.split(BO_OPT_INIT_BATCH_LIMIT)]
    values = torch.cat(values)
    if len(values) != len(points):
        raise ValueError('Restart scoring requires one acquisition value per q=1 design.')
    return values


def observed_local_points(bounds, X, ranking, count):
    """q=1 real-candidate slots for an initializer that separately handles fantasies.

    Keep several top observed points and perturb the same anchors at normalized
    widths .05 and .15. This does not initialize one-shot fantasy coordinates.
    """
    X = torch.as_tensor(X,dtype=bounds.dtype,device=bounds.device).detach()
    ranking = torch.as_tensor(ranking,dtype=bounds.dtype,device=bounds.device).reshape(-1)
    if (count < 1 or X.ndim != 2 or len(X) == 0 or len(ranking) != len(X)
            or X.shape[-1] != bounds.shape[-1] or not torch.isfinite(X).all()
            or not torch.isfinite(ranking).all() or (X < bounds[0]).any()
            or (X > bounds[1]).any()):
        raise ValueError('Observed/local starts need feasible observations and positive count.')
    anchors = X[ranking.topk(min(5,len(X))).indices]
    known_count = min(len(anchors),max(1,count//3))
    known = anchors[:known_count]
    n = count-known_count
    if n == 0:
        return known.unsqueeze(-2).clone()
    index = torch.arange(n,device=X.device)
    widths = torch.where(index % 2 == 0,.05,.15).to(bounds)
    local = anchors[index % len(anchors)]+torch.randn(n,X.shape[-1],dtype=X.dtype,
        device=X.device)*widths[:,None]*(bounds[1]-bounds[0])
    return torch.cat((known,local.clamp(bounds[0],bounds[1]))).unsqueeze(-2).detach()


def optimize_acquisition(*, acq_function, bounds, X, ranking, num_restarts, raw_samples,
                        optimizer=None, **kwargs):
    """Optimize a q=1 acquisition with observed, local and Sobol starts.

    ranking contains larger-is-better scores for local anchors. Restart slots are
    reserved for all three groups so that global samples cannot discard every
    known promising neighbourhood. The returned score is evaluated afresh and
    cannot be worse than the best finite design in the initial pool.
    One-shot KG needs augmented fantasy initial conditions and is not handled here.
    """
    q = kwargs.pop('q', 1)
    if q != 1 or isinstance(acq_function, OneShotAcquisitionFunction):
        raise ValueError('Observed/local restart pools support q=1 non-one-shot acquisitions only.')
    if 'batch_initial_conditions' in kwargs or 'ic_generator' in kwargs:
        raise ValueError('This helper supplies its own q=1 initial conditions.')
    if num_restarts < 1 or raw_samples < 1 or BO_OPT_INIT_BATCH_LIMIT < 1:
        raise ValueError('Restart counts, sample counts and scoring batch limits must be positive.')
    X = torch.as_tensor(X, dtype=bounds.dtype, device=bounds.device).detach()
    ranking = torch.as_tensor(ranking, dtype=bounds.dtype, device=bounds.device).reshape(-1)
    if (bounds.ndim != 2 or bounds.shape[0] != 2 or X.ndim != 2
            or X.shape[-1] != bounds.shape[-1] or len(X) == 0 or len(ranking) != len(X)
            or not torch.isfinite(bounds).all() or not torch.isfinite(X).all()
            or not torch.isfinite(ranking).all() or not (bounds[1] > bounds[0]).all()
            or (X < bounds[0]).any() or (X > bounds[1]).any()):
        raise ValueError('Restart pools need finite feasible observations and matching rankings.')
    width = bounds[1]-bounds[0]
    sobol = torch.quasirandom.SobolEngine(X.shape[-1], scramble=True)
    global_points = bounds[0]+width*sobol.draw(max(raw_samples,num_restarts)).to(bounds)
    anchors = X[ranking.topk(min(5,len(X))).indices]
    local_count = max(raw_samples//2,2*num_restarts)
    anchor_index = torch.arange(local_count,device=X.device) % len(anchors)
    # Two normalized widths cover a close neighbourhood and a broader one.
    scales = torch.where(torch.arange(local_count,device=X.device) % 2 == 0,.05,.15).to(bounds)
    local_points = anchors[anchor_index]+torch.randn(local_count,X.shape[-1],device=X.device,
                                                    dtype=X.dtype)*scales[:,None]*width
    local_points = local_points.clamp(bounds[0],bounds[1])
    points = torch.cat((X,local_points,global_points))
    values = _values(acq_function,points)
    finite = torch.isfinite(values)
    if not finite.any():
        raise RuntimeError('No finite acquisition value in observed, local or Sobol restart pools.')
    masked = values.masked_fill(~finite,-torch.inf)

    def top_indices(start, stop, count):
        eligible = torch.arange(start,stop,device=X.device)[finite[start:stop]]
        count = min(count,len(eligible))
        return eligible[masked[eligible].topk(count).indices]

    # With fewer than three restarts not every group can receive a slot.
    known_count = max(1,num_restarts//3)
    local_slots = max(1,num_restarts//3) if num_restarts >= 3 else 0
    known = top_indices(0,len(X),known_count)
    nearby = top_indices(len(X),len(X)+local_count,local_slots)
    global_start = len(X)+local_count
    distant = top_indices(global_start,len(points),num_restarts-len(known)-len(nearby))
    indices = torch.cat((known,nearby,distant))
    if len(indices) < num_restarts:
        # A group can be nonfinite. Fill remaining slots from other finite designs.
        available = finite.clone()
        available[indices] = False
        extra = torch.arange(len(points),device=X.device)[available]
        take = min(num_restarts-len(indices),len(extra))
        indices = torch.cat((indices,extra[masked[extra].topk(take).indices]))
    starts = points[indices].unsqueeze(-2).detach().clone()
    best_index = int(masked.argmax().item())
    best_candidate = points[best_index:best_index+1].detach().clone()
    best_value = masked[best_index].detach()
    optimizer = optimize_acqf if optimizer is None else optimizer
    candidate, _ = optimizer(acq_function=acq_function,bounds=bounds,q=1,
        num_restarts=len(starts),batch_initial_conditions=starts,**kwargs)
    candidate = candidate.detach()
    if (candidate.shape == (1,X.shape[-1]) and torch.isfinite(candidate).all()
            and (candidate >= bounds[0]).all() and (candidate <= bounds[1]).all()):
        value = _values(acq_function,candidate)[0]
        if torch.isfinite(value) and value >= best_value:
            return candidate,value.detach()
    return best_candidate,best_value
