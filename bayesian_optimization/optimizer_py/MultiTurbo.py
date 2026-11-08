"""Independent TuRBO runs, seeded from one shared initial evaluation."""
import math
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel

from BoUtils import sobol_sample, map_coilwise, fit_gp_with_noise_floor, duplicate_safe_candidate
from Kernels_v3 import CoilSetKernel
from TurBo import bo_turbo_matern
from input import K, MAX_ITER, SEED, TURBO_REGIONS, TURBO_MAX_REGIONS, FINALISTS


def choose_region_seeds(quality, correlation, total_iterations, n_regions=0, max_regions=5,
                        min_region_iterations=20):
    """Quality-restricted diversity, with effective rank choosing the automatic count.

    quality is maximized. correlation is the fitted kernel correlation, not the
    posterior covariance at training points. The top quarter (at least 2*Lmax)
    forms a promising pool. Greedy selection avoids highly correlated seeds.
    """
    quality = np.asarray(quality, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    if (len(quality) < 1 or not np.isfinite(quality).all()
            or correlation.shape != (len(quality),len(quality))
            or not np.isfinite(correlation).all() or total_iterations < 1
            or np.any(np.diag(correlation) <= 0) or min_region_iterations < 1
            or max_regions < 1 or n_regions < 0 or n_regions > max_regions):
        raise ValueError("Invalid region-selection inputs or budget.")
    pool_size = min(len(quality), max(2*max_regions, math.ceil(len(quality)/4)))
    order = np.argsort(-quality, kind='stable')[:pool_size]
    C = correlation[np.ix_(order,order)]
    effective_rank = float(np.trace(C)**2 / np.square(C).sum())
    if n_regions:
        if n_regions > total_iterations:
            raise ValueError("The budget must permit at least one iteration per requested region.")
        target = min(n_regions,pool_size)
    else:
        target = min(max_regions,pool_size,max(1,total_iterations//min_region_iterations),
                     max(1,math.ceil(effective_rank)))
    selected = [0] # Best posterior mean in the promising pool.
    limit = 1-1e-9 if n_regions else .95
    while len(selected) < target:
        similarity = C[:,selected].max(axis=1)
        similarity[selected] = np.inf
        index = int(np.argmin(similarity))
        if similarity[index] >= limit:
            break # Do not spend several regions on the same effective geometry.
        selected.append(index)
    return order[selected], {'effective_rank':effective_rank, 'pool_size':pool_size,
                              'regions':len(selected), 'requested_regions':n_regions,
                              'min_iterations_per_region':min_region_iterations}


def select_initial_regions(X, y, total_iterations, n_regions=0, max_regions=5):
    mapped = map_coilwise(X)
    gp = SingleTaskGP(mapped, torch.as_tensor(y,dtype=torch.double,device=X.device).unsqueeze(-1),
                      covar_module=ScaleKernel(CoilSetKernel(n_coils=K)),
                      outcome_transform=Standardize(m=1))
    fit_gp_with_noise_floor(gp)
    with torch.no_grad():
        quality = gp.posterior(mapped).mean.squeeze(-1).cpu().numpy()
        covariance = gp.covar_module(mapped).to_dense()
        diagonal = covariance.diagonal().sqrt()
        correlation = covariance / (diagonal[:,None]*diagonal[None,:])
    return choose_region_seeds(quality, correlation.cpu().numpy(), total_iterations,
                               n_regions, max_regions, max(4,X.shape[-1]))


def bo_multi_turbo(acqf='ts', total_iterations=None, n_regions=None,
                   max_regions=TURBO_MAX_REGIONS, max_finalists=FINALISTS, seed=SEED):
    """Run regions sequentially: each simulation already uses species threads.

    MAX_ITER is a total refinement budget, not a budget multiplied by the region
    count. Initial observations are simulated once and copied into each GP.
    Each region's centre follows only its selected seed and its own new points.
    """
    budget = MAX_ITER if total_iterations is None else int(total_iterations)
    requested = TURBO_REGIONS if n_regions is None else int(n_regions)
    if (budget < 1 or max_finalists < 1 or max_regions < 1
            or requested < 0 or requested > max_regions or requested > budget):
        raise ValueError("Invalid region count, optimization budget or finalist budget.")
    initial = sobol_sample()
    X0,y0,_ = initial
    if len(y0) < 2:
        raise ValueError("At least two finite initial observations are required.")
    indices,selection = select_initial_regions(X0,y0,budget,requested,max_regions)
    print(f"Initial pool={selection['pool_size']}, effective rank={selection['effective_rank']:.2f}; "
          f"starting {len(indices)} independent regions.")
    base,remainder = divmod(budget,len(indices))
    if base < X0.shape[-1]:
        print("Note: this short per-region budget may finish before a full failure window.")
    regions,proposals = [],[]
    for region,index in enumerate(indices):
        steps = base + (region < remainder)
        print(f"Region {region+1}: initial point {index}, budget {steps} iterations.")
        with torch.random.fork_rng():
            torch.manual_seed(int(seed)+region)
            result = bo_turbo_matern(acqf=acqf, initial_data=initial,
                                     start_index=int(index), max_iter=steps)
        X,y,acquisition,means,variances = result
        own_indices = np.r_[int(index),np.arange(len(y0),len(y))]
        winner = int(own_indices[np.argmax(y[own_indices])])
        proposals.append({'x':X[winner].copy(), 'score':float(y[winner]), 'region':region+1})
        regions.append({'region':region+1, 'start_index':int(index), 'budget':steps,
                        'X':X, 'y':y, 'acquisition':acquisition,
                        'posterior_mean':np.asarray(means,dtype=float),
                        'posterior_variance':np.asarray(variances)})
    best_initial = int(np.argmax(y0))
    proposals.append({'x':X0[best_initial].cpu().numpy().copy(),
                      'score':float(y0[best_initial]), 'region':0})
    finalists = []
    for proposal in sorted(proposals,key=lambda item:-item['score']):
        if finalists and duplicate_safe_candidate(np.stack([p['x'] for p in finalists]),proposal['x']):
            continue
        finalists.append(proposal)
        if len(finalists) == max_finalists:
            break
    return {'initial':initial, 'selection':selection, 'seed_indices':indices,
            'regions':regions, 'finalists':finalists}
