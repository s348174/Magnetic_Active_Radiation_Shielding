"""Time the exact coil-set kernel and a fixed-scale GP fit on CPU; no simulations."""
import argparse
from time import perf_counter
import warnings

import torch
from torch.utils.benchmark import Timer
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.assorted import InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood

from CoilSetModel import make_coil_kernel
from RobustGPFitting import fit_gpytorch_mll_robust
from input import K


def benchmark_acquisition(points):
    """Compare full candidate searches on the same fitted synthetic physical GP."""
    # Optional: BoUtils imports the compiled simulator, but never calls it here.
    from botorch.acquisition import qLogNoisyExpectedImprovement
    from AcquisitionStarts import optimize_acquisition
    from BoUtils import InputMappedAcquisition, map_coilwise
    from FastCoilAcquisition import CheckedLogExpectedImprovement
    from input import (D, unit_bounds, BO_COIL_RESTARTS, BO_COIL_RAW_SAMPLES,
                       BO_COIL_OPT_MAXITER)

    generator = torch.Generator().manual_seed(67)
    X = .1+.8*torch.rand(points,D,dtype=torch.double,generator=generator)
    features = map_coilwise(X)
    second = 6*min(1,K-1)
    y = -34.+.1*features[:,:1].sin()+.1*features[:,second:second+1].sin()
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',InputDataWarning)
        gp = SingleTaskGP(features,y,train_Yvar=torch.full_like(y,.0003),
                          covar_module=make_coil_kernel(K),outcome_transform=Standardize(m=1))
    fit_gpytorch_mll_robust(ExactMarginalLogLikelihood(gp.likelihood,gp))
    with torch.no_grad():
        ranking = gp.posterior(features).mean.reshape(-1)
    print(f'Acquisition benchmark GP fit and ranking: {perf_counter()-start:.3f} s.')
    cases = [('log-EI',BO_COIL_RESTARTS,BO_COIL_RAW_SAMPLES,BO_COIL_OPT_MAXITER),
             ('previous qLogNEI',60,1024,200)]
    for name,restarts,samples,maxiter in cases:
        print(f'Running {name} candidate search...',flush=True)
        # Searches receive the same RNG seed without affecting the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(67)
            start = perf_counter()
            base = (CheckedLogExpectedImprovement(gp,best_f=ranking.max()) if name == 'log-EI'
                    else qLogNoisyExpectedImprovement(gp,X_baseline=features,cache_root=True))
            acquisition = InputMappedAcquisition(base,map_coilwise)
            setup = perf_counter()-start
            start = perf_counter()
            candidate,value = optimize_acquisition(acq_function=acquisition,bounds=unit_bounds,
                X=X,ranking=ranking,num_restarts=restarts,raw_samples=samples,
                options={'maxiter':maxiter})
            elapsed = perf_counter()-start
        finite = bool(torch.isfinite(candidate).all() and torch.isfinite(value).all())
        if not finite:
            raise RuntimeError(f'{name} returned a non-finite proposal or acquisition value.')
        print(f'{name}: setup={setup:.3f} s; optimization={elapsed:.3f} s; '
              f'total={setup+elapsed:.3f} s; finite proposal={finite}; '
              f'restarts={restarts}, raw samples={samples}, maxiter={maxiter}.')
    print('Single-search timings; different acquisitions and budgets, not a quality comparison. '
          'Global posterior-mean and regret searches are excluded.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=400)
    parser.add_argument('--coils',type=int,default=K)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--acquisition',action='store_true',
                        help='Also time physical BO candidate searches; needs the compiled simulator import path')
    args = parser.parse_args(argv)
    if args.points < 2 or args.coils < 1 or args.threads < 1:
        parser.error('Use at least two points, one coil and one CPU thread.')
    if args.acquisition and args.coils != K:
        parser.error('--acquisition uses the physical mapping for K coils from input.py.')
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(67)
    coils = torch.randn(args.points,args.coils,6,dtype=torch.double,generator=generator)
    coils[..., :3] *= 2.
    coils[..., 3:] = torch.nn.functional.normalize(coils[..., 3:],dim=-1)
    X = coils.reshape(args.points,6*args.coils)
    y = coils[..., :3].sin().sum(dim=(-2,-1)).unsqueeze(-1)/args.coils**.5
    variance = torch.full_like(y,.02)
    covar = make_coil_kernel(args.coils)
    base = covar.base_kernel

    def covariance():
        return base(X,X).to_dense()

    def fit():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',InputDataWarning)
            gp = SingleTaskGP(X,y,train_Yvar=variance,
                              covar_module=make_coil_kernel(args.coils),
                              outcome_transform=Standardize(m=1))
        fit_gpytorch_mll_robust(ExactMarginalLogLikelihood(gp.likelihood,gp))
        return gp

    # Warm up PyTorch on a separate small kernel before measuring the first pass.
    make_coil_kernel(args.coils).base_kernel(X[:2],X[:2]).to_dense()
    start = perf_counter()
    covariance()
    first = perf_counter()-start
    reused = Timer('covariance()',globals={'covariance':covariance},
                   num_threads=args.threads).blocked_autorange(min_run_time=.5).median
    fitted = Timer('fit()',globals={'fit':fit},
                   num_threads=args.threads).blocked_autorange(min_run_time=.5).median
    gp = fit()
    candidates = X[:2].clone().unsqueeze(-2).requires_grad_()

    def posterior_gradient():
        posterior = gp.posterior(candidates)
        return torch.autograd.grad((posterior.mean+posterior.variance).sum(),candidates)[0]

    prediction = Timer('posterior_gradient()',globals={'posterior_gradient':posterior_gradient},
                       num_threads=args.threads).blocked_autorange(min_run_time=.5).median
    print(f'CPU: {args.threads} threads; {args.points} observations; {args.coils} coils; float64.')
    print(f'First geometry covariance: {first*1000:.3f} ms (single timing).')
    print(f'Reused geometry covariance: {reused*1000:.3f} ms (median).')
    print(f'Complete fixed-scale GP fit: {fitted*1000:.3f} ms (median).')
    print(f'Two posterior queries with input gradients: {prediction*1000:.3f} ms (median).')
    print('Synthetic targets only; this is not a complete BO iteration or a physics experiment.')
    if args.acquisition:
        benchmark_acquisition(args.points)


if __name__ == '__main__':
    main()
