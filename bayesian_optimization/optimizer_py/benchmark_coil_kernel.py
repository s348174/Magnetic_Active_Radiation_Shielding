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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=400)
    parser.add_argument('--coils',type=int,default=K)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args(argv)
    if args.points < 2 or args.coils < 1 or args.threads < 1:
        parser.error('Use at least two points, one coil and one CPU thread.')
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


if __name__ == '__main__':
    main()
