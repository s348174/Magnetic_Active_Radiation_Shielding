"""Run multi-region TuRBO or vanilla BO, then independently validate finalists."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

from MultiTurbo import bo_multi_turbo
from BoLoop import bo_matern_kernel, bo_rbf_kernel, bo_rbf_old
from BoUtils import sobol_sample, duplicate_safe_candidate
from FinalEvaluation import evaluate_finalists
from input import K, SEED, device, MAX_ITER, ENERGY_BINS, TURBO_MAX_REGIONS, FINAL_PARTICLES, FINAL_REPEATS, FINALISTS


def save_regret_history(directory, checks):
    columns = ['evaluations','iteration','max_ucb','incumbent_lcb','incumbent_index',
               'estimated_regret','delta','search_ok','below_threshold','consecutive_passes']
    temporary = Path(directory)/'regret_history.csv.tmp'
    with open(temporary,'w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        writer.writerows(checks)
    temporary.replace(Path(directory)/'regret_history.csv')


def run_vanilla(kernel='matern', iterations=None, nu=2.5, checkpoint_dir=None):
    """Adapt the existing single-loop results to the common saving/validation path."""
    budget = MAX_ITER if iterations is None else int(iterations)
    if budget < 1 or FINALISTS < 1:
        raise ValueError('Use a positive iteration and finalist budget.')
    initial = sobol_sample()
    functions = {'matern':bo_matern_kernel, 'rbf':bo_rbf_old, 'rbf-set':bo_rbf_kernel}
    kwargs = {'max_iter':budget, 'initial_data':initial, 'return_diagnostics':True}
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        np.savez(directory/'initial_design.npz',X=initial[0].cpu().numpy(),y=initial[1],variance=initial[2])
        def checkpoint(history, checks):
            temporary = directory/'bo_checkpoint.npz.tmp'
            with open(temporary,'wb') as file:
                np.savez(file,**history)
            temporary.replace(directory/'bo_checkpoint.npz')
            save_regret_history(directory,checks)
        kwargs['checkpoint'] = checkpoint
    if kernel == 'matern':
        kwargs['nu'] = nu
    X,y,acquisition,means,variances,diagnostics = functions[kernel](**kwargs)
    finalists = []
    # The regret estimate refers to this LCB incumbent. Always independently validate it.
    order = np.r_[diagnostics['incumbent_index'],np.argsort(-y,kind='stable')]
    for index in order:
        if finalists and duplicate_safe_candidate(np.stack([p['x'] for p in finalists]),X[index]):
            continue
        finalists.append({'x':X[index].copy(), 'score':float(y[index]), 'region':1})
        if len(finalists) == FINALISTS:
            break
    return {'initial':initial,
            'selection':{'regions':1, 'kernel':kernel,
                         'acquisition':'kg' if kernel == 'matern' else 'log-nei',
                         'nu':nu if kernel == 'matern' else None,
                         'stopping':{key:value for key,value in diagnostics.items() if key != 'checks'}},
            'regret_checks':diagnostics['checks'],
            'seed_indices':np.array([],dtype=int),
            'regions':[{'region':1, 'budget':budget, 'X':X, 'y':y,
                        'acquisition':acquisition, 'posterior_mean':np.asarray(means),
                        'posterior_variance':np.asarray(variances)}],
            'finalists':finalists}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('turbo','bo'),default='turbo',help='Search method; default: turbo')
    parser.add_argument('--kernel',choices=('matern','rbf','rbf-set'),default=None,help='Vanilla BO kernel; default: matern')
    parser.add_argument('--nu',type=float,choices=(.5,1.5,2.5),default=None,help='Matérn smoothness; default: 2.5')
    parser.add_argument('--regions',type=int,default=None,help='0: automatic; or request 4/5 regions')
    parser.add_argument('--iterations',type=int,default=None,help='Total refinement iterations (shared across TuRBO regions)')
    parser.add_argument('--final-particles',type=int,default=FINAL_PARTICLES,help='Particles per species per validation MC')
    parser.add_argument('--validation-repeats',type=int,default=FINAL_REPEATS)
    parser.add_argument('--output-dir',default=None,help='A new directory for this run')
    parser.add_argument('--plot',action=argparse.BooleanOptionalAction,default=True,
                        help='Show interactive plots after saving; use --no-plot for headless runs')
    args = parser.parse_args(argv)
    if args.final_particles < 2*ENERGY_BINS or args.validation_repeats < 2:
        parser.error(f'Validation requires at least {2*ENERGY_BINS} particles and two replicates.')
    if args.method == 'bo' and args.regions is not None:
        parser.error('--regions applies only to --method turbo.')
    if args.method == 'turbo' and (args.kernel is not None or args.nu is not None):
        parser.error('--kernel and --nu apply only to --method bo.')
    if args.nu is not None and args.kernel not in (None,'matern'):
        parser.error('--nu applies only to the matern kernel.')
    if args.iterations is not None and args.iterations < 1:
        parser.error('--iterations must be positive.')
    if args.regions is not None and not 0 <= args.regions <= TURBO_MAX_REGIONS:
        parser.error(f'--regions must be between 0 and {TURBO_MAX_REGIONS}.')
    if args.iterations is not None and args.iterations < max(1,args.regions or 0):
        parser.error('--iterations must allow at least one step per requested region.')
    if args.output_dir is None:
        Path('results').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f'{args.method}_K{K}_seed{SEED}_',dir='results'))
    else:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True,exist_ok=False)

    torch.manual_seed(SEED)
    print(f"Using device: {device}. Results: {directory}")
    start = time.time()
    if args.method == 'turbo':
        search = bo_multi_turbo(total_iterations=args.iterations,n_regions=args.regions)
    else:
        search = run_vanilla(kernel=args.kernel or 'matern',iterations=args.iterations,
                             nu=2.5 if args.nu is None else args.nu,checkpoint_dir=directory)
    X,y,variance = search['initial']
    np.savez(directory/'initial_design.npz',X=X.cpu().numpy(),y=y,variance=variance)
    settings_file = 'region_selection.json' if args.method == 'turbo' else 'bo_settings.json'
    with open(directory/settings_file,'w') as file:
        json.dump(dict(search['selection'],method=args.method,seed_indices=search['seed_indices'].tolist(),
                       seed=SEED,initial_evaluations=len(y),
                       total_refinement_budget=sum(r['budget'] for r in search['regions'])),file,indent=2)
    for region in search['regions']:
        history_file = f"region_{region['region']}.npz" if args.method == 'turbo' else 'bo_history.npz'
        np.savez(directory/history_file,**region)
    if args.method == 'bo':
        save_regret_history(directory,search['regret_checks'])
    # This stage is automatic: finalists are ranked by fresh raw responses at high N.
    validation = evaluate_finalists(search['finalists'],n_particles=args.final_particles,
                                    repeats=args.validation_repeats,output_dir=directory)
    best = validation[0]
    print(f"Best validation mean: {best['mean_response']:.8g}; "
          f"95% replicate interval [{best['ci95_lower']:.8g}, {best['ci95_upper']:.8g}].")
    print(f"Elapsed time: {time.time()-start:.1f}s. Saved reports and best_configuration.csv in {directory}.")
    if args.plot:
        from PlotResults import interactive_plot, plot_convergence_metrics
        for region in search['regions']:
            plot_convergence_metrics(region['acquisition'],region['posterior_mean'],region['posterior_variance'])
        interactive_plot(str(directory/'best_configuration.csv'))
    return search,validation


if __name__ == '__main__':
    main()
