"""Run several TuRBO regions, then independently validate their finalists."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

from MultiTurbo import bo_multi_turbo
from FinalEvaluation import evaluate_finalists
from input import K, SEED, device, ENERGY_BINS, TURBO_MAX_REGIONS, FINAL_PARTICLES, FINAL_REPEATS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions',type=int,default=None,help='0: automatic; or request 4/5 regions')
    parser.add_argument('--iterations',type=int,default=None,help='Total refinement iterations across regions')
    parser.add_argument('--final-particles',type=int,default=FINAL_PARTICLES,help='Particles per species per validation MC')
    parser.add_argument('--validation-repeats',type=int,default=FINAL_REPEATS)
    parser.add_argument('--output-dir',default=None,help='A new directory for this run')
    parser.add_argument('--plot',action='store_true',help='Show optional interactive plots after saving results')
    args = parser.parse_args(argv)
    if args.final_particles < 2*ENERGY_BINS or args.validation_repeats < 2:
        parser.error(f'Validation requires at least {2*ENERGY_BINS} particles and two replicates.')
    if args.regions is not None and not 0 <= args.regions <= TURBO_MAX_REGIONS:
        parser.error(f'--regions must be between 0 and {TURBO_MAX_REGIONS}.')
    if args.iterations is not None and args.iterations < max(1,args.regions or 0):
        parser.error('--iterations must allow at least one step per requested region.')
    if args.output_dir is None:
        Path('results').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f'turbo_K{K}_seed{SEED}_',dir='results'))
    else:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True,exist_ok=False)

    torch.manual_seed(SEED)
    print(f"Using device: {device}. Results: {directory}")
    start = time.time()
    search = bo_multi_turbo(total_iterations=args.iterations,n_regions=args.regions)
    X,y,variance = search['initial']
    np.savez(directory/'initial_design.npz',X=X.cpu().numpy(),y=y,variance=variance)
    with open(directory/'region_selection.json','w') as file:
        json.dump(dict(search['selection'],seed_indices=search['seed_indices'].tolist(),
                       seed=SEED,initial_evaluations=len(y),
                       total_refinement_budget=sum(r['budget'] for r in search['regions'])),file,indent=2)
    for region in search['regions']:
        np.savez(directory/f"region_{region['region']}.npz",**region)
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
