"""Compare coil configurations using repeated spatial scrambles and common energies."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import t

import input as config
import simulator
from LogDataSampler import sample_all_particles_stratified
from Objective import incident_energy, spherical_to_cartesian


def design_geometry(x):
    x = np.asarray(x,dtype=float)
    if x.shape != (config.D,) or not np.isfinite(x).all() or np.any((x < 0)|(x > 1)):
        raise ValueError('Saved designs must be normalized and match D in input.py.')
    physical = config.LOWER.cpu().numpy()+x*(config.UPPER-config.LOWER).cpu().numpy()
    r,theta,phi,theta_n,phi_n = physical.reshape(5,config.K)
    return (spherical_to_cartesian(r,theta,phi),
            spherical_to_cartesian(np.ones(config.K),theta_n,phi_n))


def saved_geometry(path):
    table = pd.read_csv(path)
    values = table[['cX','cY','cZ','nX','nY','nZ']].to_numpy(dtype=float)
    if values.shape != (config.K,6) or not np.isfinite(values).all():
        raise ValueError(f'{path} must contain {config.K} finite coil rows (check K in input.py).')
    lengths = np.linalg.norm(values[:,3:],axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Coil normal vectors must be nonzero.')
    return values[:,:3],values[:,3:]/lengths[:,None]


def best_design(path):
    with np.load(path,allow_pickle=False) as data:
        X,y = np.asarray(data['X']),np.asarray(data['y'])
    if X.ndim != 2 or y.shape != (len(X),) or not len(X) or not np.isfinite(y).all():
        raise ValueError(f'Invalid design history: {path}')
    return design_geometry(X[int(np.argmax(y))]) # Stored scores are negative log-responses.


def make_cases(run_dir):
    index = np.arange(config.K)
    z = 1-2*(index+.5)/config.K
    angle = np.pi*(3-np.sqrt(5))*index
    directions = np.column_stack((np.sqrt(1-z*z)*np.cos(angle),
                                  np.sqrt(1-z*z)*np.sin(angle),z))
    near,far = config.X_BOUNDS
    cases = []

    def add(name,geometry,source):
        centers,normals = geometry
        if any(np.allclose(centers,row['centers'],rtol=0,atol=1e-10)
               and np.allclose(normals,row['normals'],rtol=0,atol=1e-10) for row in cases):
            print(f'Skipping {name}: identical to an existing configuration.')
            return
        cases.append({'configuration':name,'current_A':float(config.I),
                      'centers':centers,'normals':normals,'source':source})

    if run_dir is None:
        add('reference',(.5*(near+far)*directions,directions.copy()),'Generated middle-radius radial coils')
        print('No --run-dir supplied: reference is a generated example, not your saved optimum.')
    else:
        run_dir = Path(run_dir).resolve()
        csv_path = run_dir/'best_configuration.csv'
        histories = [run_dir/'bo_history.npz',run_dir/'bo_checkpoint.npz']
        if csv_path.is_file():
            add('reference',saved_geometry(csv_path),str(csv_path))
        else:
            history = next((path for path in histories if path.is_file()),None)
            if history is None:
                raise ValueError('Run directory needs best_configuration.csv, bo_history.npz or bo_checkpoint.npz.')
            add('reference',best_design(history),f'Best observed score in {history}')
        initial = run_dir/'initial_design.npz'
        if initial.is_file():
            add('best_initial',best_design(initial),str(initial))
    add('near_radial',(near*directions,directions.copy()),'Generated near-bound centres with radial normals')
    add('far_aligned',(far*directions,np.tile([0.,0.,1.],(config.K,1))),
        'Generated far-bound centres with parallel normals')
    baseline = dict(cases[0],configuration='zero_current',current_A=0.,source='Reference geometry, zero current')
    return [baseline]+cases


def write_csv(path, rows):
    with open(path,'w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(case, rows, incoming):
    responses = np.array([row['response'] for row in rows])
    mean = float(responses.mean())
    variance = float(responses.var(ddof=1))
    se = np.sqrt(variance/len(rows))
    half_width = t.ppf(.975,len(rows)-1)*se
    log_variance = float(np.log(responses).var(ddof=1))
    old_log_variance = float(np.mean([row['legacy_gp_log_variance'] for row in rows]))
    return {'configuration':case['configuration'],'current_A':case['current_A'],'repeats':len(rows),
            'mean_response':mean,'scramble_sd':float(np.sqrt(variance)),'mean_se':float(se),
            'ci95_lower':float(max(0.,mean-half_width)),'ci95_upper':float(mean+half_width),
            'relative_received_percent':100*mean/incoming,
            'empirical_log_variance_single':log_variance,
            'empirical_log_variance_mean':variance/(len(rows)*mean**2),
            'legacy_gp_log_variance':old_log_variance,
            'legacy_to_empirical_variance_ratio':old_log_variance/log_variance if log_variance > 0 else float('inf')}


def compare(rows):
    comparisons = []
    for reference_name in ('reference','zero_current'):
        reference = next(row for row in rows if row['configuration'] == reference_name)
        for row in rows:
            if row['configuration'] == reference_name:
                continue
            a,b = row['mean_se']**2,reference['mean_se']**2
            variance = a+b # Every configuration uses distinct spatial seeds.
            df = variance**2/(a*a/(row['repeats']-1)+b*b/(reference['repeats']-1)) if variance > 0 else np.inf
            half_width = float(t.ppf(.975,df)*np.sqrt(variance))
            difference = row['mean_response']-reference['mean_response']
            comparisons.append({'configuration':row['configuration'],'reference':reference_name,
                                'mean_difference':difference,'difference_se':float(np.sqrt(variance)),
                                'ci95_lower':difference-half_width,'ci95_upper':difference+half_width,
                                'relative_difference_ppm':1e6*difference/reference['mean_response']})
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,help='Completed run/checkpoint directory; otherwise generate a reference')
    parser.add_argument('--particles',type=int,default=config.N,help='Particles per species; default: N in input.py')
    parser.add_argument('--repeats',type=int,default=6,help='Fresh spatial scrambles per configuration; default: 6')
    parser.add_argument('--energy-seed',type=int,default=config.BO_ENERGY_SEED,help='Common energy-bank seed')
    parser.add_argument('--spatial-seed',type=int,default=config.SEED+2000000,help='First experiment spatial seed')
    parser.add_argument('--output-dir',type=Path,help='A new output directory; default: a new folder under results/')
    args = parser.parse_args(argv)
    if args.repeats < 2 or args.particles < 2*config.ENERGY_BINS:
        parser.error(f'Use at least two repeats and {2*config.ENERGY_BINS} particles per species.')
    if min(args.energy_seed,args.spatial_seed) < 0:
        parser.error('Seeds must be nonnegative.')
    try:
        cases = make_cases(args.run_dir)
    except (ValueError,KeyError,OSError) as error:
        parser.error(str(error))
    samples,strata = sample_all_particles_stratified(
        config.FLUX_DATA,args.particles,config.ENERGY_BINS,
        np.random.default_rng(np.random.SeedSequence([1,args.energy_seed])))
    incoming = incident_energy(samples,strata)
    if args.output_dir is None:
        Path('results').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f'noise_experiment_K{config.K}_',dir='results'))
    else:
        directory = args.output_dir
        directory.mkdir(parents=True,exist_ok=False)
    (directory/'configurations').mkdir()
    settings = {'K':config.K,'coil_radius_m':config.R,'particles_per_species':args.particles,
                'energy_bins':config.ENERGY_BINS,'energy_seed':args.energy_seed,
                'energy_seed_entropy':[1,args.energy_seed],'spatial_seed_start':args.spatial_seed,
                'repeats':args.repeats,'sampled_incident_energy':incoming,
                'configurations':[{key:value for key,value in case.items() if key not in ('centers','normals')}
                                  for case in cases]}
    with open(directory/'settings.json','w') as file:
        json.dump(settings,file,indent=2)
    for case in cases:
        pd.DataFrame(np.column_stack((case['centers'],case['normals'])),
                     columns=['cX','cY','cZ','nX','nY','nZ']).to_csv(
                         directory/'configurations'/f"{case['configuration']}.csv",index=False)
    print(f'{len(cases)} configurations x {args.repeats} scrambles = {len(cases)*args.repeats} simulations.')
    print(f'Common incident energy: {incoming:.12e}. Results: {directory.resolve()}',flush=True)
    summaries = []
    columns = ['configuration','replicate','spatial_seed','energy_seed','particles_per_species',
               'current_A','response','legacy_within_variance','legacy_gp_log_variance',
               'sampled_incident_energy','relative_received_percent']
    with open(directory/'runs.csv','w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        file.flush()
        for case_index,case in enumerate(cases):
            rows = []
            for replicate in range(args.repeats):
                seed = args.spatial_seed+case_index*args.repeats+replicate
                mean,variance = simulator.launch_simulation(
                    seed,args.particles,config.K,case['current_A'],config.R,
                    case['centers'],case['normals'],samples,strata)
                if not np.isfinite([mean,variance]).all() or mean <= 0 or variance < 0:
                    raise FloatingPointError(f'Invalid response for {case["configuration"]}, seed {seed}.')
                row = dict(configuration=case['configuration'],replicate=replicate+1,
                           spatial_seed=seed,energy_seed=args.energy_seed,particles_per_species=args.particles,
                           current_A=case['current_A'],response=mean,legacy_within_variance=variance,
                           legacy_gp_log_variance=variance/(mean**2+1e-6)+1e-6,
                           sampled_incident_energy=incoming,relative_received_percent=100*mean/incoming)
                rows.append(row)
                writer.writerow(row)
                file.flush()
                print(f"{case['configuration']} {replicate+1}/{args.repeats}: response={mean:.12e}, "
                      f"received={100*mean/incoming:.9f}%, spatial seed={seed}",flush=True)
            summary = summarize(case,rows,incoming)
            summaries.append(summary)
            write_csv(directory/'summary.csv',summaries)
            print(f"{case['configuration']}: mean SE={summary['mean_se']/summary['mean_response']*1e6:.6g} ppm; "
                  f"empirical single-run log variance={summary['empirical_log_variance_single']:.6g}; "
                  f"legacy GP log variance={summary['legacy_gp_log_variance']:.6g}",flush=True)
    comparisons = compare(summaries)
    write_csv(directory/'comparisons.csv',comparisons)
    print('Differences relative to the reference (negative means lower received energy):')
    for row in comparisons:
        if row['reference'] == 'reference':
            print(f"  {row['configuration']}: {row['relative_difference_ppm']:.6g} ppm; "
                  f"95% difference interval [{row['ci95_lower']:.6g}, {row['ci95_upper']:.6g}]")
    print(f'Completed. Reports: {directory.resolve()}',flush=True)
    return directory


if __name__ == '__main__':
    main()
