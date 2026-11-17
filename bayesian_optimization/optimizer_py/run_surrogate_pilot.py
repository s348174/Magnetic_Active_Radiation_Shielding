"""Pilot corrected coil-set BO on saved data, then independently validate two designs."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError

import input as config
import Objective
from BoLoop import bo_coil_set_kernel
from BoUtils import map_coilwise
from FinalEvaluation import evaluate_finalists
from bo_main import save_regret_history
from validate_surrogate import load_run


def baseline_index(run_dir, X):
    """Match the independently selected original winner to its stored design."""
    path = Path(run_dir)/'best_configuration.csv'
    table = np.genfromtxt(path,delimiter=',',names=True)
    fields = ('cX','cY','cZ','nX','nY','nZ')
    if not table.dtype.names or any(name not in table.dtype.names for name in fields):
        raise ValueError('Original best_configuration.csv needs Cartesian centres and normals.')
    values = np.column_stack([np.atleast_1d(table[name]) for name in fields])
    if values.shape != (config.K,6) or not np.isfinite(values).all():
        raise ValueError('Original best_configuration.csv does not match the configured coil count.')
    geometry = map_coilwise(torch.as_tensor(X,dtype=torch.double,device=config.device))
    geometry = geometry.detach().cpu().numpy().reshape(len(X),config.K,6)
    matches = np.isclose(geometry,values,rtol=1e-8,atol=1e-9).all(axis=(1,2))
    if not matches.any():
        raise ValueError('Original best_configuration.csv does not match any saved source design.')
    return int(np.flatnonzero(matches)[0])


def _save_json(path, value):
    def clean(item):
        if isinstance(item,np.ndarray):
            return clean(item.tolist())
        if isinstance(item,(float,np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item,np.integer):
            return int(item)
        if isinstance(item,dict):
            return {key:clean(val) for key,val in item.items()}
        if isinstance(item,(list,tuple)):
            return [clean(val) for val in item]
        return item
    temporary = path.with_name(path.name+'.tmp')
    with open(temporary,'w') as file:
        json.dump(clean(value),file,indent=2,allow_nan=False)
    temporary.replace(path)


def run_pilot(run_dir, output_dir=None, evaluations=10, final_particles=None,
              repeats=3, spatial_seed=None, validation_seed=None):
    """Reuse saved common-bank observations without reducing their particle count."""
    source = Path(run_dir).expanduser().resolve()
    X,y,noise,settings,history_path = load_run(source)
    old_best = baseline_index(source,X)
    if not isinstance(evaluations,int) or evaluations < 1:
        raise ValueError('Pilot evaluations must be a positive integer.')
    if not isinstance(repeats,int) or repeats < 2:
        raise ValueError('Final validation needs at least two independent repeats.')
    final_particles = config.FINAL_PARTICLES if final_particles is None else final_particles
    if not isinstance(final_particles,int) or final_particles < 2*config.ENERGY_BINS:
        raise ValueError('Final particle count must provide at least two particles per energy bin.')
    minimum_seed = config.SEED+len(X)
    source_checkpoint = source/'bo_checkpoint.npz'
    if source_checkpoint.is_file():
        with np.load(source_checkpoint,allow_pickle=False) as stored:
            if 'next_spatial_seed' in stored:
                minimum_seed = max(minimum_seed,int(stored['next_spatial_seed']))
    spatial_seed = max(config.SEED+500_000,minimum_seed) if spatial_seed is None else spatial_seed
    validation_seed = config.SEED+3_000_000 if validation_seed is None else validation_seed
    if (not isinstance(spatial_seed,int) or spatial_seed < minimum_seed
            or spatial_seed+evaluations >= config.SEED+1_000_000):
        raise ValueError('Pilot spatial seeds must follow source evaluations and remain below SEED+1000000.')
    if not isinstance(validation_seed,int) or validation_seed < config.SEED+3_000_000:
        raise ValueError('Validation seeds must start at or above SEED+3000000.')
    with open(source/'noise_calibration.json') as file:
        calibration = json.load(file)
    if output_dir is None:
        Path('results').mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='surrogate_pilot_',dir='results')).resolve()
    else:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True,exist_ok=False)
    initial = (torch.as_tensor(X,dtype=torch.double,device=config.device),y,np.full(len(y),noise))
    np.savez(directory/'initial_design.npz',X=X,y=y,variance=initial[2])
    _save_json(directory/'noise_calibration.json',calibration)
    report = dict(status='running',source_run=str(source),source_history=str(history_path),
                  initial_evaluations=len(y),requested_evaluations=evaluations,
                  baseline_source_index=old_best,source_best_observed_index=int(y.argmax()),
                  source_best_observed_response=float(np.exp(-y.max())),
                  source_baseline_observed_response=float(np.exp(-y[old_best])),
                  physical_settings=settings,optimization_particles_per_species=config.N,
                  optimization_energy_seed=config.BO_ENERGY_SEED,
                  pilot_spatial_seed_start=spatial_seed,
                  final_particles_per_species=final_particles,final_repeats=repeats,
                  final_validation_seed_start=validation_seed,
                  final_validation_sampling='fresh independent energy and spatial seeds')
    _save_json(directory/'pilot_report.json',report)

    def checkpoint(history, checks):
        temporary = directory/'bo_checkpoint.npz.tmp'
        with open(temporary,'wb') as file:
            np.savez(file,**history)
        temporary.replace(directory/'bo_checkpoint.npz')
        save_regret_history(directory,checks)
        if (str(np.asarray(history.get('status',''))) == 'ready'
                and int(np.asarray(history.get('evaluations',-1))) == 0
                and not (directory/'initial_model_state.npz').exists()):
            np.savez(directory/'initial_model_state.npz',**history)

    def model_callback(checks):
        _save_json(directory/'model_diagnostics.json',checks)

    start = time.monotonic()
    previous_seed = Objective.optimization_seed_state()
    Objective.set_optimization_seed(spatial_seed)
    print(f'Pilot uses {len(y)} saved observations, N={config.N}; up to {evaluations} new steps. '
          f'Reports: {directory}',flush=True)
    try:
        devices = [config.device.index] if config.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.SEED)
            result = bo_coil_set_kernel(max_iter=evaluations,initial_data=initial,
                                         calibration=calibration,return_diagnostics=True,
                                         checkpoint=checkpoint,model_callback=model_callback)
            next_spatial_seed = Objective.optimization_seed_state()
    except (ModelFittingError,RuntimeError,ValueError) as error:
        report.update(status='pilot_failed',error=str(error),elapsed_seconds=time.monotonic()-start)
        _save_json(directory/'pilot_report.json',report)
        raise RuntimeError(f'Pilot stopped: {error}. Saved observations: {directory}/bo_checkpoint.npz') from error
    finally:
        Objective.set_optimization_seed(previous_seed)

    final_X,final_y,acquisition,means,variances,diagnostics = result
    final_X,final_y = np.asarray(final_X),np.asarray(final_y)
    np.savez(directory/'bo_history.npz',X=final_X,y=final_y,acquisition=acquisition,
             posterior_mean=means,posterior_variance=variances,next_spatial_seed=next_spatial_seed)
    _save_json(directory/'bo_settings.json',dict(
        method='bo',kernel='coil-set',acquisition=diagnostics.get('acquisition','log-nei'),initial_evaluations=len(y),
        seed=config.SEED,optimization_energy_seed=config.BO_ENERGY_SEED,
        total_refinement_budget=evaluations,
        stopping={key:value for key,value in diagnostics.items()
                  if key not in ('checks','model_checks','noise_calibration')}))
    completed = len(final_y)-len(y)
    if completed < 1:
        report.update(status='no_new_observations',elapsed_seconds=time.monotonic()-start)
        _save_json(directory/'pilot_report.json',report)
        raise RuntimeError(f'Pilot produced no new observations. Reports: {directory}')
    new_best = len(y)+int(np.argmax(final_y[len(y):]))
    report.update(status='validating',completed_evaluations=completed,
                  next_spatial_seed=next_spatial_seed,
                  best_new_index=new_best,best_new_observed_response=float(np.exp(-final_y[new_best])),
                  stopping=diagnostics['settings'],stop_reason=diagnostics['stop_reason'],
                  model_checks=diagnostics.get('model_checks',[]),
                  regret_checks=diagnostics.get('checks',[]),
                  optimization_elapsed_seconds=time.monotonic()-start)
    _save_json(directory/'pilot_report.json',report)
    finalists = [dict(x=X[old_best],region=1),dict(x=final_X[new_best],region=1)]
    validation_start = time.monotonic()
    print('Independently validating original winner (finalist 1) and best NEW design (finalist 2).',flush=True)
    try:
        validation = evaluate_finalists(finalists,n_particles=final_particles,repeats=repeats,
                                         seed=validation_seed,output_dir=directory/'validation')
    except (RuntimeError,ValueError) as error:
        report.update(status='validation_failed',error=str(error),elapsed_seconds=time.monotonic()-start)
        _save_json(directory/'pilot_report.json',report)
        raise RuntimeError(f'Final pilot validation failed: {error}. Reports: {directory}') from error
    serialized = [{key:value.tolist() if isinstance(value,np.ndarray) else value
                   for key,value in row.items()} for row in validation]
    by_finalist = {int(row['finalist']):row for row in validation}
    report.update(status='complete',final_validation=serialized,
                  validation_relative_new_minus_baseline=float(
                      by_finalist[2]['mean_response']/by_finalist[1]['mean_response']-1),
                  validation_elapsed_seconds=time.monotonic()-validation_start,
                  elapsed_seconds=time.monotonic()-start)
    _save_json(directory/'pilot_report.json',report)
    print(f'Pilot complete: {directory}. New/baseline validation change: '
          f'{100*report["validation_relative_new_minus_baseline"]:.6g}%.',flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,help='New directory; default: results/surrogate_pilot_*')
    parser.add_argument('--evaluations',type=int,default=10)
    parser.add_argument('--final-particles',type=int,default=config.FINAL_PARTICLES)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--spatial-seed',type=int,default=config.SEED+500_000)
    parser.add_argument('--validation-seed',type=int,default=config.SEED+3_000_000)
    args = parser.parse_args(argv)
    try:
        return run_pilot(args.run_dir,args.output_dir,args.evaluations,args.final_particles,
                         args.repeats,args.spatial_seed,args.validation_seed)
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        parser.exit(1,f'{error}\n')


if __name__ == '__main__':
    main()
