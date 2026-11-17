"""Combine two consecutive BO regret histories without running a GP or simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_regret import read_checks


def folder_hashes(folder):
    return {str(path.relative_to(folder)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file()}


def load_arrays(path):
    with np.load(path,allow_pickle=False) as stored:
        return {key:stored[key].copy() for key in stored.files}


def merge_histories(first,second,output):
    first,second,output = (Path(path).expanduser().resolve() for path in (first,second,output))
    if (first == second or 'results' in output.parts
            or any(output == source or source in output.parents or output in source.parents
                   for source in (first,second))):
        raise ValueError('Choose distinct sources and a new output directory outside results and the source trees.')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory to avoid overwriting files.')
    before = [folder_hashes(source) for source in (first,second)]
    initial = load_arrays(first/'initial_design.npz')
    checkpoint = load_arrays(first/'bo_checkpoint.npz')
    resumed = load_arrays(second/'initial_design.npz')
    final = load_arrays(second/('bo_history.npz' if (second/'bo_history.npz').exists() else 'bo_checkpoint.npz'))
    for key in ('X','y'):
        if (not np.array_equal(initial[key],checkpoint[key][:len(initial['y'])])
                or not np.array_equal(checkpoint[key],resumed[key])
                or not np.array_equal(resumed[key],final[key][:len(resumed['y'])])):
            raise ValueError(f'The saved {key} arrays do not describe consecutive runs.')
    calibration = [json.loads((source/'noise_calibration.json').read_text()) for source in (first,second)]
    if calibration[0] != calibration[1]:
        raise ValueError('Noise calibrations differ; inspect compatibility before combining histories.')
    offset = len(checkpoint['y'])-len(initial['y'])
    continuation = len(final['y'])-len(resumed['y'])
    total = offset+continuation
    first_iterations = int(checkpoint.get('iteration',offset))
    segments = []
    for source,shift,iteration_shift,limit in ((first,0,0,offset),(second,offset,first_iterations,continuation)):
        checked = read_checks(source) # Validate flags and the UCB--LCB identity.
        # Python's float parser preserves written double values across pandas versions.
        converters = {column:lambda value: float(value) if value else np.nan
                      for column in checked.select_dtypes(include='floating').columns}
        data = pd.read_csv(source/'regret_history.csv',converters=converters)
        if ((data['evaluations'] < 1) | (data['evaluations'] > limit)).any():
            raise ValueError('A saved evaluation lies outside its segment.')
        data['source_run'] = source.name
        data['local_evaluations'] = data['evaluations']
        data['local_iteration'] = data['iteration']
        data['evaluations'] += shift
        data['iteration'] += iteration_shift
        segments.append(data)
    records = pd.concat(segments,ignore_index=True).sort_values('evaluations')
    if records['evaluations'].duplicated().any():
        raise ValueError('The shifted histories contain overlapping evaluation numbers.')
    records['record_available'] = True
    dense = records.set_index('evaluations').reindex(range(1,total+1))
    dense.index.name = 'evaluations'
    dense['record_available'] = dense['record_available'].eq(True)
    dense['missing_reason'] = np.where(dense['record_available'],'','not_recorded')
    if offset in dense.index and not dense.loc[offset,'record_available']:
        dense.loc[offset,'missing_reason'] = 'missing_at_resume_boundary; no saved fitted-model regret'
    # The sources here use one iteration per evaluation. Do not invent iteration
    # numbers for missing records in runs that skipped optimization attempts.
    if (records['iteration'] == records['evaluations']).all():
        dense['iteration'] = dense.index
    output.mkdir(parents=True,exist_ok=False)
    records.to_csv(output/'regret_history.csv',index=False)
    dense.reset_index().to_csv(output/'regret_history_all_iterations.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    even_counts = sorted(set(range(2,total+1,2)) | set(records['evaluations']))
    graph = dense.reindex(even_counts)
    fig,axes = plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(graph.index,graph['estimated_regret'],label='Recorded grid UCB--LCB gap')
    axes[0].plot(graph.index,graph['delta'],'--',label='Configured tolerance')
    invalid_model = records['model_ok'].eq(False)
    axes[0].scatter(records.loc[invalid_model,'evaluations'],records.loc[invalid_model,'estimated_regret'],
                    marker='x',s=16,color='tab:red',label='Model check failed')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Estimated log-regret gap')
    for column,label in (('mean_gap','Mean contribution'),('uncertainty_gap','Uncertainty contribution')):
        if column in graph:
            axes[1].plot(graph.index,graph[column],label=label)
    axes[1].axhline(0,color='grey',linewidth=.7)
    axes[1].set_ylabel('Log-response gap')
    axes[1].set_xlabel('BO evaluations across both runs (initial Sobol design excluded)')
    for axis in axes:
        axis.axvline(offset,color='black',linestyle=':',label=f'Resume at {offset} (no saved estimate)')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle(f'Combined regret history: {total} BO evaluations')
    fig.text(.5,.01,'Saved model-based grid estimates; no interpolation or new regret estimation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    for extension in ('png','pdf'):
        fig.savefig(output/f'regret_combined.{extension}',dpi=180)
    plt.close(fig)
    after = [folder_hashes(source) for source in (first,second)]
    if before != after:
        raise RuntimeError('Source contents changed during recovery; inspect the input directories.')
    missing_even = [int(i) for i in range(2,total+1,2) if not dense.loc[i,'record_available']]
    report = dict(total_evaluations=total,first_evaluations=offset,second_evaluations=continuation,
                  initial_sobol_evaluations=len(initial['y']),saved_regret_records=len(records),
                  missing_even_evaluations=missing_even,originals_unchanged=True,
                  sources=[dict(directory=str(source),sha256=hashes) for source,hashes in zip((first,second),before)],
                  note='Missing estimates are blank/NaN, never interpolated. Incumbent row indices already refer '
                       'to the shared observation prefix and are not shifted. The GP was refitted on resume; '
                       'this is a concatenation of recorded diagnostics, not one uninterrupted model history.')
    (output/'recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'README.md').write_text(
        f'# Combined regret history\n\n{offset} + {continuation} = {total} BO evaluations, excluding '
        f'{len(initial["y"])} initial Sobol evaluations.\n\n'
        f'`regret_history.csv` contains the {len(records)} original saved estimates with global counts and provenance.\n'
        '`regret_history_all_iterations.csv` contains one row per evaluation, with blank values where regret was not recorded.\n\n'
        f'Missing even evaluations: {missing_even}. No interpolation or GP refitting was performed. '
        'The resume boundary is marked in the plot; a model refit there can change the diagnostic abruptly. '
        'The estimates are grid diagnostics, not measured true regret or an optimality certificate.\n\n'
        'SHA-256 hashes confirm both source folders were unchanged. See `recovery_report.json`.\n')
    print(f'Recovered {len(records)} saved estimates over {total} BO evaluations. Missing even evaluations: {missing_even}.')
    print(f'Original folders unchanged. Output: {output}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        return merge_histories(args.first,args.second,args.output_dir)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'Recovery failed: {error}\n')


if __name__ == '__main__':
    main()
