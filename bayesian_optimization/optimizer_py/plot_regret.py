"""Plot saved vanilla-BO regret checks without rerunning the GP or simulator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_checks(directory):
    data = pd.read_csv(directory/'regret_history.csv')
    columns = ['evaluations','estimated_regret','delta','max_ucb','incumbent_lcb',
               'search_ok','below_threshold','consecutive_passes']
    if any(column not in data for column in columns):
        raise ValueError(f'regret_history.csv needs these columns: {", ".join(columns)}')
    if data.empty:
        raise ValueError('No regret checks were recorded. The budget may have ended before the first check.')
    for column in ('search_ok','below_threshold'):
        values = data[column].astype(str).str.lower()
        if not values.isin(['true','false','1','0']).all():
            raise ValueError(f'Invalid boolean values in {column}.')
        data[column] = values.isin(['true','1'])
    for column in set(columns)-{'search_ok','below_threshold'}:
        data[column] = pd.to_numeric(data[column],errors='raise')
    data = data.sort_values('evaluations').reset_index(drop=True)
    counts = data['evaluations'].to_numpy()
    if (not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or len(np.unique(counts)) != len(counts)):
        raise ValueError('Evaluation counts must be distinct nonnegative integers.')
    if not (np.isfinite(data['delta']) & (data['delta'] > 0)).all():
        raise ValueError('Every regret tolerance must be finite and positive.')
    passes = data['consecutive_passes'].to_numpy()
    if not np.isfinite(passes).all() or np.any(passes < 0) or np.any(passes != np.floor(passes)):
        raise ValueError('Consecutive-pass counts must be nonnegative integers.')
    finite = np.isfinite(data[['estimated_regret','max_ucb','incumbent_lcb']]).all(axis=1)
    data['valid'] = data['search_ok'] & finite & (data['estimated_regret'] >= 0)
    good = data['valid']
    gap = (data.loc[good,'max_ucb']-data.loc[good,'incumbent_lcb']).clip(lower=0)
    if not np.allclose(gap,data.loc[good,'estimated_regret'],rtol=1e-5,atol=1e-10):
        raise ValueError('Recorded regret is inconsistent with the UCB minus incumbent LCB.')
    if (data.loc[good,'below_threshold'] !=
            (data.loc[good,'estimated_regret'] < data.loc[good,'delta'])).any():
        raise ValueError('Recorded pass flags are inconsistent with the regret tolerance.')
    return data


def read_settings(directory):
    path = directory/'bo_settings.json'
    if not path.exists():
        return {}
    with open(path) as file:
        return json.load(file)


def convergence_figure(data, settings, name):
    import matplotlib.pyplot as plt

    x = data['evaluations'].to_numpy()
    good = data['valid'].to_numpy()
    bound = np.where(good,data['estimated_regret'],np.nan)
    tolerance = data['delta'].to_numpy()
    figure,axes = plt.subplots(3,1,figsize=(8,9),sharex=True)
    ax = axes[0]
    ax.plot(x,bound,'o-',color='tab:blue',label='Estimated simple-regret bound')
    ax.plot(x,tolerance,'--',color='tab:orange',label='Stopping tolerance')
    if np.any(bound[good] == 0):
        ax.set_yscale('symlog',linthresh=float(tolerance.min()/10))
    else:
        ax.set_yscale('log')
    ax.set_ylabel('Estimated log-regret')
    ax.set_title('Estimated regret versus stopping tolerance')
    if np.any(~good):
        ax.scatter(x[~good],np.full(np.sum(~good),.96),transform=ax.get_xaxis_transform(),
                   marker='x',color='tab:red',label='Failed/invalid check: no bound',zorder=5)
    ax.legend(fontsize=9)

    ax = axes[1]
    if good.any():
        origin = float(data.loc[good,'incumbent_lcb'].iloc[0])
        upper = np.where(good,data['max_ucb']-origin,np.nan)
        lower = np.where(good,data['incumbent_lcb']-origin,np.nan)
        ax.plot(x,upper,'o-',color='tab:blue',label='Maximum UCB (shifted)')
        ax.plot(x,lower,'o-',color='tab:green',label='Best evaluated LCB (shifted)')
        ax.fill_between(x,lower,upper,where=good,color='tab:blue',alpha=.12,label='Estimated regret gap')
        ax.legend(fontsize=9)
    else:
        ax.text(.5,.5,'No successful UCB searches',ha='center',transform=ax.transAxes)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
    ax.set_ylabel('Shifted objective')
    ax.set_title('UCB–LCB gap, relative to the first valid LCB')

    ax = axes[2]
    passes = np.where(good,data['consecutive_passes'],0)
    ax.step(x,passes,where='post',color='tab:blue',label='Recorded consecutive passing checks')
    passing = good & data['below_threshold'].to_numpy()
    ax.scatter(x[passing],passes[passing],color='tab:green',label='Passing check',zorder=4)
    failing = ~passing
    if failing.any():
        ax.scatter(x[failing],passes[failing],color='tab:red',marker='x',label='Failing/invalid check',zorder=4)
    stopping = settings.get('stopping',{})
    patience = stopping.get('settings',{}).get('patience')
    if patience is not None and float(patience) > 0:
        ax.axhline(patience,color='tab:orange',linestyle='--',label=f'Required passes: {patience}')
    ax.set_ylim(bottom=-.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('Consecutive passes')
    ax.set_xlabel('Successful new evaluations (initial design excluded)')
    ax.legend(fontsize=9)
    if stopping.get('stop_reason') == 'estimated_simple_regret':
        endpoint = stopping.get('successful_evaluations')
        if endpoint is not None:
            for ax in axes:
                ax.axvline(endpoint,color='tab:green',linestyle=':',alpha=.7)
            axes[0].plot([],[],':',color='tab:green',label='Recorded stopping point')
            axes[0].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    figure.suptitle(f'Estimated simple-regret convergence\n{name}',fontsize=13)
    figure.text(.5,.01,'Model-based estimates, not measured regret or a global-optimality certificate.',
                ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.035,1,.94),h_pad=1.6)
    return figure


def context_figure(directory, settings, data):
    import matplotlib.pyplot as plt

    path = next((directory/name for name in ('bo_history.npz','bo_checkpoint.npz')
                 if (directory/name).exists()),None)
    if path is None:
        return None
    initial = settings.get('initial_evaluations')
    if initial is None and (directory/'initial_design.npz').exists():
        with np.load(directory/'initial_design.npz',allow_pickle=False) as stored:
            initial = len(stored['y'])
    if initial is None:
        print('Skipping context plot: initial evaluation count is unavailable.')
        return None
    with np.load(path,allow_pickle=False) as stored:
        scores = np.asarray(stored['y'],dtype=float)
        variance = np.asarray(stored['posterior_variance'],dtype=float) if 'posterior_variance' in stored else np.array([])
    initial = int(initial)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= initial <= len(scores):
        raise ValueError('Invalid observation history or initial evaluation count.')
    best = np.maximum.accumulate(scores)
    # Use differences of logs to avoid exponentiating the large absolute response.
    relative_response = 100*np.exp(-(best[initial-1:]-best[initial-1]))
    figure,axes = plt.subplots(2,1,figsize=(8,6),sharex=True)
    axes[0].plot(np.arange(len(relative_response)),relative_response,color='tab:blue')
    axes[0].set_ylabel('Best / initial response (%)')
    axes[0].set_title('Best observed response (not independently validated)')
    axes[0].ticklabel_format(axis='y',style='plain',useOffset=False)
    if variance.size:
        if variance.ndim != 1 or len(variance) != len(scores)-initial:
            raise ValueError('Posterior variance history does not match the successful evaluation count.')
        valid = np.isfinite(variance) & (variance >= 0)
        axes[1].plot(np.arange(1,len(variance)+1),np.where(valid,variance,np.nan),color='tab:purple')
        if valid.any():
            if np.any(variance[valid] == 0):
                axes[1].set_yscale('symlog',linthresh=1e-12)
            else:
                axes[1].set_yscale('log')
        else:
            axes[1].text(.5,.5,'No valid local variance values',ha='center',transform=axes[1].transAxes)
        if np.any(np.isclose(variance[valid],1e-10,rtol=.01,atol=0)):
            print('Note: saved local variances include values near 1e-10; inspect possible variance-floor effects.')
    else:
        axes[1].text(.5,.5,'No posterior variance history',ha='center',transform=axes[1].transAxes)
    axes[1].set_ylabel('Local posterior variance')
    axes[1].set_title('Mean latent variance at the top-five observations')
    axes[1].set_xlabel('Successful new evaluations (initial design excluded)')
    for ax in axes:
        ax.grid(alpha=.25)
        for count in data['evaluations']:
            ax.axvline(count,color='grey',alpha=.15,linewidth=.7)
    figure.suptitle('Optimization context',fontsize=13)
    figure.tight_layout(rect=(0,0,1,.95))
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path,help='Vanilla BO output folder containing regret_history.csv')
    parser.add_argument('--output-dir',type=Path,help='Plot output folder; default: RUN_DIR/regret_plots')
    parser.add_argument('--show',action='store_true',help='Also open figures with the available desktop Matplotlib backend')
    args = parser.parse_args(argv)
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        data = read_checks(args.run_dir)
        settings = read_settings(args.run_dir)
        figures = {'regret_convergence':convergence_figure(data,settings,args.run_dir.name)}
        context = context_figure(args.run_dir,settings,data)
        if context is not None:
            figures['optimization_context'] = context
    except (ValueError,KeyError,OSError) as error:
        parser.error(str(error))
    output = args.output_dir or args.run_dir/'regret_plots'
    output.mkdir(parents=True,exist_ok=True)
    for name,figure in figures.items():
        for extension in ('png','pdf'):
            figure.savefig(output/f'{name}.{extension}',dpi=180,bbox_inches='tight')
    count = int(data['valid'].sum())
    print(f'Plotted {len(data)} recorded checks: {count} valid and {len(data)-count} failed/invalid.')
    if len(data) < 5:
        print('Few checkpoints: the plot cannot resolve behavior between recorded checks.')
    print(f'Plots saved in {output.resolve()}')
    if args.show:
        plt.show()
    plt.close('all')
    return output


if __name__ == '__main__':
    main()
