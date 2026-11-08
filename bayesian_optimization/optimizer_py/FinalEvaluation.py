"""Independent high-particle-count validation, outside the BO training loop."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t

from BoUtils import denormalize, unpack_configuration
from Objective import evaluate_configuration, spherical_to_cartesian
from input import K, SEED, FINAL_PARTICLES, FINAL_REPEATS


def save_configuration(filename, x):
    physical = denormalize(np.asarray(x)).detach().cpu().numpy()
    centers,normals = unpack_configuration(physical)
    centers = spherical_to_cartesian(centers[:,0],centers[:,1],centers[:,2])
    normals = spherical_to_cartesian(np.ones(K),normals[:,0],normals[:,1])
    with open(filename,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['cX','cY','cZ','nX','nY','nZ'])
        writer.writerows(np.concatenate((centers,normals),axis=1))


def evaluate_finalists(finalists, n_particles=FINAL_PARTICLES, repeats=FINAL_REPEATS,
                        seed=SEED+1000000, output_dir=None):
    """Rank raw response means from independent MC replicates, not log means.

    Each replicate uses n_particles per species and a new energy/spatial seed.
    Error bars use the variance BETWEEN replicates, including spatial scrambling;
    the within-run stratified variance is retained separately as a diagnostic.
    """
    if repeats < 2 or n_particles < 2 or not finalists:
        raise ValueError("Need finalists, at least two particles and two independent replicates.")
    directory = Path(output_dir) if output_dir is not None else None
    columns = ['finalist','region','seed','particles_per_species','response','within_run_variance']
    if directory is not None:
        directory.mkdir(parents=True,exist_ok=True)
        with open(directory/'validation_runs.csv','w',newline='') as file:
            csv.writer(file).writerow(columns)
    results = []
    for candidate,proposal in enumerate(finalists):
        x = np.asarray(proposal['x'])
        centers,normals = unpack_configuration(denormalize(x).detach().cpu().numpy())
        responses,within,seeds = [],[],[]
        if directory is not None:
            save_configuration(directory/f'finalist_{candidate+1}.csv',x)
        for replicate in range(repeats):
            current_seed = int(seed)+candidate*repeats+replicate
            response,variance = evaluate_configuration(current_seed,centers,normals,
                                                       n_particles=n_particles)
            if not np.isfinite([response,variance]).all() or response <= 0 or variance < 0:
                raise FloatingPointError("Invalid final-validation response.")
            responses.append(response); within.append(variance); seeds.append(current_seed)
            print(f"Finalist {candidate+1}, replicate {replicate+1}/{repeats}, "
                  f"N={n_particles}: response={response:.8g}")
            if directory is not None:
                with open(directory/'validation_runs.csv','a',newline='') as file:
                    csv.writer(file).writerow([candidate+1,proposal['region'],current_seed,
                                               n_particles,response,variance])
        mean = float(np.mean(responses))
        se = float(np.std(responses,ddof=1)/np.sqrt(repeats))
        half_width = float(t.ppf(.975,repeats-1))*se
        results.append({'finalist':candidate+1, 'region':proposal['region'], 'x':x.copy(),
                        'mean_response':mean, 'replicate_se':se,
                        'ci95_lower':max(0.,mean-half_width), 'ci95_upper':mean+half_width,
                        'mean_within_run_variance':float(np.mean(within)),
                        'particles_per_species':n_particles, 'repeats':repeats,
                        'seeds':';'.join(map(str,seeds))})
    results.sort(key=lambda row:row['mean_response'])
    best = results[0]
    for rank,row in enumerate(results,1):
        row['rank'] = rank
        row['ci_overlaps_best'] = row['ci95_lower'] <= best['ci95_upper'] and row['ci95_upper'] >= best['ci95_lower']
    if directory is not None:
        fields = ['rank','finalist','region','mean_response','replicate_se','ci95_lower','ci95_upper',
                  'ci_overlaps_best','mean_within_run_variance','particles_per_species','repeats','seeds']
        with open(directory/'final_validation.csv','w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
            writer.writeheader(); writer.writerows(results)
        save_configuration(directory/'best_configuration.csv',best['x'])
    return results
