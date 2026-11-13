# Optimizer memory use and numerical recovery

The BO loops keep the same physical simulator, particle counts, kernels,
acquisition objectives and logarithmic response. This change controls how the
numerical optimizers process their work and how failures are handled.

## Why the issue appeared after changing stopping

The old plateau rule could terminate the four-coil run after only 41 new
observations. The regret rule allows longer searches, with larger GP datasets,
and adds periodic UCB optimization. Both increase the numerical workload.
The old acquisition and posterior-mean calls could optimize 30--60 restarts
together and evaluate up to 1024 initial candidates together.

In addition, wrapping the GP in `InputMappedModel` prevented BoTorch from
recognizing it as a GPyTorch model supporting noisy-EI baseline root caching.
Repeated batched baseline covariance and factorization work therefore consumed
unnecessary memory and time. These were resource-handling weaknesses exposed by
the longer runs, not a change to the magnetic-field or particle equations.

L-BFGS-B `ABNORMAL` warnings indicate numerical convergence failures, often in
line search on flat or poorly conditioned objectives. The stopping change does
not necessarily cause such a failure; it adds searches and exposes more of the
existing optimization workload. Without the surrounding log, the previous
generic warning cannot identify which objective produced it.

## Memory controls

The shared `optimizer_py/AcquisitionOptimizer.py` applies these limits to BO,
regret UCB and TuRBO's numerical acquisition/posterior-mean optimizers:

| input.py setting | Default | Meaning |
| --- | --- | --- |
| `BO_OPT_BATCH_LIMIT` | 2 | Simultaneous outer optimizer restarts |
| `BO_OPT_INIT_BATCH_LIMIT` | 16 | Initial acquisition candidates evaluated together |
| `BO_POSTERIOR_BATCH_LIMIT` | 32 | Regret grid/LCB posterior evaluation batch size |

The total requested outer restart and raw-sample counts are retained. For
example, 60 restarts are processed in groups of at most two. This reduces peak
memory at the cost of more sequential work. Recovery uses one restart at a time
and at most eight initial candidates, subject to the configured lower limit.

BoTorch's KG initializer launches an internal value-function optimizer without
forwarding the outer batch limits. Its internal search is therefore capped at
two restarts and sixteen raw samples. The forty outer KG restarts, fantasy count,
and KG objective are retained; the initialization search is smaller.

Noisy EI now receives the actual GP and the already mapped baseline. The
acquisition itself is wrapped with `InputMappedAcquisition`, so the optimizer
still uses normalized physical-design parameters and mapping occurs once.
This enables baseline Cholesky caching. Candidate/acquisition references are
released before refitting; diagnostics retain scalar floats, not gradient graphs.

For a machine with less RAM, the three limits can be reduced to 1, 8 and 16.
Particle counts need not be reduced to apply these controls. Exact GP fitting
still requires growing memory and computation as observations accumulate; the
limits do not guarantee that every dataset fits every machine.

## Handling optimizer warnings

1. Run L-BFGS-B in small batches, with an increased line-search allowance.
2. On optimizer warnings, non-finite/infeasible output, or numerical exceptions,
   retry once using SLSQP in batches of one. BoTorch's additional automatic retry
   is disabled so the total recovery work is bounded.
3. Ordinary BO/diagnostic searches may use the best finite feasible result or
   a small, batched Sobol fallback if numerical convergence still fails. The
   console identifies the acquisition class and recovery action. A finite
   exploratory candidate need not be a converged acquisition maximum.
4. Regret UCB searches are stricter: if neither attempt completes cleanly, the
   check is inconclusive even when a finite candidate exists. It records infinite
   estimated regret, `search_ok=False`, and resets the stopping patience.

Only recognized optimizer warnings are intercepted for recovery. Other warning
categories remain visible. Persistent numerical difficulties can still occur;
they are handled explicitly rather than treated as proof of convergence.

The optimized UCB is still an approximation to a global maximum even when the
local solver succeeds. The limitations in [SIMPLE_REGRET_STOPPING.md](SIMPLE_REGRET_STOPPING.md)
continue to apply.

## Checkpoint files

Vanilla runs through `bo_main.py` now save the initial design before the search.
After each completed successful BO update, they atomically replace:

- `bo_checkpoint.npz`: normalized designs, negative log-responses, their supplied
  variances, acquisition history, posterior-mean history and variance history;
- `regret_history.csv`: the completed regret checks.

If a later update or system kill interrupts the process, the last completed
checkpoint remains readable. An interrupted temporary write may leave a `.tmp`
file; it does not replace the preceding complete checkpoint. Automatic resume
is not implemented. `bo_history.npz` and final-validation reports are still saved
at normal completion. A kernel OOM kill cannot be caught by Python recovery.

## Verification

Regression checks exercise batch limits, SLSQP recovery from injected SciPy
warnings, rejection of warned UCB checks, batched fallback including KG's
augmented candidates, cached noisy-EI values/gradients, checkpoint persistence
after a later failure, and all existing physics/mapping tests.

Reduced-particle real-simulator runs cover periodic RBF, Matérn and pooled RBF,
including regret checks, checkpoints and independent final validation. A separate
synthetic-GP workload uses 600 observations, an unpruned 600-point noisy-EI
baseline, 60 restarts and 1024 raw samples to exercise later-run resource use.
This is a resource test, not a full 500-iteration physical optimization campaign.
In the cloud test environment, cached noisy EI, posterior-mean optimization and
a 1024-point regret UCB check completed with a measured peak process RSS of
601 MiB. Noisy-EI and posterior-mean stress optimizations used a 30-iteration
solver cap. This measurement is environment/workload specific, not a bound for
every machine or configuration.
