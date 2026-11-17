# Import libraries for optimization
import math
import torch
from dataclasses import dataclass

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from AcquisitionOptimizer import optimize_acqf

# GPyTorch imports
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
import gpytorch

# Standard libraries
import numpy as np
import warnings
from botorch.models.utils.assorted import InputDataWarning

# Import external functions — same as BoLoop.py
from BoUtils import (
    denormalize,
    normalize,
    unpack_configuration,
    canonicalize_candidate,
    map_fn,
    map_cart,
    map_set,
    map_coilwise,
    InputMappedAcquisition,
    InputMappedModel,
    sobol_sample,
    stopping_criterion,
    duplicate_safe_candidate,
    fit_gp_with_noise_floor,
    likelihood_noise_scalar,
)
from Objective import objective_function, next_optimization_seed
from input import (device, MAX_ITER, unit_bounds, MAPPED_D, Q, K,
                   TURBO_FAILURE_TOLERANCE, TURBO_LENGTH_MIN, TURBO_LENGTH_INIT,
                   TURBO_CANDIDATES)
from Kernels_v3 import CoilSetKernel, CoilMomentKernel
from PosteriorChecks import raw_mean_variance

# TUNING KNOBS FOR TS SAMPLING
import math
N_CANDIDATES = TURBO_CANDIDATES
NUM_RESTARTS = 40
RAW_SAMPLES  = 512

# ---------------------------------------------------------------------------
# TuRBO state for short local searches
# ---------------------------------------------------------------------------

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = TURBO_LENGTH_INIT
    length_min: float = TURBO_LENGTH_MIN
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = TURBO_FAILURE_TOLERANCE
    success_counter: int = 0
    success_tolerance: int = 10            # original paper uses 3; 10 is more conservative
    best_value: float = -float("inf")
    restart_triggered: bool = False
    best_index: int | None = None # Index of this region's own incumbent in X_train

    def __post_init__(self):
        if (min(self.dim,self.batch_size,self.failure_tolerance,self.success_tolerance) < 1
                or not all(math.isfinite(v) for v in (self.length,self.length_min,self.length_max))
                or not 0 < self.length_min <= self.length <= self.length_max):
            raise ValueError('Trust-region dimensions, windows and lengths must be positive and ordered.')

def update_state(state: TurboState, Y_next: torch.Tensor, new_index=None) -> TurboState:
    """Update one accepted q=1 outcome; ten consecutive failures halve length.

    A strict new observed best resets the failure window. A relative threshold
    multiplied by the absolute log-objective would depend on its arbitrary offset.
    """
    values = torch.as_tensor(Y_next).detach().reshape(-1)
    if values.numel() != 1 or not torch.isfinite(values).all():
        raise ValueError('Trust-region updates require one finite observation.')
    value = float(values.item())
    if value > state.best_value:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:   # expand TR
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # shrink TR
        state.length /= 2.0
        state.failure_counter = 0

    if value > state.best_value:
        state.best_value = value
        if new_index is not None:
            state.best_index = new_index
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


# ---------------------------------------------------------------------------
# Trust-region candidate generation
# ---------------------------------------------------------------------------

def _tr_bounds(state: TurboState, X_train: torch.Tensor, y_train: np.ndarray) -> torch.Tensor:
    """
    Build axis-aligned trust-region bounds centred on the current best point.

    The TR is a hypercube of side-length `state.length` clipped to [0, 1]^D,
    matching the normalised unit-cube convention used throughout BoLoop.py.
    """
    best_idx = state.best_index if state.best_index is not None else int(np.argmax(y_train))
    x_centre = X_train[best_idx]                  # shape (D,)
    D = x_centre.shape[0]
    half = state.length / 2.0

    lb = torch.clamp(x_centre - half, 0.0, 1.0)
    ub = torch.clamp(x_centre + half, 0.0, 1.0)

    # Shape expected by optimize_acqf: (2, D)
    tr_bounds = torch.stack([lb, ub], dim=0).to(dtype=torch.double, device=device)
    return tr_bounds


def _generate_batch_ts(
    state: TurboState,
    gp: SingleTaskGP,
    X_train: torch.Tensor,
    y_train: np.ndarray,
    map_local,
    n_candidates: int,
    num_restarts: int,
    raw_samples: int,
) -> torch.Tensor:
    """
    Generate the next batch of Q candidates inside the trust region using
    Thompson Sampling (MaxPosteriorSampling), consistent with the TuRBO paper.

    Candidates are drawn in the *original* normalised space (D-dim) and the
    GP is evaluated via InputMappedModel, exactly as in bo_matern_kernel().
    """
    tr_bounds = _tr_bounds(state, X_train, y_train)

    # Draw a large Sobol grid restricted to the TR
    sobol = torch.quasirandom.SobolEngine(dimension=X_train.shape[-1], scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=torch.double, device=device)
    # Rescale from [0,1]^D to the TR
    X_cand = tr_bounds[0] + (tr_bounds[1] - tr_bounds[0]) * X_cand  # (n_cand, D)

    # Thompson sampling over the candidate set in mapped space
    ts = MaxPosteriorSampling(model=InputMappedModel(gp, map_local), replacement=False)
    with torch.no_grad():
        X_next = ts(X_cand, num_samples=Q)  # (Q, D)

    return X_next


# ---------------------------------------------------------------------------
# Main TuRBO loop — Matérn kernel variant (mirrors bo_matern_kernel style)
# ---------------------------------------------------------------------------

def bo_turbo_matern(nu: float = 2.5, acqf: str = "ts", initial_data=None,
                    start_index=None, max_iter=None, return_diagnostics=False) -> tuple:
    """
    One trust-region run using the active permutation-invariant coil kernel.
    The legacy name and nu argument are retained for existing callers.
    Optional initial_data reuses a shared initial design without reevaluating it;
    start_index chooses this region's incumbent, independently of other regions.

    Parameters
    ----------
    nu    : Matérn smoothness (default 2.5).
    acqf  : Acquisition strategy — 'ts' (Thompson Sampling, default) or
            'ei' (Log-Expected Improvement optimised over the TR).

    Returns
    -------
    X_train  : np.ndarray  — all evaluated points (normalised).
    y_train  : np.ndarray  — all observed (negated) energies.
    ei_array : np.ndarray  — acquisition values per iteration.
    best_mean_hist : list  — best posterior mean per iteration.
    variance_hist  : list  — mean posterior variance per iteration.
    diagnostics : dict, optional — lifecycle and accepted length history when
        return_diagnostics=True. A terminal observation is retained without an
        unnecessary refit; its posterior diagnostic entries are NaN.
    """
    warnings_counter = 0
    iterations = MAX_ITER if max_iter is None else int(max_iter)
    if iterations < 1 or Q != 1 or acqf not in ('ts','ei'):
        raise ValueError("Use a positive budget, Q=1 and acqf='ts' or 'ei'.")
    if N_CANDIDATES < 1:
        raise ValueError('The Thompson candidate count must be positive.')
    warnings.filterwarnings("ignore", category=InputDataWarning)

    # ------------------------------------------------------------------
    # Step 1: Sobol initialisation (reuses your existing helper)
    # ------------------------------------------------------------------
    X_train, y_train, train_yvar = sobol_sample() if initial_data is None else initial_data
    X_train = X_train.detach().clone()
    y_train = np.asarray(y_train).copy()
    initial_count = len(y_train)
    train_yvar = torch.tensor(train_yvar, dtype=torch.double).unsqueeze(-1).to(device)
    train_yvar = torch.clamp(train_yvar, min=1e-6)

    # Raw input dimension (normalised space)
    D = X_train.shape[-1]

    # ------------------------------------------------------------------
    # Step 2: Initial GP fit
    # ------------------------------------------------------------------
    # Lengthscale constraints recommended in the TuRBO paper
    """covar_module = ScaleKernel(
        MaternKernel(
            nu=nu,
            ard_num_dims=MAPPED_D,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )"""
    """covar_module = RBFKernel(
        ard_num_dims=MAPPED_D,
        lengthscale_constraint=Interval(0.005, 4.0),
    )"""
    covar_module = ScaleKernel(CoilSetKernel(n_coils=K, normalize=True, chunk_size=200))
                    # + ScaleKernel(CoilMomentKernel(n_coils=K)) -> to be added later if Turbo plateaus
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

    X_train_model = map_coilwise(X_train) # Normalized designs -> physical coil features
    assert X_train_model.shape[-1] == K * 6, f"got {X_train_model.shape[-1]}, expected {K*6} -- check mapping call sites"
    gp = SingleTaskGP(
        X_train_model.detach().clone().to(device),
        torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device),
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    fit_gp_with_noise_floor(gp)

    # ------------------------------------------------------------------
    # Step 3: Initialise TuRBO state
    # ------------------------------------------------------------------
    state = TurboState(dim=D, batch_size=Q)
    # A region may use all initial observations to fit its GP, but follows its own incumbent.
    state.best_index = int(np.argmax(y_train)) if start_index is None else int(start_index)
    if state.best_index < 0 or state.best_index >= len(y_train):
        raise ValueError("Invalid initial region index.")
    state.best_value = float(y_train[state.best_index])

    ei_array       = np.empty(0)
    best_mean_hist = []
    variance_hist  = []
    length_history = []
    attempted_iterations = 0
    stop_reason = 'iteration_budget'
    final_model_evaluations = 0

    # ------------------------------------------------------------------
    # Step 4: TuRBO loop — runs until restart triggered or MAX_ITER reached
    # ------------------------------------------------------------------
    for iteration in range(iterations):
        if state.restart_triggered:
            stop_reason = 'length_min'
            print(f"TuRBO region terminated before iteration {iteration+1} "
                  f"(TR length={state.length:.4f} < min={state.length_min:.4f}).")
            break
        attempted_iterations = iteration+1

        # 1. Generate next candidate(s) inside the trust region
        if acqf == "ts":
            # Thompson Sampling — fast, no gradient optimisation required
            candidate = _generate_batch_ts(
                state=state,
                gp=gp,
                X_train=X_train,
                y_train=y_train,
                map_local=map_coilwise,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            acq_value = torch.tensor(float("nan"))  # TS has no scalar acq value
        else:
            # Log-EI optimised within the TR bounds
            tr_bounds = _tr_bounds(state, X_train, y_train)
            log_ei = InputMappedAcquisition(qLogNoisyExpectedImprovement(
                model=gp, X_baseline=map_coilwise(X_train),cache_root=True),map_coilwise)
            candidate, acq_value = optimize_acqf(
                acq_function=log_ei,
                bounds=tr_bounds,
                q=Q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            del log_ei

        # Take only the first point from the batch for duplicate checking
        cand = candidate[0:1].detach().clone().to(device)
        acq_scalar = acq_value.detach().item()
        del candidate, acq_value
        if duplicate_safe_candidate(X_train, cand):
            print(f"Iteration {iteration+1}: duplicate candidate — skipping.")
            continue

        # 2. Decode candidate → physical parameters
        candidate_np    = cand.cpu().numpy()
        candidate_unnorm = denormalize(candidate_np)          # back to physical scale
        centers_next, normals_next = unpack_configuration(candidate_unnorm[0])

        # 3. Evaluate the simulator
        energy_next, var_next = objective_function(next_optimization_seed(), centers_next, normals_next)
        if not np.isfinite([energy_next,var_next]).all() or var_next < 0:
            warnings.warn(
                f"Invalid energy or variance at iteration {iteration+1}. Skipping."
            )
            warnings_counter += 1
            if warnings_counter > iterations // 3:
                raise ValueError("Too many non-finite energies — aborting.")
            continue

        # 4. Append new observation
        X_train = torch.vstack((X_train, cand))
        y_train = np.append(y_train, float(-energy_next))     # negate for maximisation
        train_yvar = torch.cat((train_yvar, torch.tensor([[var_next]], dtype=torch.double, device=device)))
        train_yvar = torch.clamp(train_yvar, min=1e-6)
        assert X_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch: X={X_train.shape}, y={y_train.shape}"

        # 5. Update TuRBO state with the latest batch outcome
        Y_next_tensor = torch.tensor(
            [-energy_next], dtype=torch.double, device=device
        ).unsqueeze(-1)
        state = update_state(state=state, Y_next=Y_next_tensor, new_index=len(y_train)-1)
        length_history.append(float(state.length))
        ei_array = np.append(ei_array, acq_scalar)
        if state.restart_triggered:
            # This observation still belongs in the region's result and finalist
            # ranking. The terminated region does not need another model fit.
            best_mean_hist.append(float('nan'))
            variance_hist.append(float('nan'))
            stop_reason = 'length_min'
            print(f'TuRBO region terminated after {len(length_history)} evaluations '
                  f'(TR length={state.length:.5f} < min={state.length_min:.5f}).')
            break

        # 6. Refit GP on all data collected so far
        X_train_model = map_coilwise(X_train)
        gp = SingleTaskGP(
            X_train_model.detach().clone().to(device),
            torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device),
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        fit_gp_with_noise_floor(gp)
        final_model_evaluations = len(y_train)-initial_count

        # 7. Convergence diagnostics (mirrors bo_matern_kernel exactly)
        # Diagnostics describe this region's incumbent and own observations;
        # a global 60-restart search just for plotting dominated CPU run time.
        own_indices = np.r_[int(start_index) if start_index is not None else
                            int(np.argmax(y_train[:initial_count])),
                            np.arange(initial_count,len(y_train))]
        top_k = min(5,len(own_indices))
        top_idx = torch.as_tensor(
            own_indices[np.argsort(y_train[own_indices])[-top_k:]],
            dtype=torch.long, device=device
        )
        with torch.no_grad():
            incumbent_mean,_ = raw_mean_variance(
                gp,X_train_model[state.best_index:state.best_index+1])
            best_mean_hist.append(float(incumbent_mean.item()))
            _,posterior_variance = raw_mean_variance(gp,X_train_model[top_idx])
            posterior_var = posterior_variance.mean().item()
        variance_hist.append(posterior_var)

        """if stopping_criterion(best_mean_hist, learned_noise_var, posterior_var):
            print(
                f"Convergence reached at iteration {iteration+1} | "
                f"acq={acq_scalar:.6f}, best_mean={current_best_mean:.4f}, "
                f"var={posterior_var:.6f}, TR_length={state.length:.4f}"
            )
            break"""

        print(
            f"Iter {iteration+1}/{iterations} | "
            f"Energy={np.exp(energy_next):.4f} | "
            f"TR_length={state.length:.4f} | "
            f"best={state.best_value:.4f}"
        )

    print(
        f"TuRBO loop finished — {attempted_iterations} attempted iterations, "
        f"{warnings_counter} non-finite energy warnings."
    )
    result = (X_train.cpu().numpy(), y_train, ei_array, best_mean_hist, variance_hist)
    if return_diagnostics:
        diagnostics = dict(stop_reason=stop_reason,budget=iterations,
                           attempted_iterations=attempted_iterations,
                           successful_evaluations=len(y_train)-initial_count,
                           final_model_evaluations=final_model_evaluations,
                           length_history=length_history,final_length=float(state.length),
                           length_min=float(state.length_min),length_init=float(TURBO_LENGTH_INIT),
                           failure_tolerance=state.failure_tolerance,
                           success_tolerance=state.success_tolerance,
                           posterior_mean_scope='region_incumbent',
                           best_index=state.best_index,best_value=state.best_value)
        return (*result,diagnostics)
    return result
