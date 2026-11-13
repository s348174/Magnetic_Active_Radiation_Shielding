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
from Objective import objective_function
from input import device, MAX_ITER, unit_bounds, MAPPED_D, rng, Q, K
from Kernels_v3 import CoilSetKernel, CoilMomentKernel

# TUNING KNOBS FOR TS SAMPLING
import math
MAX_MATRIX_BYTES = 2e9  # ~2GB budget for the test-test covariance matrix
N_CANDIDATES = min(
    5000,
    max(200, int(math.sqrt(MAX_MATRIX_BYTES / 8) / K)),
)
NUM_RESTARTS = 40
RAW_SAMPLES  = 512

# ---------------------------------------------------------------------------
# TuRBO state dataclass (faithful to the original BoTorch tutorial)
# ---------------------------------------------------------------------------

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7          # ~0.0078
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = -1  # post-initialised
    success_counter: int = 0
    success_tolerance: int = 10            # original paper uses 3; 10 is more conservative
    best_value: float = -float("inf")
    restart_triggered: bool = False
    best_index: int | None = None # Index of this region's own incumbent in X_train

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )

def update_state(state: TurboState, Y_next: torch.Tensor, new_index=None) -> TurboState:
    """Expand or shrink the trust region based on the latest batch outcome."""
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
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

    if max(Y_next).item() > state.best_value:
        state.best_value = max(Y_next).item()
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
                    start_index=None, max_iter=None) -> tuple:
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
    """
    warnings_counter = 0
    iterations = MAX_ITER if max_iter is None else int(max_iter)
    if iterations < 1 or Q != 1 or acqf not in ('ts','ei'):
        raise ValueError("Use a positive budget, Q=1 and acqf='ts' or 'ei'.")
    warnings.filterwarnings("ignore", category=InputDataWarning)

    # ------------------------------------------------------------------
    # Step 1: Sobol initialisation (reuses your existing helper)
    # ------------------------------------------------------------------
    X_train, y_train, train_yvar = sobol_sample() if initial_data is None else initial_data
    X_train = X_train.detach().clone()
    y_train = np.asarray(y_train).copy()
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

    # ------------------------------------------------------------------
    # Step 4: TuRBO loop — runs until restart triggered or MAX_ITER reached
    # ------------------------------------------------------------------
    for iteration in range(iterations):
        if state.restart_triggered:
            print(f"TuRBO restart triggered at iteration {iteration+1} "
                  f"(TR length={state.length:.4f} < min={state.length_min:.4f}).")
            break

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
        energy_next, var_next = objective_function(rng.integers(1e6), centers_next, normals_next)
        if np.isnan(energy_next) or np.isinf(energy_next):
            warnings.warn(
                f"Non-finite energy at iteration {iteration+1}. Skipping."
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

        # 7. Convergence diagnostics (mirrors bo_matern_kernel exactly)
        ei_array = np.append(ei_array, acq_scalar)

        from botorch.acquisition import PosteriorMean  # local import avoids circular issues
        post_mean = PosteriorMean(gp)
        post_mean_mapped = InputMappedAcquisition(post_mean, map_coilwise)
        _, current_best_mean = optimize_acqf(
            acq_function=post_mean_mapped,
            bounds=unit_bounds,
            q=1,
            num_restarts=60,
            raw_samples=1024,
        )
        best_mean_hist.append(current_best_mean.detach().item())
        del post_mean, post_mean_mapped, current_best_mean

        #learned_noise_var = likelihood_noise_scalar(gp)
        top_k   = min(5, X_train.shape[0])
        top_idx = torch.as_tensor(
            np.argsort(y_train)[-top_k:], dtype=torch.long, device=device
        )
        with torch.no_grad():
            posterior_var = gp.posterior(X_train_model[top_idx]).variance.mean().item()
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
        f"TuRBO loop finished — {iteration+1} iterations, "
        f"{warnings_counter} non-finite energy warnings."
    )
    return X_train.cpu().numpy(), y_train, ei_array, best_mean_hist, variance_hist
