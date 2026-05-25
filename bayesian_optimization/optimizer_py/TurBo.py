# Import libraries for optimization
import math
import torch
from dataclasses import dataclass, field

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf

# GPyTorch imports
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
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

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )

# Re-declare so the field() call is resolved correctly at class-definition time.
TurboState = dataclass(TurboState)  # re-apply decorator after fixing imports


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
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

    state.best_value = max(state.best_value, max(Y_next).item())
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
    best_idx = int(np.argmax(y_train))
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

def bo_turbo_matern(nu: float = 2.5, acqf: str = "ts") -> tuple:
    """
    TuRBO-1 loop using a Matérn-5/2 ARD kernel.

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
    warnings.filterwarnings("ignore", category=InputDataWarning)

    # ------------------------------------------------------------------
    # Step 1: Sobol initialisation (reuses your existing helper)
    # ------------------------------------------------------------------
    X_train, y_train, train_yvar = sobol_sample()
    train_yvar = torch.tensor(train_yvar, dtype=torch.double).unsqueeze(-1).to(device)
    train_yvar = torch.clamp(train_yvar, min=1e-6)

    # Raw input dimension (normalised space)
    D = X_train.shape[-1]

    # ------------------------------------------------------------------
    # Step 2: Initial GP fit
    # ------------------------------------------------------------------
    # Lengthscale constraints recommended in the TuRBO paper
    covar_module = ScaleKernel(
        MaternKernel(
            nu=nu,
            ard_num_dims=MAPPED_D,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

    X_train_model = map_cart(X_train)
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
    # Seed best_value from Sobol observations
    state.best_value = float(y_train.max())

    # Tuning knobs (mirror BoTorch tutorial defaults)
    N_CANDIDATES = min(5000, max(2000, 200 * D))
    NUM_RESTARTS = 40
    RAW_SAMPLES  = 512

    ei_array       = np.empty(0)
    best_mean_hist = []
    variance_hist  = []

    # ------------------------------------------------------------------
    # Step 4: TuRBO loop — runs until restart triggered or MAX_ITER reached
    # ------------------------------------------------------------------
    for iteration in range(MAX_ITER):
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
                map_local=map_cart,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            acq_value = torch.tensor(float("nan"))  # TS has no scalar acq value
        else:
            # Log-EI optimised within the TR bounds
            tr_bounds = _tr_bounds(state, X_train, y_train)
            best_f = torch.tensor(y_train.max(), dtype=torch.double, device=device)
            log_ei = qLogNoisyExpectedImprovement(
                model=InputMappedModel(gp, map_cart),
                best_f=best_f,
            )
            candidate, acq_value = optimize_acqf(
                acq_function=log_ei,
                bounds=tr_bounds,
                q=Q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )

        # Take only the first point from the batch for duplicate checking
        cand = candidate[0:1].detach().clone().to(device)
        if duplicate_safe_candidate(X_train, cand):
            print(f"Iteration {iteration+1}: duplicate candidate — skipping.")
            continue

        # 2. Decode candidate → physical parameters
        candidate_np    = candidate.detach().cpu().numpy()
        candidate_unnorm = denormalize(candidate_np)          # back to physical scale
        centers_next, normals_next = unpack_configuration(candidate_unnorm[0])

        # 3. Evaluate the simulator
        energy_next, var_next = objective_function(rng.integers(1e6), centers_next, normals_next)
        train_yvar = torch.cat(
            (train_yvar, torch.tensor([[var_next]], dtype=torch.double).to(device)), dim=0
        )
        train_yvar = torch.clamp(train_yvar, min=1e-6)

        if np.isnan(energy_next) or np.isinf(energy_next):
            warnings.warn(
                f"Non-finite energy at iteration {iteration+1}. Skipping."
            )
            warnings_counter += 1
            if warnings_counter > MAX_ITER // 3:
                raise ValueError("Too many non-finite energies — aborting.")
            continue

        # 4. Append new observation
        X_train = torch.vstack((X_train, cand))
        y_train = np.append(y_train, float(-energy_next))     # negate for maximisation
        assert X_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch: X={X_train.shape}, y={y_train.shape}"

        # 5. Update TuRBO state with the latest batch outcome
        Y_next_tensor = torch.tensor(
            [-energy_next], dtype=torch.double, device=device
        ).unsqueeze(-1)
        state = update_state(state=state, Y_next=Y_next_tensor)

        # 6. Refit GP on all data collected so far
        X_train_model = map_cart(X_train)
        gp = SingleTaskGP(
            X_train_model.detach().clone().to(device),
            torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device),
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        fit_gp_with_noise_floor(gp)

        # 7. Convergence diagnostics (mirrors bo_matern_kernel exactly)
        acq_scalar = acq_value.item() if not torch.isnan(acq_value) else 0.0
        ei_array = np.append(ei_array, acq_scalar)

        from botorch.acquisition import PosteriorMean  # local import avoids circular issues
        post_mean = PosteriorMean(gp)
        post_mean_mapped = InputMappedAcquisition(post_mean, map_cart)
        _, current_best_mean = optimize_acqf(
            acq_function=post_mean_mapped,
            bounds=unit_bounds,
            q=1,
            num_restarts=60,
            raw_samples=1024,
        )
        best_mean_hist.append(current_best_mean)

        #learned_noise_var = likelihood_noise_scalar(gp)
        top_k   = min(5, X_train.shape[0])
        top_idx = torch.as_tensor(
            np.argsort(y_train)[-top_k:], dtype=torch.long, device=device
        )
        posterior     = gp.posterior(X_train_model[top_idx])
        posterior_var = posterior.variance.mean().item()
        variance_hist.append(posterior_var)

        """if stopping_criterion(best_mean_hist, learned_noise_var, posterior_var):
            print(
                f"Convergence reached at iteration {iteration+1} | "
                f"acq={acq_scalar:.6f}, best_mean={current_best_mean:.4f}, "
                f"var={posterior_var:.6f}, TR_length={state.length:.4f}"
            )
            break"""

        print(
            f"Iter {iteration+1}/{MAX_ITER} | "
            f"Energy={np.exp(energy_next):.4f} | "
            f"TR_length={state.length:.4f} | "
            f"best={state.best_value:.4f}"
        )

    print(
        f"TuRBO loop finished — {iteration+1} iterations, "
        f"{warnings_counter} non-finite energy warnings."
    )
    return X_train.cpu().numpy(), y_train, ei_array, best_mean_hist, variance_hist