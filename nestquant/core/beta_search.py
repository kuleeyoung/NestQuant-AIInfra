"""Appendix F dynamic programming for First-beta codebook selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .codec import decode_nested, encode_nested
from .lattice import E8_DIMENSION


@dataclass(frozen=True)
class BetaSearchResult:
    """Selected beta subset and First-beta assignment for each sample."""

    beta_indices: tuple[int, ...]
    beta_values: tuple[float, ...]
    assignments: torch.Tensor
    total_mse: float


def build_candidate_tables(
    vectors: torch.Tensor,
    q: int,
    betas: Sequence[float],
    *,
    dither: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure reconstruction MSE and overload for every vector/beta pair.

    ``vectors`` must already be in the scale at which the beta codebooks are
    applied (for Algorithm 3 this means after row normalization). The returned
    tables have shape ``[num_vectors, num_betas]``.
    """
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a torch.Tensor")
    if not vectors.is_floating_point():
        raise TypeError("vectors must have a floating-point dtype")
    if vectors.ndim != 2 or vectors.shape[1] != E8_DIMENSION or vectors.shape[0] == 0:
        raise ValueError("vectors must have shape [num_vectors, 8] with at least one row")
    if not torch.isfinite(vectors).all().item():
        raise ValueError("vectors must contain only finite values")
    beta_values = _validate_betas(betas)
    if dither is not None:
        if dither.shape != vectors.shape or dither.device != vectors.device or dither.dtype != vectors.dtype:
            raise ValueError("dither must match vectors shape/device/dtype")
        if not torch.isfinite(dither).all().item():
            raise ValueError("dither must contain only finite values")

    mse_columns = []
    overload_columns = []
    for beta in beta_values:
        codes, overloaded = encode_nested(vectors / beta, q, dither=dither)
        reconstruction = decode_nested(codes, q, dtype=vectors.dtype, dither=dither) * beta
        mse_columns.append((vectors - reconstruction).square().sum(dim=-1))
        overload_columns.append(overloaded)
    return torch.stack(mse_columns, dim=-1), torch.stack(overload_columns, dim=-1)


def select_beta_subset(
    betas: Sequence[float],
    mse: torch.Tensor,
    overloaded: torch.Tensor,
    k: int,
) -> BetaSearchResult:
    """Select an exact-size beta subset minimizing First-beta total squared error.

    The recurrence follows paper Appendix F, Algorithm 6. Candidate betas must
    be strictly increasing, and overload must be monotone (a larger beta may
    resolve overload, but must not create it). A sentinel at index -1 is
    treated as overloaded for every vector so the smallest selected beta owns
    all samples it can encode.
    """
    beta_values = _validate_betas(betas)
    if any(left >= right for left, right in zip(beta_values, beta_values[1:])):
        raise ValueError("betas must be strictly increasing")
    if not isinstance(mse, torch.Tensor) or not mse.is_floating_point():
        raise TypeError("mse must be a floating-point torch.Tensor")
    if not isinstance(overloaded, torch.Tensor) or overloaded.dtype != torch.bool:
        raise TypeError("overloaded must be a bool torch.Tensor")
    if mse.ndim != 2 or mse.shape != overloaded.shape or mse.shape[1] != len(beta_values):
        raise ValueError("mse and overloaded must have shape [num_vectors, num_betas]")
    if mse.shape[0] == 0:
        raise ValueError("at least one vector is required")
    if overloaded.device != mse.device:
        raise ValueError("mse and overloaded must be on the same device")
    if not torch.isfinite(mse).all().item() or (mse < 0).any().item():
        raise ValueError("mse values must be finite and nonnegative")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1 or k > len(beta_values):
        raise ValueError("k must be between 1 and the number of beta candidates")

    # Overload regions are nested: once a sample fits, increasing beta must
    # keep it representable. This property is needed by the First-beta DP.
    creates_overload = (~overloaded[:, :-1]) & overloaded[:, 1:]
    if creates_overload.any().item():
        raise ValueError("overload must not reappear as beta increases")

    _, num_betas = mse.shape
    device = mse.device
    dp = [[math.inf] * (k + 1) for _ in range(num_betas + 1)]
    previous = [[-1] * (k + 1) for _ in range(num_betas + 1)]
    dp[0][0] = 0.0

    # transition_cost[s, i] is the cost assigned to beta i for vectors that
    # overflowed at previous beta s but fit at beta i. Row zero is the sentinel.
    fit = (~overloaded).to(mse.dtype)
    weighted_fit = fit * mse
    transition_cost = torch.zeros((num_betas + 1, num_betas), dtype=mse.dtype, device=device)
    transition_cost[0] = weighted_fit.sum(dim=0)
    transition_cost[1:] = overloaded.to(mse.dtype).T @ weighted_fit
    # Move this small M-by-M table once. Doing .item() inside the nested DP
    # loops would force thousands of GPU synchronization points.
    transition_values = transition_cost.detach().cpu().tolist()

    # State row i+1 represents beta candidate i; row 0 is the sentinel.
    for i in range(num_betas):
        state_i = i + 1
        max_count = min(k, state_i)
        for count in range(1, max_count + 1):
            best_value = math.inf
            best_previous = -1
            for state_s in range(state_i):
                prior = dp[state_s][count - 1]
                if math.isinf(prior):
                    continue
                value = prior + transition_values[state_s][i]
                if value < best_value:
                    best_value = value
                    best_previous = state_s
            if best_previous >= 0:
                dp[state_i][count] = best_value
                previous[state_i][count] = best_previous

    feasible = torch.nonzero(~overloaded.any(dim=0), as_tuple=False).flatten()
    feasible_indices = feasible.detach().cpu().tolist()
    final_states = [index + 1 for index in feasible_indices]
    final_states = [state for state in final_states if not math.isinf(dp[state][k])]
    if not final_states:
        raise ValueError("no beta subset of the requested size covers every vector")
    best_final_state = min(final_states, key=lambda state: dp[state][k])
    total_mse = float(dp[best_final_state][k])

    selected_reversed = []
    state = int(best_final_state)
    for count in range(k, 0, -1):
        selected_reversed.append(state - 1)
        state = previous[state][count]
    if state != 0:
        raise RuntimeError("dynamic-programming backtrace did not reach the sentinel")
    selected = tuple(reversed(selected_reversed))

    selected_overload = overloaded[:, list(selected)]
    first_fitting = (~selected_overload).to(torch.int64).argmax(dim=1)
    if selected_overload.gather(1, first_fitting.unsqueeze(1)).any().item():
        raise RuntimeError("selected beta subset left a vector overloaded")
    assignments = torch.tensor(selected, device=device, dtype=torch.int64)[first_fitting]

    return BetaSearchResult(
        beta_indices=selected,
        beta_values=tuple(beta_values[index] for index in selected),
        assignments=assignments,
        total_mse=total_mse,
    )


def _validate_betas(betas: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(beta) for beta in betas)
    except (TypeError, ValueError) as error:
        raise TypeError("betas must be a nonempty sequence of numbers") from error
    if not values:
        raise ValueError("betas must be nonempty")
    if any(not math.isfinite(beta) or beta <= 0 for beta in values):
        raise ValueError("every beta must be positive and finite")
    return values
