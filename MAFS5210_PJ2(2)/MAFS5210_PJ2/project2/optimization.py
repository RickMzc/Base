from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_model import RiskModelFit, align_weights, portfolio_total_risk


@dataclass
class OptimizationResult:
    weights: pd.Series
    success: bool
    message: str
    objective_value: float


def optimize_min_tracking_error(
    fit: RiskModelFit,
    initial_weights: pd.Series,
    cap: float = 0.10,
) -> OptimizationResult:
    initial_weights = align_weights(initial_weights, fit.beta.index)
    tickers = fit.beta.index
    n_assets = len(tickers)
    effective_cap = max(cap, 1.0 / n_assets + 1e-6)

    beta = fit.beta.loc[tickers].to_numpy(dtype=float)
    factor_cov = fit.factor_cov.to_numpy(dtype=float)
    idio = fit.idio_var.loc[tickers].to_numpy(dtype=float)
    proxy_benchmark_beta = fit.proxy_benchmark_beta.reindex(fit.factor_cov.index).to_numpy(dtype=float)
    if fit.proxy_benchmark_weights is None:
        proxy_benchmark_weights = np.zeros(n_assets, dtype=float)
    else:
        proxy_benchmark_weights = (
            fit.proxy_benchmark_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        )

    def objective(w: np.ndarray) -> float:
        exposure = beta.T @ w
        active_exposure = exposure - proxy_benchmark_beta
        factor_var = float(active_exposure.T @ factor_cov @ active_exposure)
        portfolio_idio_var = float(np.sum((w**2) * idio))
        idio_overlap_cov = float(np.sum(w * proxy_benchmark_weights * idio))
        idio_var = portfolio_idio_var + fit.proxy_benchmark_idio_var - 2.0 * idio_overlap_cov
        return factor_var + idio_var

    bounds = [(0.0, effective_cap) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = initial_weights.to_numpy(dtype=float)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        weights = initial_weights
    else:
        weights = pd.Series(result.x, index=tickers, dtype=float)
        weights = align_weights(weights, tickers)

    return OptimizationResult(
        weights=weights,
        success=bool(result.success),
        message=str(result.message),
        objective_value=float(objective(weights.to_numpy(dtype=float))),
    )


def optimize_mvo_with_risk_cap(
    expected_scores: pd.Series,
    fit: RiskModelFit,
    initial_weights: pd.Series,
    target_variance: float,
    cap: float = 0.10,
) -> OptimizationResult:
    expected_scores = expected_scores.reindex(fit.beta.index).fillna(0.0).astype(float)
    if expected_scores.std(ddof=0) > 0:
        expected_scores = (expected_scores - expected_scores.mean()) / expected_scores.std(ddof=0)
    initial_weights = align_weights(initial_weights, fit.beta.index)

    tickers = fit.beta.index
    n_assets = len(tickers)
    effective_cap = max(cap, 1.0 / n_assets + 1e-6)
    covariance = fit.covariance_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    mu = expected_scores.to_numpy(dtype=float)

    def objective(w: np.ndarray) -> float:
        return -float(mu @ w)

    def risk_constraint(w: np.ndarray) -> float:
        variance = float(w.T @ covariance @ w)
        return target_variance - variance

    bounds = [(0.0, effective_cap) for _ in range(n_assets)]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": risk_constraint},
    ]
    x0 = initial_weights.to_numpy(dtype=float)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        weights = initial_weights
        success = False
        message = str(result.message)
    else:
        weights = align_weights(pd.Series(result.x, index=tickers, dtype=float), tickers)
        success = True
        message = str(result.message)

    realized_variance = portfolio_total_risk(weights, fit)["variance_daily"]
    if realized_variance > target_variance * 1.0001:
        weights = initial_weights
        success = False
        message = "Fell back to initial weights because optimized solution violated the risk cap."

    return OptimizationResult(
        weights=weights,
        success=success,
        message=message,
        objective_value=float(expected_scores @ weights),
    )
