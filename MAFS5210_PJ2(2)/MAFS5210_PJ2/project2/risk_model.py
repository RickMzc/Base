from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RiskModelFit:
    factor_returns: pd.DataFrame
    factor_cov: pd.DataFrame
    beta: pd.DataFrame
    idio_var: pd.Series
    proxy_benchmark_returns: pd.Series
    proxy_benchmark_beta: pd.Series
    proxy_benchmark_idio_var: float
    proxy_benchmark_weights: pd.Series | None = None

    @property
    def factor_names(self) -> list[str]:
        return list(self.factor_returns.columns)

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        beta_values = self.beta.to_numpy(dtype=float)
        factor_cov_values = self.factor_cov.to_numpy(dtype=float)
        d_matrix = np.diag(self.idio_var.reindex(self.beta.index).to_numpy(dtype=float))
        covariance = beta_values @ factor_cov_values @ beta_values.T + d_matrix
        return pd.DataFrame(covariance, index=self.beta.index, columns=self.beta.index)

    def subset(self, tickers: Iterable[str]) -> "RiskModelFit":
        subset_index = pd.Index(tickers).intersection(self.beta.index)
        return RiskModelFit(
            factor_returns=self.factor_returns.copy(),
            factor_cov=self.factor_cov.copy(),
            beta=self.beta.loc[subset_index].copy(),
            idio_var=self.idio_var.loc[subset_index].copy(),
            proxy_benchmark_returns=self.proxy_benchmark_returns.copy(),
            proxy_benchmark_beta=self.proxy_benchmark_beta.copy(),
            proxy_benchmark_idio_var=self.proxy_benchmark_idio_var,
            proxy_benchmark_weights=(
                None
                if self.proxy_benchmark_weights is None
                else self.proxy_benchmark_weights.copy()
            ),
        )


def orthogonalize_factor_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    market_cols = [column for column in factor_returns.columns if column == "Market"]
    sector_cols = [column for column in factor_returns.columns if column.startswith("Sector_")]
    style_cols = [column for column in factor_returns.columns if column not in market_cols + sector_cols]

    orthogonalized = pd.DataFrame(index=factor_returns.index)

    if market_cols:
        orthogonalized[market_cols[0]] = factor_returns[market_cols[0]].astype(float)

    def residualize_block(target: pd.DataFrame, regressors: pd.DataFrame | None) -> pd.DataFrame:
        if target.empty:
            return target.copy()
        if regressors is None or regressors.empty:
            return target.astype(float).copy()

        common_index = target.index.intersection(regressors.index)
        target_aligned = target.loc[common_index].astype(float)
        reg_aligned = regressors.loc[common_index].astype(float)
        x_values = np.column_stack([np.ones(len(common_index), dtype=float), reg_aligned.to_numpy(dtype=float)])
        y_values = target_aligned.to_numpy(dtype=float)
        beta = np.linalg.pinv(x_values) @ y_values
        resid = y_values - x_values @ beta
        return pd.DataFrame(resid, index=common_index, columns=target.columns, dtype=float)

    market_block = orthogonalized[market_cols] if market_cols else None
    sector_block = factor_returns[sector_cols].astype(float) if sector_cols else pd.DataFrame(index=factor_returns.index)
    sector_orth = residualize_block(sector_block, market_block)
    for column in sector_cols:
        orthogonalized[column] = sector_orth[column]

    if style_cols:
        style_regressors = pd.concat(
            [orthogonalized[market_cols] if market_cols else pd.DataFrame(index=factor_returns.index),
             orthogonalized[sector_cols] if sector_cols else pd.DataFrame(index=factor_returns.index)],
            axis=1,
        )
        style_block = factor_returns[style_cols].astype(float)
        style_orth = residualize_block(style_block, style_regressors)
        for column in style_cols:
            orthogonalized[column] = style_orth[column]

    return orthogonalized[factor_returns.columns]


def fit_time_series_factor_model(
    stock_returns: pd.DataFrame,
    raw_factor_returns: pd.DataFrame,
    proxy_benchmark_returns: pd.Series,
    proxy_benchmark_weights: pd.Series | None = None,
    min_valid_obs_ratio: float = 0.8,
    idio_floor: float = 1e-8,
) -> RiskModelFit:
    aligned_index = (
        stock_returns.index.intersection(raw_factor_returns.index).intersection(proxy_benchmark_returns.index)
    )
    factor_window = raw_factor_returns.loc[aligned_index].dropna()
    proxy_window = proxy_benchmark_returns.loc[factor_window.index].dropna()
    factor_window = factor_window.loc[proxy_window.index]

    stock_window = stock_returns.loc[factor_window.index]
    min_valid_obs = max(1, int(len(stock_window) * min_valid_obs_ratio))
    valid_columns = stock_window.columns[stock_window.notna().sum(axis=0) >= min_valid_obs]
    stock_window = stock_window.loc[:, valid_columns].fillna(0.0)
    if stock_window.empty:
        raise ValueError("No stocks left after applying the minimum valid observation filter.")

    factor_window = orthogonalize_factor_returns(factor_window)
    proxy_window = proxy_window.reindex(factor_window.index).fillna(0.0)
    factor_cov = factor_window.cov().astype(float)

    x = factor_window.to_numpy(dtype=float)
    x_pinv = np.linalg.pinv(x)
    y_matrix = stock_window.to_numpy(dtype=float)
    beta_matrix = x_pinv @ y_matrix
    residual_matrix = y_matrix - x @ beta_matrix
    idio_values = np.maximum(np.var(residual_matrix, axis=0, ddof=1), idio_floor)

    proxy_y = proxy_window.to_numpy(dtype=float)
    proxy_beta = x_pinv @ proxy_y
    proxy_residual = proxy_y - x @ proxy_beta
    proxy_idio_var = max(float(np.var(proxy_residual, ddof=1)), idio_floor)

    beta_df = pd.DataFrame(beta_matrix.T, index=stock_window.columns, columns=factor_window.columns, dtype=float)
    idio_var = pd.Series(idio_values, index=stock_window.columns, dtype=float)

    normalized_proxy_weights = None
    if proxy_benchmark_weights is not None:
        normalized_proxy_weights = proxy_benchmark_weights.reindex(beta_df.index).fillna(0.0).astype(float)
        proxy_weight_sum = float(normalized_proxy_weights.sum())
        if proxy_weight_sum > 0.0:
            normalized_proxy_weights = normalized_proxy_weights / proxy_weight_sum
            proxy_beta = beta_df.T @ normalized_proxy_weights
            proxy_idio_var = max(
                float((normalized_proxy_weights.pow(2) * idio_var).sum()),
                idio_floor,
            )
        else:
            normalized_proxy_weights = None

    return RiskModelFit(
        factor_returns=factor_window,
        factor_cov=factor_cov,
        beta=beta_df,
        idio_var=idio_var,
        proxy_benchmark_returns=proxy_window.rename("proxy_benchmark_return"),
        proxy_benchmark_beta=pd.Series(proxy_beta, index=factor_window.columns, dtype=float),
        proxy_benchmark_idio_var=proxy_idio_var,
        proxy_benchmark_weights=normalized_proxy_weights,
    )


def align_weights(weights: pd.Series, universe: Iterable[str]) -> pd.Series:
    aligned = weights.reindex(pd.Index(universe)).fillna(0.0).astype(float)
    weight_sum = float(aligned.sum())
    if weight_sum == 0.0:
        raise ValueError("Weight vector sums to zero after alignment.")
    return aligned / weight_sum


def portfolio_total_risk(weights: pd.Series, fit: RiskModelFit, annualization: int = 252) -> dict[str, object]:
    weights = align_weights(weights, fit.beta.index)
    covariance = fit.covariance_matrix
    w = weights.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    exposures = fit.beta.T @ weights
    factor_cov_times_u = fit.factor_cov @ exposures
    factor_var_contrib = exposures * factor_cov_times_u
    systematic_var = float(factor_var_contrib.sum())

    idio_var_contrib = (weights.pow(2) * fit.idio_var.reindex(weights.index)).fillna(0.0)
    idio_var = float(idio_var_contrib.sum())
    total_var = max(systematic_var + idio_var, 0.0)
    total_vol = float(np.sqrt(total_var))

    stock_var_contrib = pd.Series(w * (sigma @ w), index=weights.index, dtype=float)
    stock_vol_contrib = stock_var_contrib / total_vol if total_vol > 0.0 else stock_var_contrib * np.nan

    return {
        "weights": weights,
        "variance_daily": total_var,
        "vol_daily": total_vol,
        "vol_annual": total_vol * np.sqrt(annualization),
        "systematic_variance_daily": systematic_var,
        "systematic_vol_annual": np.sqrt(max(systematic_var, 0.0)) * np.sqrt(annualization),
        "idiosyncratic_variance_daily": idio_var,
        "idiosyncratic_vol_annual": np.sqrt(max(idio_var, 0.0)) * np.sqrt(annualization),
        "factor_exposure": exposures,
        "factor_variance_contribution": pd.Series(factor_var_contrib, index=fit.factor_cov.index, dtype=float),
        "idiosyncratic_variance_contribution": idio_var_contrib,
        "stock_variance_contribution": stock_var_contrib,
        "stock_vol_contribution": stock_vol_contrib,
        "covariance": covariance,
    }


def portfolio_tracking_error(weights: pd.Series, fit: RiskModelFit, annualization: int = 252) -> dict[str, object]:
    weights = align_weights(weights, fit.beta.index)
    total = portfolio_total_risk(weights, fit, annualization=annualization)

    exposures = total["factor_exposure"]
    active_exposure = exposures - fit.proxy_benchmark_beta.reindex(exposures.index).fillna(0.0)
    factor_cov_times_active = fit.factor_cov @ active_exposure
    factor_var_contrib = active_exposure * factor_cov_times_active
    factor_var = float(factor_var_contrib.sum())

    idio_var_by_stock = fit.idio_var.reindex(weights.index).fillna(0.0)
    portfolio_idio_var = float((weights.pow(2) * idio_var_by_stock).sum())
    if fit.proxy_benchmark_weights is None:
        proxy_weights = pd.Series(0.0, index=weights.index, dtype=float)
        active_idio_var = portfolio_idio_var + fit.proxy_benchmark_idio_var
    else:
        proxy_weights = fit.proxy_benchmark_weights.reindex(weights.index).fillna(0.0).astype(float)
        idio_overlap_cov = float((weights * proxy_weights * idio_var_by_stock).sum())
        active_idio_var = portfolio_idio_var + fit.proxy_benchmark_idio_var - 2.0 * idio_overlap_cov
    active_idio_var = max(active_idio_var, 0.0)
    proxy_te_var = max(factor_var + active_idio_var, 0.0)
    proxy_te_vol = float(np.sqrt(proxy_te_var))

    beta_values = fit.beta.reindex(weights.index).to_numpy(dtype=float)
    factor_gradient = beta_values @ (fit.factor_cov.to_numpy(dtype=float) @ active_exposure.to_numpy(dtype=float))
    idio_gradient = (weights - proxy_weights).to_numpy(dtype=float) * idio_var_by_stock.to_numpy(dtype=float)
    proxy_active_gradient = factor_gradient + idio_gradient
    proxy_stock_var_contrib = pd.Series(
        weights.to_numpy(dtype=float) * proxy_active_gradient,
        index=weights.index,
        dtype=float,
    )
    active_weights = (weights - proxy_weights).astype(float)
    if proxy_te_vol > 0.0:
        proxy_stock_mcar = pd.Series(proxy_active_gradient / proxy_te_vol, index=weights.index, dtype=float)
    else:
        proxy_stock_mcar = pd.Series(np.nan, index=weights.index, dtype=float)
    proxy_stock_te_contrib = active_weights * proxy_stock_mcar
    mcar_reconciliation_error = float(proxy_stock_te_contrib.sum() - proxy_te_vol)

    return {
        "weights": weights,
        "proxy_weights": proxy_weights,
        "active_weights": active_weights,
        "variance_daily": proxy_te_var,
        "vol_daily": proxy_te_vol,
        "vol_annual": proxy_te_vol * np.sqrt(annualization),
        "factor_active_exposure": active_exposure,
        "factor_variance_contribution": pd.Series(factor_var_contrib, index=fit.factor_cov.index, dtype=float),
        "factor_variance_daily": factor_var,
        "idiosyncratic_variance_daily": active_idio_var,
        "proxy_benchmark_idiosyncratic_variance_daily": fit.proxy_benchmark_idio_var,
        "stock_proxy_variance_contribution": proxy_stock_var_contrib,
        "stock_proxy_mcar": proxy_stock_mcar,
        "stock_proxy_te_contribution": proxy_stock_te_contrib,
        "mcar_reconciliation_error": mcar_reconciliation_error,
    }


def compounded_rolling_returns(returns: pd.Series, horizon: int) -> pd.Series:
    return (1.0 + returns).rolling(horizon).apply(np.prod, raw=True) - 1.0


def historical_risk_statistics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    annualization: int = 252,
) -> dict[str, object]:
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    aligned["active"] = aligned["portfolio"] - aligned["benchmark"]

    port_mean = float(aligned["portfolio"].mean())
    port_std = float(aligned["portfolio"].std(ddof=1))
    active_std = float(aligned["active"].std(ddof=1))

    historical_var_rows = []
    normal_var_rows = []
    for horizon in horizons:
        portfolio_h = compounded_rolling_returns(aligned["portfolio"], horizon).dropna()
        for level in confidence_levels:
            quantile = 1.0 - level
            hist_threshold = float(portfolio_h.quantile(quantile)) if not portfolio_h.empty else np.nan
            z_value = float(norm.ppf(quantile))
            mu_h = port_mean * horizon
            sigma_h = port_std * np.sqrt(horizon)
            normal_threshold = mu_h + sigma_h * z_value
            historical_var_rows.append(
                {
                    "horizon_days": horizon,
                    "confidence": level,
                    "threshold_return": hist_threshold,
                }
            )
            normal_var_rows.append(
                {
                    "horizon_days": horizon,
                    "confidence": level,
                    "threshold_return": normal_threshold,
                }
            )

    return {
        "historical_vol_annual": port_std * np.sqrt(annualization),
        "historical_te_annual": active_std * np.sqrt(annualization),
        "historical_var": pd.DataFrame(historical_var_rows),
        "normal_var": pd.DataFrame(normal_var_rows),
        "return_history": aligned,
    }
