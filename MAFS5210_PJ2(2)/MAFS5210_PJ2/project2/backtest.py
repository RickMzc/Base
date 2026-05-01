from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .benchmark import BenchmarkWeightsData, load_benchmark_weights
from .data import MarketData, full_weight_series, load_market_data
from .factors import FactorData, compute_factor_data
from .optimization import optimize_min_tracking_error, optimize_mvo_with_risk_cap
from .risk_model import (
    RiskModelFit,
    fit_time_series_factor_model,
    historical_risk_statistics,
    portfolio_total_risk,
    portfolio_tracking_error,
)


ANNUALIZATION = 252
PORTFOLIO_ORDER = ("Original", "Minimum_TE", "MVO")
PORTFOLIO_SLUGS = {
    "Original": "original",
    "Minimum_TE": "minimum_te",
    "MVO": "mvo",
}
HISTORICAL_BENCHMARK_NAME = "S&P500 Index"


@dataclass
class Project2Context:
    market_data: MarketData
    factor_data: FactorData
    benchmark_weights: BenchmarkWeightsData | None = None


@dataclass
class PreparedPeriod:
    rebalance_date: pd.Timestamp
    next_rebalance_date: pd.Timestamp | None
    eligible_names: pd.Index
    selected_names: pd.Index
    benchmark_source: str
    benchmark_holdings_date: pd.Timestamp | None
    benchmark_local_coverage_ratio: float | None
    proxy_benchmark_weights: pd.Series
    fit_full: RiskModelFit
    investable_fit: RiskModelFit
    signal_scores: pd.Series
    original_weights: pd.Series
    trailing_stock_returns: pd.DataFrame
    holding_period_returns: pd.DataFrame
    trailing_historical_benchmark_returns: pd.Series
    holding_historical_benchmark_returns: pd.Series


def prepare_project2_context(
    project1_dir: Path,
    factor_quantile: float = 0.2,
    require_benchmark_weights: bool = True,
    risk_factor_model: str = "barra_proxy",
) -> Project2Context:
    market_data = load_market_data(project1_dir)
    factor_data = compute_factor_data(
        market_data,
        top_bottom_quantile=factor_quantile,
        risk_factor_model=risk_factor_model,
        project2_data_dir=project1_dir.parent / "Project2_data",
    )
    benchmark_weights = load_benchmark_weights(
        project1_dir,
        required=require_benchmark_weights,
    )
    return Project2Context(
        market_data=market_data,
        factor_data=factor_data,
        benchmark_weights=benchmark_weights,
    )


def get_rebalance_pairs(monthly_signal: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    valid_signal = monthly_signal.dropna(how="all")
    dates = list(valid_signal.index)
    return list(zip(dates[:-1], dates[1:]))


def get_rebalance_dates(monthly_signal: pd.DataFrame) -> list[pd.Timestamp]:
    valid_signal = monthly_signal.dropna(how="all")
    return list(valid_signal.index)


def compute_expected_scores(signal_scores: pd.Series) -> pd.Series:
    scores = signal_scores.astype(float).copy()
    score_std = float(scores.std(ddof=0))
    if score_std <= 0.0 or np.isnan(score_std):
        return pd.Series(0.0, index=scores.index, dtype=float)
    return (scores - float(scores.mean())) / score_std


def compute_portfolio_daily_returns(stock_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if stock_returns.empty:
        return pd.Series(dtype=float, name="portfolio_return")
    clean_weights = weights[weights.abs() > 0.0].astype(float)
    if clean_weights.empty:
        return pd.Series(0.0, index=stock_returns.index, name="portfolio_return")
    aligned_returns = stock_returns.reindex(columns=clean_weights.index).fillna(0.0)
    portfolio_returns = aligned_returns.dot(clean_weights)
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def compute_turnover(current_weights: pd.Series, previous_weights: pd.Series) -> float:
    universe = current_weights.index.union(previous_weights.index)
    current = current_weights.reindex(universe).fillna(0.0).astype(float)
    previous = previous_weights.reindex(universe).fillna(0.0).astype(float)
    return float((current - previous).abs().sum() / 2.0)


def _trading_date_on_or_before(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    eligible_dates = index[index <= date]
    if eligible_dates.empty:
        return None
    return pd.Timestamp(eligible_dates.max())


def prepare_period(
    context: Project2Context,
    rebalance_date: pd.Timestamp,
    next_rebalance_date: pd.Timestamp | None,
    top_pct: float = 0.05,
    risk_window: int = 252,
    min_universe: int = 50,
    min_risk_obs: int | None = None,
) -> PreparedPeriod | None:
    market_data = context.market_data
    factor_data = context.factor_data
    if rebalance_date not in factor_data.monthly_signal.index:
        return None

    signal_scores = factor_data.monthly_signal.loc[rebalance_date].dropna().astype(float)
    if signal_scores.empty:
        return None

    raw_factor_returns = factor_data.factor_returns.loc[:rebalance_date].tail(risk_window)
    fit_index = market_data.returns.index.intersection(raw_factor_returns.index)
    raw_factor_returns = raw_factor_returns.loc[fit_index].dropna()
    stock_window = market_data.returns.loc[raw_factor_returns.index]

    required_risk_obs = (
        min_risk_obs if min_risk_obs is not None else max(15, raw_factor_returns.shape[1] * 3)
    )
    if len(raw_factor_returns) < required_risk_obs or stock_window.empty:
        return None

    min_stock_obs = max(1, int(len(stock_window) * 0.8))
    valid_names = stock_window.columns[stock_window.notna().sum(axis=0) >= min_stock_obs]
    selection_candidates = signal_scores.index.intersection(valid_names)

    benchmark_source = "Equal-weight local universe proxy"
    benchmark_holdings_date: pd.Timestamp | None = None
    benchmark_local_coverage_ratio: float | None = None
    risk_model_names = pd.Index(valid_names)
    proxy_benchmark_weights = None
    benchmark_data = context.benchmark_weights
    if benchmark_data is not None:
        if rebalance_date not in benchmark_data.by_rebalance:
            raise KeyError(
                f"Missing cached benchmark weights for {rebalance_date.date().isoformat()}. "
                "Re-run `python download_ivv_benchmark_weights.py`."
            )
        raw_benchmark_weights = benchmark_data.by_rebalance[rebalance_date].reindex(risk_model_names).fillna(0.0)
        raw_benchmark_weights = raw_benchmark_weights[raw_benchmark_weights > 0.0]
        if raw_benchmark_weights.empty:
            raise ValueError(
                f"Cached benchmark weights for {rebalance_date.date().isoformat()} have no overlap "
                "with the local risk universe."
            )
        proxy_benchmark_weights = (raw_benchmark_weights / raw_benchmark_weights.sum()).rename(
            "proxy_benchmark_weight"
        )
        selection_candidates = selection_candidates.intersection(proxy_benchmark_weights.index)
        risk_model_names = proxy_benchmark_weights.index
        benchmark_source = benchmark_data.source_name
        benchmark_holdings_date = benchmark_data.holdings_date_by_rebalance.get(rebalance_date)
        benchmark_local_coverage_ratio = float(
            benchmark_data.coverage_ratio_by_rebalance.get(rebalance_date, np.nan)
        )
        if len(selection_candidates) < min_universe:
            return None

    if proxy_benchmark_weights is None:
        risk_model_names = selection_candidates
        proxy_benchmark_weights = pd.Series(
            1.0 / len(risk_model_names),
            index=risk_model_names,
            dtype=float,
            name="proxy_benchmark_weight",
        )
        if len(selection_candidates) < min_universe:
            return None

    proxy_benchmark_returns = compute_portfolio_daily_returns(
        stock_window.loc[:, risk_model_names],
        proxy_benchmark_weights,
    )
    proxy_benchmark_returns.name = "proxy_benchmark_return"

    fit_full = fit_time_series_factor_model(
        stock_returns=stock_window.loc[:, risk_model_names],
        raw_factor_returns=raw_factor_returns,
        proxy_benchmark_returns=proxy_benchmark_returns,
        proxy_benchmark_weights=proxy_benchmark_weights,
        min_valid_obs_ratio=0.0,
    )

    eligible_names = pd.Index(fit_full.beta.index)
    eligible_scores = signal_scores.reindex(selection_candidates.intersection(eligible_names)).dropna()
    if len(eligible_scores) < min_universe:
        return None

    selected_count = max(1, int(len(eligible_scores) * top_pct))
    selected_names = pd.Index(eligible_scores.nlargest(selected_count).index)
    original_weights = pd.Series(
        1.0 / len(selected_names),
        index=selected_names,
        dtype=float,
        name="weight",
    )

    investable_fit = fit_full.subset(selected_names)
    stored_proxy_weights = fit_full.proxy_benchmark_weights
    if stored_proxy_weights is not None:
        proxy_benchmark_weights = stored_proxy_weights.copy()

    rebalance_trading_date = _trading_date_on_or_before(market_data.returns.index, rebalance_date)
    if rebalance_trading_date is None:
        return None
    if next_rebalance_date is None:
        next_trading_date = rebalance_trading_date
    else:
        next_trading_date = _trading_date_on_or_before(market_data.returns.index, next_rebalance_date)
        if next_trading_date is None:
            return None

    holding_mask = (market_data.returns.index > rebalance_trading_date) & (
        market_data.returns.index <= next_trading_date
    )
    holding_period_returns = market_data.returns.loc[holding_mask, selected_names]

    trailing_stock_returns = stock_window.loc[:, selected_names]
    trailing_historical_benchmark_returns = (
        market_data.benchmark["ben_daily_return"].reindex(trailing_stock_returns.index).fillna(0.0)
    )
    holding_historical_benchmark_returns = (
        market_data.benchmark["ben_daily_return"].reindex(holding_period_returns.index).fillna(0.0)
    )

    return PreparedPeriod(
        rebalance_date=pd.Timestamp(rebalance_date),
        next_rebalance_date=pd.Timestamp(next_rebalance_date),
        eligible_names=eligible_names,
        selected_names=selected_names,
        benchmark_source=benchmark_source,
        benchmark_holdings_date=benchmark_holdings_date,
        benchmark_local_coverage_ratio=benchmark_local_coverage_ratio,
        proxy_benchmark_weights=proxy_benchmark_weights,
        fit_full=fit_full,
        investable_fit=investable_fit,
        signal_scores=eligible_scores.reindex(selected_names),
        original_weights=original_weights,
        trailing_stock_returns=trailing_stock_returns,
        holding_period_returns=holding_period_returns,
        trailing_historical_benchmark_returns=trailing_historical_benchmark_returns,
        holding_historical_benchmark_returns=holding_historical_benchmark_returns,
    )


def performance_summary(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnovers: list[float],
    annualization: int = ANNUALIZATION,
) -> dict[str, float]:
    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping portfolio and benchmark returns for performance summary.")

    aligned["active"] = aligned["portfolio"] - aligned["benchmark"]
    days = len(aligned)
    portfolio_nav = (1.0 + aligned["portfolio"]).cumprod()
    benchmark_nav = (1.0 + aligned["benchmark"]).cumprod()

    cumulative_return = float(portfolio_nav.iloc[-1] - 1.0)
    benchmark_cumulative_return = float(benchmark_nav.iloc[-1] - 1.0)
    annualized_return = float(portfolio_nav.iloc[-1] ** (annualization / days) - 1.0)
    benchmark_annualized_return = float(benchmark_nav.iloc[-1] ** (annualization / days) - 1.0)
    annualized_volatility = float(aligned["portfolio"].std(ddof=1) * np.sqrt(annualization))
    historical_tracking_error = float(aligned["active"].std(ddof=1) * np.sqrt(annualization))
    active_return_annual = float(aligned["active"].mean() * annualization)
    sharpe_ratio = (
        float(aligned["portfolio"].mean() * annualization / annualized_volatility)
        if annualized_volatility > 0.0
        else np.nan
    )
    information_ratio = (
        active_return_annual / historical_tracking_error if historical_tracking_error > 0.0 else np.nan
    )
    max_drawdown = float((portfolio_nav / portfolio_nav.cummax() - 1.0).min())
    benchmark_max_drawdown = float((benchmark_nav / benchmark_nav.cummax() - 1.0).min())

    return {
        "Cumulative Return": cumulative_return,
        "Historical Benchmark Cumulative Return": benchmark_cumulative_return,
        "Annualized Return": annualized_return,
        "Historical Benchmark Annualized Return": benchmark_annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Historical Tracking Error vs S&P500": historical_tracking_error,
        "Sharpe Ratio": sharpe_ratio,
        "Information Ratio vs S&P500": information_ratio,
        "Maximum Drawdown": max_drawdown,
        "Historical Benchmark Maximum Drawdown": benchmark_max_drawdown,
        "Average Turnover": float(np.mean(turnovers)) if turnovers else np.nan,
    }


def _annualized_vol_from_variance(variance_daily: float) -> float:
    return float(np.sqrt(max(float(variance_daily), 0.0)) * np.sqrt(ANNUALIZATION))


def _series_contribution_frame(
    series: pd.Series,
    total_variance: float,
    value_column: str,
    pct_column: str,
    index_name: str,
) -> pd.DataFrame:
    frame = series.rename(value_column).reset_index()
    frame = frame.rename(columns={"index": index_name})
    denominator = float(total_variance)
    if denominator != 0.0:
        frame[pct_column] = frame[value_column] / denominator
    else:
        frame[pct_column] = np.nan
    return frame


def _stock_total_contribution_frame(total_risk: dict[str, object]) -> pd.DataFrame:
    weights = total_risk["weights"]
    frame = pd.DataFrame(
        {
            "Ticker": weights.index,
            "Weight": weights.values,
            "Total Variance Contribution Daily": total_risk["stock_variance_contribution"].reindex(weights.index).values,
            "Total Risk Contribution Annual": (
                total_risk["stock_vol_contribution"].reindex(weights.index).values * np.sqrt(ANNUALIZATION)
            ),
        }
    )
    total_variance = float(total_risk["variance_daily"])
    frame["Total Variance Contribution Pct"] = (
        frame["Total Variance Contribution Daily"] / total_variance if total_variance != 0.0 else np.nan
    )
    return frame.sort_values("Total Variance Contribution Daily", key=lambda s: s.abs(), ascending=False)


def _stock_proxy_te_contribution_frame(tracking_error: dict[str, object]) -> pd.DataFrame:
    weights = tracking_error["weights"]
    proxy_weights = tracking_error.get("proxy_weights", pd.Series(0.0, index=weights.index, dtype=float))
    active_weights = tracking_error.get("active_weights", weights - proxy_weights)
    stock_mcar = tracking_error.get("stock_proxy_mcar", pd.Series(np.nan, index=weights.index, dtype=float))
    stock_te_contrib = tracking_error.get(
        "stock_proxy_te_contribution",
        active_weights * stock_mcar,
    )
    frame = pd.DataFrame(
        {
            "Ticker": weights.index,
            "Weight": weights.values,
            "Proxy Weight": proxy_weights.reindex(weights.index).values,
            "Active Weight": active_weights.reindex(weights.index).values,
            "Proxy TE Variance Contribution Daily": (
                tracking_error["stock_proxy_variance_contribution"].reindex(weights.index).values
            ),
            "MCAR to TE": stock_mcar.reindex(weights.index).values,
            "Proxy TE Contribution Daily": stock_te_contrib.reindex(weights.index).values,
        }
    )
    te_variance = float(tracking_error["variance_daily"])
    frame["Proxy TE Variance Contribution Pct"] = (
        frame["Proxy TE Variance Contribution Daily"] / te_variance if te_variance != 0.0 else np.nan
    )
    return frame.sort_values("Proxy TE Variance Contribution Daily", key=lambda s: s.abs(), ascending=False)


def _stock_te_mcar_frame(weights: pd.Series, fit: RiskModelFit) -> pd.DataFrame:
    universe = fit.beta.index
    portfolio_weights = weights.reindex(universe).fillna(0.0).astype(float)
    if fit.proxy_benchmark_weights is None:
        proxy_weights = pd.Series(0.0, index=universe, dtype=float)
    else:
        proxy_weights = fit.proxy_benchmark_weights.reindex(universe).fillna(0.0).astype(float)
        proxy_sum = float(proxy_weights.sum())
        if proxy_sum > 0.0:
            proxy_weights = proxy_weights / proxy_sum

    active_weights = portfolio_weights - proxy_weights
    sigma = fit.covariance_matrix.loc[universe, universe].to_numpy(dtype=float)
    a = active_weights.to_numpy(dtype=float)
    te_var = max(float(a.T @ sigma @ a), 0.0)
    te_vol = float(np.sqrt(te_var))
    gradient = sigma @ a
    if te_vol > 0.0:
        mcar = gradient / te_vol
    else:
        mcar = np.full_like(gradient, np.nan, dtype=float)
    te_contrib = a * mcar
    recon_error = float(np.nansum(te_contrib) - te_vol)

    frame = pd.DataFrame(
        {
            "Ticker": universe,
            "Weight": portfolio_weights.values,
            "Proxy Weight": proxy_weights.values,
            "Active Weight": active_weights.values,
            "MCAR to TE": mcar,
            "Proxy TE Contribution Daily": te_contrib,
            "Proxy TE Daily": te_vol,
            "MCAR Reconciliation Error": recon_error,
        }
    )
    frame["Proxy TE Contribution Pct"] = (
        frame["Proxy TE Contribution Daily"] / te_vol if te_vol > 0.0 else np.nan
    )
    return frame.sort_values("Proxy TE Contribution Daily", key=lambda s: s.abs(), ascending=False)


def build_risk_snapshot_rows(
    date: pd.Timestamp,
    portfolio_name: str,
    total_risk: dict[str, object],
    tracking_error: dict[str, object],
    turnover: float | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    snapshot_rows = [
        {
            "Date": date,
            "Portfolio": portfolio_name,
            "Predicted Total Risk": total_risk["vol_annual"],
            "Predicted Proxy Tracking Error": tracking_error["vol_annual"],
            "Systematic Risk": total_risk["systematic_vol_annual"],
            "Idiosyncratic Risk": total_risk["idiosyncratic_vol_annual"],
            "Proxy Active Systematic Risk": _annualized_vol_from_variance(tracking_error["factor_variance_daily"]),
            "Proxy Active Idiosyncratic Risk": _annualized_vol_from_variance(
                tracking_error["idiosyncratic_variance_daily"]
            ),
            "Proxy Benchmark Idiosyncratic Variance Daily": tracking_error[
                "proxy_benchmark_idiosyncratic_variance_daily"
            ],
            "Turnover": turnover,
        }
    ]

    factor_total = _series_contribution_frame(
        total_risk["factor_variance_contribution"],
        float(total_risk["variance_daily"]),
        "Total Variance Contribution Daily",
        "Total Variance Contribution Pct",
        "Factor",
    )
    factor_proxy = _series_contribution_frame(
        tracking_error["factor_variance_contribution"],
        float(tracking_error["variance_daily"]),
        "Proxy Active Variance Contribution Daily",
        "Proxy Active Variance Contribution Pct",
        "Factor",
    )
    stock_total = _stock_total_contribution_frame(total_risk)
    stock_proxy = _stock_proxy_te_contribution_frame(tracking_error)

    for frame in (factor_total, factor_proxy, stock_total, stock_proxy):
        frame.insert(0, "Portfolio", portfolio_name)
        frame.insert(0, "Date", date)

    return (
        snapshot_rows,
        factor_total.to_dict("records"),
        factor_proxy.to_dict("records"),
        stock_total.to_dict("records"),
        stock_proxy.to_dict("records"),
    )


def _build_portfolio_weights(
    prepared_period: PreparedPeriod,
    cap: float,
) -> tuple[dict[str, pd.Series], list[dict[str, object]]]:
    original_weights = prepared_period.original_weights
    total_original = portfolio_total_risk(original_weights, prepared_period.investable_fit)
    min_te = optimize_min_tracking_error(
        fit=prepared_period.investable_fit,
        initial_weights=original_weights,
        cap=cap,
    )
    mvo = optimize_mvo_with_risk_cap(
        expected_scores=compute_expected_scores(prepared_period.signal_scores),
        fit=prepared_period.investable_fit,
        initial_weights=original_weights,
        target_variance=float(total_original["variance_daily"]),
        cap=cap,
    )

    weights = {
        "Original": original_weights,
        "Minimum_TE": min_te.weights,
        "MVO": mvo.weights,
    }
    status_rows = [
        {
            "Date": prepared_period.rebalance_date,
            "Portfolio": "Original",
            "Success": True,
            "Objective Value": np.nan,
            "Message": "Equal-weight top 5% signal portfolio.",
        },
        {
            "Date": prepared_period.rebalance_date,
            "Portfolio": "Minimum_TE",
            "Success": min_te.success,
            "Objective Value": min_te.objective_value,
            "Message": min_te.message,
        },
        {
            "Date": prepared_period.rebalance_date,
            "Portfolio": "MVO",
            "Success": mvo.success,
            "Objective Value": mvo.objective_value,
            "Message": mvo.message,
        },
    ]
    return weights, status_rows


def build_latest_risk_report(prepared_period: PreparedPeriod, weights: pd.Series) -> dict[str, object]:
    total_risk = portfolio_total_risk(weights, prepared_period.investable_fit)
    tracking_error = portfolio_tracking_error(weights, prepared_period.investable_fit)
    trailing_portfolio_returns = compute_portfolio_daily_returns(
        prepared_period.trailing_stock_returns,
        weights,
    )
    historical = historical_risk_statistics(
        trailing_portfolio_returns,
        prepared_period.trailing_historical_benchmark_returns,
    )

    return_history = historical["return_history"].reset_index()
    return_history = return_history.rename(
        columns={
            "index": "Date",
            "portfolio": "Portfolio Return",
            "benchmark": "Historical Benchmark Return",
            "active": "Active Return vs S&P500",
        }
    )

    summary = pd.DataFrame(
        [
            {"Metric": "Latest Rebalance Date", "Value": prepared_period.rebalance_date.date().isoformat()},
            {
                "Metric": "Next Rebalance Date",
                "Value": (
                    prepared_period.next_rebalance_date.date().isoformat()
                    if prepared_period.next_rebalance_date is not None
                    else "-"
                ),
            },
            {"Metric": "Eligible Universe Count", "Value": len(prepared_period.eligible_names)},
            {"Metric": "Selected Portfolio Count", "Value": len(prepared_period.selected_names)},
            {"Metric": "Benchmark Weight Source", "Value": prepared_period.benchmark_source},
            {
                "Metric": "Benchmark Holdings Date",
                "Value": (
                    prepared_period.benchmark_holdings_date.date().isoformat()
                    if prepared_period.benchmark_holdings_date is not None
                    else "-"
                ),
            },
            {
                "Metric": "Benchmark Local Coverage Ratio",
                "Value": prepared_period.benchmark_local_coverage_ratio,
            },
            {"Metric": "Historical Volatility", "Value": historical["historical_vol_annual"]},
            {
                "Metric": "Historical Tracking Error vs S&P500",
                "Value": historical["historical_te_annual"],
            },
            {"Metric": "Predicted Total Risk", "Value": total_risk["vol_annual"]},
            {"Metric": "Predicted Proxy Tracking Error", "Value": tracking_error["vol_annual"]},
            {"Metric": "Systematic Risk", "Value": total_risk["systematic_vol_annual"]},
            {"Metric": "Idiosyncratic Risk", "Value": total_risk["idiosyncratic_vol_annual"]},
            {
                "Metric": "Total Variance Annualized",
                "Value": float(total_risk["variance_daily"]) * ANNUALIZATION,
            },
            {
                "Metric": "Systematic Variance Annualized",
                "Value": float(total_risk["systematic_variance_daily"]) * ANNUALIZATION,
            },
            {
                "Metric": "Idiosyncratic Variance Annualized",
                "Value": float(total_risk["idiosyncratic_variance_daily"]) * ANNUALIZATION,
            },
            {
                "Metric": "Factor Explained Ratio",
                "Value": (
                    float(total_risk["systematic_variance_daily"]) / float(total_risk["variance_daily"])
                    if float(total_risk["variance_daily"]) > 0.0
                    else np.nan
                ),
            },
            {
                "Metric": "Idiosyncratic Ratio",
                "Value": (
                    float(total_risk["idiosyncratic_variance_daily"]) / float(total_risk["variance_daily"])
                    if float(total_risk["variance_daily"]) > 0.0
                    else np.nan
                ),
            },
            {
                "Metric": "Proxy Active Systematic Risk",
                "Value": _annualized_vol_from_variance(tracking_error["factor_variance_daily"]),
            },
            {
                "Metric": "Proxy Active Idiosyncratic Risk",
                "Value": _annualized_vol_from_variance(tracking_error["idiosyncratic_variance_daily"]),
            },
            {
                "Metric": "MCAR Reconciliation Error",
                "Value": tracking_error["mcar_reconciliation_error"],
            },
            {
                "Metric": "MCAR Mean Absolute Value",
                "Value": float(tracking_error["stock_proxy_mcar"].abs().mean()),
            },
            {
                "Metric": "MCAR Max Absolute Value",
                "Value": float(tracking_error["stock_proxy_mcar"].abs().max()),
            },
            {
                "Metric": "Proxy Benchmark Idiosyncratic Variance Daily",
                "Value": tracking_error["proxy_benchmark_idiosyncratic_variance_daily"],
            },
        ]
    )

    factor_total = _series_contribution_frame(
        total_risk["factor_variance_contribution"],
        float(total_risk["variance_daily"]),
        "Total Variance Contribution Daily",
        "Total Variance Contribution Pct",
        "Factor",
    )
    factor_proxy = _series_contribution_frame(
        tracking_error["factor_variance_contribution"],
        float(tracking_error["variance_daily"]),
        "Proxy Active Variance Contribution Daily",
        "Proxy Active Variance Contribution Pct",
        "Factor",
    )
    stock_total = _stock_total_contribution_frame(total_risk)
    stock_proxy = _stock_proxy_te_contribution_frame(tracking_error)
    stock_te_mcar = _stock_te_mcar_frame(weights, prepared_period.fit_full)
    weights_frame = weights.sort_values(ascending=False).rename("Weight").reset_index()
    weights_frame = weights_frame.rename(columns={"index": "Ticker"})

    return {
        "risk_summary": summary,
        "historical_var": historical["historical_var"],
        "normal_var": historical["normal_var"],
        "return_history": return_history,
        "weights": weights_frame,
        "factor_total_contribution": factor_total,
        "factor_proxy_active_contribution": factor_proxy,
        "total_risk_contributors": stock_total,
        "proxy_te_contributors": stock_proxy,
        "te_mcar": stock_te_mcar,
        "total_risk": total_risk,
        "tracking_error": tracking_error,
        "investable_fit": prepared_period.investable_fit,
    }


def plot_cumulative_returns(returns_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if returns_df.empty:
        return
    cumulative = (1.0 + returns_df.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    cumulative.plot(ax=ax, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Net Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return
    plot_data = data.copy()
    plot_data["_abs_value"] = plot_data[y_col].abs()
    plot_data = plot_data.sort_values("_abs_value", ascending=False).head(top_n)
    plot_data = plot_data.sort_values(y_col)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data[x_col].astype(str), plot_data[y_col])
    ax.set_title(title)
    ax.set_xlabel(y_col)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_risk_timeseries(
    risk_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    if risk_df.empty or value_column not in risk_df.columns:
        return
    pivot = risk_df.pivot(index="Date", columns="Portfolio", values=value_column)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(value_column)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _safe_sheet_name(name: str, used: set[str]) -> str:
    clean = "".join(ch if ch.isalnum() or ch in (" ", "_") else "_" for ch in name)[:31]
    if clean not in used:
        used.add(clean)
        return clean
    base = clean[:27]
    counter = 1
    while f"{base}_{counter}" in used:
        counter += 1
    sheet_name = f"{base}_{counter}"
    used.add(sheet_name)
    return sheet_name


def export_dataframe_bundle_excel(dataframes: dict[str, pd.DataFrame], output_path: Path) -> None:
    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(output_path) as writer:
        for name, frame in dataframes.items():
            if frame is None:
                continue
            sheet_name = _safe_sheet_name(name, used_sheet_names)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_methods_note(
    output_dir: Path,
    mode: str,
    latest_date: pd.Timestamp | None,
    benchmark_weights: BenchmarkWeightsData | None,
) -> Path:
    latest_line = (
        f"- Latest risk snapshot date: {latest_date.date().isoformat()}.\n"
        if latest_date is not None
        else ""
    )
    if benchmark_weights is None:
        benchmark_section = """- Predicted tracking error is labelled as proxy tracking error because benchmark constituent weights are not available.

## Proxy Benchmark For Risk Model
- At each rebalance date, the eligible universe is the set of stocks with a valid Project 1 signal and enough trailing return history in the risk window.
- The proxy risk benchmark is equal-weight across that eligible universe.
- Minimum tracking error and predicted active risk are optimized and reported relative to this proxy benchmark, not relative to true S&P500 constituent weights.
"""
        limitation_line = "- No benchmark constituent weight history is available, so all forecast tracking error and active risk numbers are proxy estimates."
    else:
        benchmark_section = """- Predicted tracking error is labelled as proxy tracking error because benchmark constituent weights come from IVV ETF holdings, not the licensed S&P 500 index file.

## Proxy Benchmark For Risk Model
- At each rebalance date, benchmark weights are taken from official iShares IVV holdings on the rebalance trading date or the closest prior available holdings date.
- Because the local stock panel does not contain every historical S&P 500 constituent, IVV holdings are restricted to the observable local universe and renormalized within that local universe.
- The stock-selection and optimization universe is also restricted to names present in the same IVV holdings proxy at that rebalance date.
- Minimum tracking error and predicted active risk are optimized and reported relative to this IVV-holdings proxy benchmark, not relative to the licensed full S&P500 constituent-weight file.
"""
        limitation_line = "- Benchmark constituent weights and historical investable-universe membership are proxied by IVV ETF holdings restricted to the observable local universe, so forecast tracking error remains a proxy estimate."

    note = f"""# Project 2 Method Notes

## Scope
- Output mode: {mode}.
{latest_line}- Historical performance and historical tracking error are measured against the S&P500 index return in `Project1/SP500.xlsx`.
- Rebalance dates use completed month-end signals only. A trailing partial month is excluded to avoid labeling an incomplete signal as a month-end rebalance.
- Historical performance uses completed holding-period returns only.
{benchmark_section}

## Factor Risk Model
- The model uses a time-series factor regression with no intercept.
- The trailing estimation window expands from the start of the sample until 252 daily observations are available, and then stays on a 252-day cap.
- Factor order follows the course slides' orthogonalization logic and now uses a Barra USE4-inspired public proxy model.
- Risk factors are Market, public sector factors, and style factor-mimicking returns: Size, Nonlinear Size, Momentum, Residual Volatility, Liquidity, Book-to-Price, Earnings Yield, Leverage, and Dividend Yield.
- Project 1 signals are still used for stock selection, but Mom_Gap, Rel_Strength, and Vol_Trend are no longer used as risk factors.
- Total risk is decomposed into systematic factor risk and idiosyncratic risk.
- Proxy tracking error is decomposed into proxy active factor risk and active idiosyncratic risk.

## Portfolio Construction
- Original: equal-weight among the top 5% Project 1 signal names within the eligible universe.
- Minimum_TE: long-only portfolio on the same selected names, fully invested, with a per-name cap, minimizing predicted proxy tracking error.
- MVO: long-only portfolio on the same selected names, fully invested, with a per-name cap, maximizing standardized signal score under the Original portfolio's predicted total variance cap.

## Data Limitations
{limitation_line}
- Barra-style proxy sector and fundamental exposures come from public data cached in `Project2_data/barra_proxy_metadata.csv`; this is not licensed MSCI Barra data.
- Sector classifications and fundamentals are public-data proxies and are not guaranteed to be fully historical point-in-time.
- Historical return comparison remains against the S&P500 index itself.
"""
    output_path = output_dir / "Project2_method_notes.md"
    output_path.write_text(note, encoding="utf-8")
    return output_path


def plot_risk_timeseries(
    risk_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    if risk_df.empty or value_column not in risk_df.columns:
        return
    pivot = risk_df.pivot(index="Date", columns="Portfolio", values=value_column)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(value_column)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_factor_attribution_by_portfolio(factor_attrib_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot factor attribution (annualized variance) by portfolio."""
    if factor_attrib_df.empty:
        return
    
    portfolios = sorted(factor_attrib_df["Portfolio"].unique())
    colors = {"Market": "#FF6B6B", "Sector": "#4ECDC4", "Style": "#45B7D1"}
    
    fig, axes = plt.subplots(1, len(portfolios), figsize=(6 * len(portfolios), 6))
    if len(portfolios) == 1:
        axes = [axes]
    
    fig.suptitle("Top Factor Contributors to Systematic Active Risk (Annualized Variance)", 
                 fontsize=14, fontweight="bold")
    
    for ax, portfolio in zip(axes, portfolios):
        df = factor_attrib_df[factor_attrib_df["Portfolio"] == portfolio].copy()
        df = df.sort_values("ActV_Factor_AnnualizedVariance", ascending=False)
        
        bar_colors = [colors.get(block, "#999999") for block in df["Block"]]
        bars = ax.barh(df["Factor"], df["ActV_Factor_AnnualizedVariance"], color=bar_colors)
        ax.set_xlabel("Annualized Variance", fontsize=11)
        ax.set_title(f"{portfolio}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        
        # Add value labels
        for bar, val in zip(bars, df["ActV_Factor_AnnualizedVariance"]):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.6f}", 
                    va="center", ha="left", fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k) for k in ["Market", "Sector", "Style"]]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), 
               ncol=3, fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_dir / "latest_systematic_active_risk_top_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_factor_attribution_top_20(factor_attrib_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot top 20 factor contributors combined across all portfolios."""
    if factor_attrib_df.empty:
        return
    
    colors = {"Market": "#FF6B6B", "Sector": "#4ECDC4", "Style": "#45B7D1"}
    
    # Create combined label and sort
    df = factor_attrib_df.copy()
    df["Portfolio_Factor"] = df["Portfolio"] + " - " + df["Factor"]
    df_top = df.nlargest(20, "ActV_Factor_AnnualizedVariance")
    
    bar_colors = [colors.get(block, "#999999") for block in df_top["Block"]]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_top["Portfolio_Factor"], df_top["ActV_Factor_AnnualizedVariance"], color=bar_colors)
    ax.set_xlabel("Annualized Variance", fontsize=12, fontweight="bold")
    ax.set_title("Top 20 Factor Contributors to Systematic Active Risk", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Add value labels
    for bar, val in zip(bars, df_top["ActV_Factor_AnnualizedVariance"]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.6f}", 
                va="center", ha="left", fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k) for k in ["Market", "Sector", "Style"]]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), 
               ncol=3, fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_dir / "latest_systematic_active_risk_top_20_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_block_attribution_by_portfolio(block_attrib_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot block-level (Market/Sector/Style) attribution by portfolio."""
    if block_attrib_df.empty:
        return
    
    portfolios = sorted(block_attrib_df["Portfolio"].unique())
    colors = {"Market": "#FF6B6B", "Sector": "#4ECDC4", "Style": "#45B7D1"}
    
    fig, axes = plt.subplots(1, len(portfolios), figsize=(6 * len(portfolios), 5))
    if len(portfolios) == 1:
        axes = [axes]
    
    fig.suptitle("Block Attribution to Systematic Active Risk (Annualized Variance)", 
                 fontsize=14, fontweight="bold")
    
    for ax, portfolio in zip(axes, portfolios):
        df = block_attrib_df[block_attrib_df["Portfolio"] == portfolio].copy()
        df = df.sort_values("ActV_Block_AnnualizedVariance", ascending=True)
        
        bar_colors = [colors.get(block, "#999999") for block in df["Block"]]
        bars = ax.barh(df["Block"], df["ActV_Block_AnnualizedVariance"], color=bar_colors)
        ax.set_xlabel("Annualized Variance", fontsize=11)
        ax.set_title(f"{portfolio}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        
        # Add value labels
        for bar, val in zip(bars, df["ActV_Block_AnnualizedVariance"]):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.6f}", 
                    va="center", ha="left", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_dir / "latest_systematic_active_risk_block_attribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _export_factor_correlation_matrix(
    latest_artifacts: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    """Export post-orthogonalization factor correlation matrix (CSV + heatmap)."""
    fit = None
    for portfolio_name in PORTFOLIO_ORDER:
        candidate_fit = latest_artifacts.get(portfolio_name, {}).get("investable_fit")
        if candidate_fit is not None:
            fit = candidate_fit
            break
    if fit is None:
        return

    factor_returns = fit.factor_returns.copy().astype(float)
    if factor_returns.empty:
        return

    corr = factor_returns.corr()
    corr.to_csv(output_dir / "latest_post_orthogonalization_factor_correlation.csv", index=True)

    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel("Correlation", rotation=270, labelpad=14)

    labels = list(corr.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Post-Orthogonalization Factor Correlation")

    corr_values = corr.to_numpy(dtype=float)
    for row_index in range(corr_values.shape[0]):
        for col_index in range(corr_values.shape[1]):
            value = corr_values[row_index, col_index]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=5, color=text_color)

    fig.tight_layout()
    fig.savefig(output_dir / "latest_post_orthogonalization_factor_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_factor_attribution_and_blocks(
    latest_artifacts: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute systematic active risk factor attribution and block decomposition."""
    
    rows_factor = []
    rows_block = []
    rows_check = []
    
    def factor_block_name(f: str) -> str:
        if f == "Market":
            return "Market"
        if f.startswith("Sector_"):
            return "Sector"
        return "Style"
    
    for portfolio in PORTFOLIO_ORDER:
        fit = latest_artifacts[portfolio].get("investable_fit")
        if fit is None:
            continue
            
        weights = latest_artifacts[portfolio].get("weights")
        if weights is None or weights.empty:
            continue
        
        w_aligned = weights.set_index("Ticker")["Weight"] if isinstance(weights, pd.DataFrame) else weights
        w_aligned = w_aligned.reindex(fit.beta.index).fillna(0.0).astype(float)
        
        if fit.proxy_benchmark_weights is None:
            bmk_w = pd.Series(0.0, index=fit.beta.index, dtype=float)
        else:
            bmk_w = fit.proxy_benchmark_weights.reindex(fit.beta.index).fillna(0.0).astype(float)
            s = float(bmk_w.sum())
            if s > 0:
                bmk_w = bmk_w / s
        
        a = (w_aligned - bmk_w).astype(float)
        a_vec = a.to_numpy(dtype=float)
        B = fit.beta.to_numpy(dtype=float)
        F = fit.factor_cov.to_numpy(dtype=float)
        
        total_sys = float(a_vec @ B @ F @ B.T @ a_vec)
        
        factors = list(fit.factor_cov.columns)
        market_cols = [c for c in factors if c == "Market"]
        sector_cols = [c for c in factors if c.startswith("Sector_")]
        style_cols = [c for c in factors if c not in market_cols + sector_cols]
        
        factor_sum = 0.0
        for factor in factors:
            F_i = pd.DataFrame(0.0, index=fit.factor_cov.index, columns=fit.factor_cov.columns, dtype=float)
            F_i.loc[factor, factor] = fit.factor_cov.loc[factor, factor]
            attrib_i = float(a_vec @ B @ F_i.to_numpy(dtype=float) @ B.T @ a_vec)
            factor_sum += attrib_i
            block_name = factor_block_name(factor)
            
            rows_factor.append({
                "Portfolio": portfolio,
                "Factor": factor,
                "Block": block_name,
                "ActV_Factor_Daily": attrib_i,
                "ActV_Factor_AnnualizedVariance": attrib_i * ANNUALIZATION,
            })
        
        # Compute block-level decomposition
        for block_name in ["Market", "Sector", "Style"]:
            cols_in_block = {
                "Market": market_cols,
                "Sector": sector_cols,
                "Style": style_cols,
            }[block_name]
            
            F_block = pd.DataFrame(0.0, index=fit.factor_cov.index, columns=fit.factor_cov.columns, dtype=float)
            for col in cols_in_block:
                F_block.loc[col, col] = fit.factor_cov.loc[col, col]
            
            block_var = float(a_vec @ B @ F_block.to_numpy(dtype=float) @ B.T @ a_vec)
            
            rows_block.append({
                "Portfolio": portfolio,
                "Block": block_name,
                "ActV_Block_Daily": block_var,
                "ActV_Block_AnnualizedVariance": block_var * ANNUALIZATION,
            })
        
        diff = factor_sum - total_sys
        rows_check.append({
            "Portfolio": portfolio,
            "ActV_Systematic_Total_Daily": total_sys,
            "Factor_Attribution_Sum_Daily": factor_sum,
            "Diff_FactorSumMinusTotal_Daily": diff,
            "Abs_Diff": abs(diff),
            "ActV_Systematic_Total_AnnualizedVariance": total_sys * ANNUALIZATION,
            "Factor_Attribution_Sum_AnnualizedVariance": factor_sum * ANNUALIZATION,
        })
    
    return (
        pd.DataFrame(rows_factor),
        pd.DataFrame(rows_block),
        pd.DataFrame(rows_check),
    )


def _export_latest_portfolio_report(
    portfolio_name: str,
    artifacts: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_map = {
        "risk_summary": "risk_summary.csv",
        "historical_var": "historical_var.csv",
        "normal_var": "normal_var.csv",
        "return_history": "return_history.csv",
        "weights": "weights.csv",
        "factor_total_contribution": "factor_total_contribution.csv",
        "factor_proxy_active_contribution": "factor_proxy_active_contribution.csv",
        "total_risk_contributors": "total_risk_contributors.csv",
        "proxy_te_contributors": "proxy_te_contributors.csv",
    }
    for key, filename in csv_map.items():
        artifacts[key].to_csv(output_dir / filename, index=False)

    return_history = artifacts["return_history"].copy()
    return_history["Date"] = pd.to_datetime(return_history["Date"])
    returns_for_plot = return_history.set_index("Date")[
        ["Portfolio Return", "Historical Benchmark Return"]
    ].rename(columns={"Portfolio Return": portfolio_name, "Historical Benchmark Return": HISTORICAL_BENCHMARK_NAME})
    plot_cumulative_returns(
        returns_for_plot,
        output_dir / "portfolio_vs_historical_benchmark.png",
        f"{portfolio_name}: trailing returns vs {HISTORICAL_BENCHMARK_NAME}",
    )
    plot_bar_chart(
        artifacts["total_risk_contributors"],
        "Ticker",
        "Total Variance Contribution Daily",
        f"{portfolio_name}: top total risk contributors",
        output_dir / "total_risk_contributors.png",
    )
    plot_bar_chart(
        artifacts["proxy_te_contributors"],
        "Ticker",
        "Proxy TE Variance Contribution Daily",
        f"{portfolio_name}: top proxy TE contributors",
        output_dir / "proxy_te_contributors.png",
    )
    plot_bar_chart(
        artifacts["factor_total_contribution"],
        "Factor",
        "Total Variance Contribution Daily",
        f"{portfolio_name}: factor total risk contribution",
        output_dir / "factor_total_contribution.png",
        top_n=20,
    )
    plot_bar_chart(
        artifacts["factor_proxy_active_contribution"],
        "Factor",
        "Proxy Active Variance Contribution Daily",
        f"{portfolio_name}: factor proxy active contribution",
        output_dir / "factor_proxy_active_contribution.png",
        top_n=20,
    )


def _with_portfolio(frame: pd.DataFrame, portfolio_name: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "Portfolio", portfolio_name)
    return output


def _build_annual_risk_explainability(latest_artifacts: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for portfolio_name in PORTFOLIO_ORDER:
        total_risk = latest_artifacts.get(portfolio_name, {}).get("total_risk")
        if not isinstance(total_risk, dict):
            continue

        total_var_daily = float(total_risk.get("variance_daily", 0.0))
        systematic_var_daily = float(total_risk.get("systematic_variance_daily", 0.0))
        idiosyncratic_var_daily = float(total_risk.get("idiosyncratic_variance_daily", 0.0))

        total_var_annualized = max(total_var_daily, 0.0) * ANNUALIZATION
        systematic_var_annualized = max(systematic_var_daily, 0.0) * ANNUALIZATION
        idiosyncratic_var_annualized = max(idiosyncratic_var_daily, 0.0) * ANNUALIZATION

        explained_ratio = (
            systematic_var_annualized / total_var_annualized
            if total_var_annualized > 0.0
            else np.nan
        )
        unexplained_ratio = (
            idiosyncratic_var_annualized / total_var_annualized
            if total_var_annualized > 0.0
            else np.nan
        )

        rows.append(
            {
                "Portfolio": portfolio_name,
                "Total Variance Annualized": total_var_annualized,
                "Systematic Variance Annualized": systematic_var_annualized,
                "Idiosyncratic Variance Annualized": idiosyncratic_var_annualized,
                "Factor Explained Ratio": explained_ratio,
                "Idiosyncratic Ratio": unexplained_ratio,
            }
        )
    return pd.DataFrame(rows)


def _plot_risk_explainability(annual_explainability: pd.DataFrame, output_dir: Path) -> None:
    if annual_explainability.empty:
        return

    plot_df = annual_explainability.set_index("Portfolio").reindex(PORTFOLIO_ORDER).dropna(how="all")
    if plot_df.empty:
        return

    explained_pct = (plot_df["Factor Explained Ratio"].clip(lower=0.0, upper=1.0) * 100.0).fillna(0.0)
    idio_pct = (plot_df["Idiosyncratic Ratio"].clip(lower=0.0, upper=1.0) * 100.0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = plot_df.index.astype(str)
    ax.bar(x_labels, explained_pct, label="Systematic (Factor)", color="#4ECDC4")
    ax.bar(x_labels, idio_pct, bottom=explained_pct, label="Idiosyncratic", color="#FF6B6B")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of Total Risk Variance (%)")
    ax.set_title("How Much Risk Factors Explain Total Risk (Annualized Variance)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for idx, (portfolio, explained) in enumerate(zip(x_labels, explained_pct.to_numpy(dtype=float))):
        ax.text(idx, min(explained + 1.5, 99.0), f"{explained:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "latest_risk_factor_explainability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(plot_df.index))
    width = 0.38
    systematic_var = plot_df["Systematic Variance Annualized"].to_numpy(dtype=float)
    idio_var = plot_df["Idiosyncratic Variance Annualized"].to_numpy(dtype=float)
    bars1 = ax2.bar(x_pos - width / 2, systematic_var, width=width, color="#45B7D1", label="Systematic Variance")
    bars2 = ax2.bar(x_pos + width / 2, idio_var, width=width, color="#FFA726", label="Idiosyncratic Variance")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_df.index.astype(str))
    ax2.set_ylabel("Annualized Variance")
    ax2.set_title("Systematic vs Idiosyncratic Risk (Annualized Variance)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right")
    for bar in list(bars1) + list(bars2):
        height = float(bar.get_height())
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}", ha="center", va="bottom", fontsize=9)

    fig2.tight_layout()
    fig2.savefig(output_dir / "latest_risk_variance_decomposition_annualized.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def _plot_mcar_summary(latest_artifacts: dict[str, dict[str, object]], output_dir: Path) -> None:
    rows: list[dict[str, float | str]] = []
    for portfolio_name in PORTFOLIO_ORDER:
        tracking_error = latest_artifacts.get(portfolio_name, {}).get("tracking_error")
        if not isinstance(tracking_error, dict):
            continue
        rows.append(
            {
                "Portfolio": portfolio_name,
                "MCAR Reconciliation Error": float(tracking_error.get("mcar_reconciliation_error", np.nan)),
                "MCAR Mean Absolute Value": float(tracking_error.get("stock_proxy_mcar", pd.Series(dtype=float)).abs().mean()),
                "MCAR Max Absolute Value": float(tracking_error.get("stock_proxy_mcar", pd.Series(dtype=float)).abs().max()),
            }
        )

    if not rows:
        return

    plot_df = pd.DataFrame(rows).set_index("Portfolio").reindex(PORTFOLIO_ORDER).dropna(how="all")
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MCAR Summary by Portfolio", fontsize=14, fontweight="bold")

    metrics = [
        ("MCAR Reconciliation Error", "Reconciliation Error", "#7E57C2"),
        ("MCAR Mean Absolute Value", "Mean |MCAR|", "#4ECDC4"),
        ("MCAR Max Absolute Value", "Max |MCAR|", "#FF8A65"),
    ]

    for ax, (column, title, color) in zip(axes, metrics):
        values = plot_df[column].to_numpy(dtype=float)
        bars = ax.bar(plot_df.index.astype(str), values, color=color)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=0)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "latest_mcar_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mcar_top_contributors(latest_te_mcar: pd.DataFrame, output_dir: Path) -> None:
    if latest_te_mcar.empty:
        return

    plot_df = latest_te_mcar.copy()
    if "Portfolio" not in plot_df.columns or "Ticker" not in plot_df.columns:
        return
    if "Proxy TE Contribution Daily" not in plot_df.columns:
        return

    portfolios = [portfolio for portfolio in PORTFOLIO_ORDER if portfolio in set(plot_df["Portfolio"])]
    if not portfolios:
        portfolios = sorted(plot_df["Portfolio"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, len(portfolios), figsize=(6.4 * len(portfolios), 6.2), sharex=False)
    if len(portfolios) == 1:
        axes = [axes]

    fig.suptitle("Top MCAR-based Proxy TE Contributors by Portfolio", fontsize=15, fontweight="bold")

    for ax, portfolio_name in zip(axes, portfolios):
        portfolio_df = plot_df.loc[plot_df["Portfolio"] == portfolio_name].copy()
        if portfolio_df.empty:
            ax.set_axis_off()
            continue

        portfolio_df["Abs TE"] = portfolio_df["Proxy TE Contribution Daily"].abs()
        portfolio_df = portfolio_df.sort_values("Abs TE", ascending=False).head(10)
        portfolio_df = portfolio_df.sort_values("Proxy TE Contribution Daily")

        bars = ax.barh(
            portfolio_df["Ticker"].astype(str),
            portfolio_df["Proxy TE Contribution Daily"],
            color="#2B7BBA",
        )
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(portfolio_name, fontsize=13)
        ax.set_xlabel("TE contrib (daily)")
        ax.grid(True, axis="x", alpha=0.25)

        for bar, value in zip(bars, portfolio_df["Proxy TE Contribution Daily"].to_numpy(dtype=float)):
            xpos = value + (0.00003 if value >= 0 else -0.00003)
            ha = "left" if value >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{value:.6f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "latest_mcar_top_contributors_by_portfolio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _export_latest_root_files(
    latest_artifacts: dict[str, dict[str, object]],
    latest_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    combined_historical_var = pd.concat(
        [_with_portfolio(latest_artifacts[name]["historical_var"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_normal_var = pd.concat(
        [_with_portfolio(latest_artifacts[name]["normal_var"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_return_history = pd.concat(
        [_with_portfolio(latest_artifacts[name]["return_history"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_total_risk_contributors = pd.concat(
        [_with_portfolio(latest_artifacts[name]["total_risk_contributors"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_proxy_te_contributors = pd.concat(
        [_with_portfolio(latest_artifacts[name]["proxy_te_contributors"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_factor_total = pd.concat(
        [_with_portfolio(latest_artifacts[name]["factor_total_contribution"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_factor_proxy = pd.concat(
        [_with_portfolio(latest_artifacts[name]["factor_proxy_active_contribution"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )
    combined_te_mcar = pd.concat(
        [_with_portfolio(latest_artifacts[name]["te_mcar"], name) for name in PORTFOLIO_ORDER],
        ignore_index=True,
    )

    latest_comparison.to_csv(output_dir / "latest_risk_summary.csv", index=False)
    combined_historical_var.to_csv(output_dir / "latest_historical_var.csv", index=False)
    combined_normal_var.to_csv(output_dir / "latest_normal_var.csv", index=False)
    combined_return_history.to_csv(output_dir / "latest_return_history.csv", index=False)
    combined_total_risk_contributors.to_csv(output_dir / "latest_total_risk_contributors.csv", index=False)
    combined_proxy_te_contributors.to_csv(output_dir / "latest_proxy_te_contributors.csv", index=False)
    combined_proxy_te_contributors.to_csv(output_dir / "latest_te_contributors.csv", index=False)
    latest_date_map = latest_comparison[["Portfolio", "Latest Rebalance Date"]].drop_duplicates("Portfolio")
    latest_date_map = latest_date_map.rename(columns={"Latest Rebalance Date": "Date"})
    latest_te_mcar = combined_te_mcar.merge(latest_date_map, on="Portfolio", how="left")
    latest_te_mcar = latest_te_mcar[
        [
            "Date",
            "Portfolio",
            "Ticker",
            "Weight",
            "Proxy Weight",
            "Active Weight",
            "MCAR to TE",
            "Proxy TE Contribution Daily",
            "Proxy TE Contribution Pct",
            "Proxy TE Daily",
            "MCAR Reconciliation Error",
        ]
    ].copy()
    latest_te_mcar.to_csv(output_dir / "latest_te_mcar.csv", index=False)
    _plot_mcar_top_contributors(latest_te_mcar, output_dir)
    combined_factor_total.to_csv(output_dir / "latest_factor_total_contribution.csv", index=False)
    combined_factor_proxy.to_csv(output_dir / "latest_factor_proxy_active_contribution.csv", index=False)
    combined_factor_proxy.to_csv(output_dir / "latest_factor_te_contribution.csv", index=False)

    annual_explainability = _build_annual_risk_explainability(latest_artifacts)
    if not annual_explainability.empty:
        annual_explainability.to_csv(output_dir / "latest_risk_explainability_annualized.csv", index=False)
        _plot_risk_explainability(annual_explainability, output_dir)

    _plot_mcar_summary(latest_artifacts, output_dir)

    # Compute and export factor attribution and block decomposition
    factor_attrib_df, block_attrib_df, attrib_check_df = _compute_factor_attribution_and_blocks(latest_artifacts)
    
    if not factor_attrib_df.empty:
        factor_attrib_df.to_csv(output_dir / "latest_systematic_active_risk_factor_attribution.csv", index=False)
        attrib_check_df.to_csv(output_dir / "latest_systematic_active_risk_factor_attribution_check.csv", index=False)
        
        block_group_df = (
            factor_attrib_df
            .groupby(["Portfolio", "Block"], as_index=False)
            .agg({
                "ActV_Factor_Daily": "sum",
                "ActV_Factor_AnnualizedVariance": "sum",
            })
            .rename(columns={
                "ActV_Factor_Daily": "ActV_Block_Daily",
                "ActV_Factor_AnnualizedVariance": "ActV_Block_AnnualizedVariance",
            })
        )
        block_group_df.to_csv(output_dir / "latest_systematic_active_risk_block_from_factor_sum.csv", index=False)
        block_attrib_df.to_csv(output_dir / "latest_systematic_active_risk_block_attribution.csv", index=False)
        
        # Plot factor attribution by portfolio
        _plot_factor_attribution_by_portfolio(factor_attrib_df, output_dir)
        
        # Plot top 20 factors combined
        _plot_factor_attribution_top_20(factor_attrib_df, output_dir)
        
        # Plot block attribution by portfolio
        _plot_block_attribution_by_portfolio(block_attrib_df, output_dir)

    _export_factor_correlation_matrix(latest_artifacts, output_dir)

    returns_for_plot = pd.DataFrame()
    for portfolio_name in PORTFOLIO_ORDER:
        history = latest_artifacts[portfolio_name]["return_history"].copy()
        history["Date"] = pd.to_datetime(history["Date"])
        history = history.set_index("Date")
        returns_for_plot[portfolio_name] = history["Portfolio Return"]
        if HISTORICAL_BENCHMARK_NAME not in returns_for_plot:
            returns_for_plot[HISTORICAL_BENCHMARK_NAME] = history["Historical Benchmark Return"]
    plot_cumulative_returns(
        returns_for_plot,
        output_dir / "latest_portfolio_vs_benchmark.png",
        f"Latest trailing returns vs {HISTORICAL_BENCHMARK_NAME}",
    )

    total_plot = combined_total_risk_contributors.copy()
    total_plot["Label"] = total_plot["Portfolio"] + ":" + total_plot["Ticker"].astype(str)
    proxy_plot = combined_proxy_te_contributors.copy()
    proxy_plot["Label"] = proxy_plot["Portfolio"] + ":" + proxy_plot["Ticker"].astype(str)
    factor_total_plot = combined_factor_total.copy()
    factor_total_plot["Label"] = factor_total_plot["Portfolio"] + ":" + factor_total_plot["Factor"].astype(str)
    factor_proxy_plot = combined_factor_proxy.copy()
    factor_proxy_plot["Label"] = factor_proxy_plot["Portfolio"] + ":" + factor_proxy_plot["Factor"].astype(str)

    plot_bar_chart(
        total_plot,
        "Label",
        "Total Variance Contribution Daily",
        "Latest top total risk contributors",
        output_dir / "latest_total_risk_contributors.png",
    )
    plot_bar_chart(
        proxy_plot,
        "Label",
        "Proxy TE Variance Contribution Daily",
        "Latest top proxy TE contributors",
        output_dir / "latest_te_contributors.png",
    )
    plot_bar_chart(
        proxy_plot,
        "Label",
        "Proxy TE Variance Contribution Daily",
        "Latest top proxy TE contributors",
        output_dir / "latest_tracking_error_contributors.png",
    )
    plot_bar_chart(
        factor_total_plot,
        "Label",
        "Total Variance Contribution Daily",
        "Latest factor total risk contribution",
        output_dir / "latest_factor_total_contribution.png",
        top_n=20,
    )
    plot_bar_chart(
        factor_proxy_plot,
        "Label",
        "Proxy Active Variance Contribution Daily",
        "Latest factor proxy active contribution",
        output_dir / "latest_factor_te_contribution.png",
        top_n=20,
    )
    plot_bar_chart(
        factor_proxy_plot,
        "Label",
        "Proxy Active Variance Contribution Daily",
        "Latest factor proxy active contribution",
        output_dir / "latest_factor_proxy_active_contribution.png",
        top_n=20,
    )


def _build_latest_summary_comparison(
    latest_artifacts: dict[str, dict[str, object]],
    optimization_status: pd.DataFrame,
) -> pd.DataFrame:
    status = optimization_status.set_index("Portfolio")
    rows = []
    metrics = [
        "Latest Rebalance Date",
        "Eligible Universe Count",
        "Selected Portfolio Count",
        "Historical Volatility",
        "Historical Tracking Error vs S&P500",
        "Benchmark Weight Source",
        "Benchmark Holdings Date",
        "Benchmark Local Coverage Ratio",
        "Predicted Total Risk",
        "Predicted Proxy Tracking Error",
        "Systematic Risk",
        "Idiosyncratic Risk",
        "Total Variance Annualized",
        "Systematic Variance Annualized",
        "Idiosyncratic Variance Annualized",
        "Factor Explained Ratio",
        "Idiosyncratic Ratio",
        "Proxy Active Systematic Risk",
        "Proxy Active Idiosyncratic Risk",
        "MCAR Reconciliation Error",
        "MCAR Mean Absolute Value",
        "MCAR Max Absolute Value",
        "Proxy Benchmark Idiosyncratic Variance Daily",
    ]
    for portfolio_name in PORTFOLIO_ORDER:
        summary = latest_artifacts[portfolio_name]["risk_summary"].set_index("Metric")["Value"]
        row = {"Portfolio": portfolio_name}
        for metric in metrics:
            row[metric] = summary.get(metric, np.nan)
        if portfolio_name in status.index:
            row["Optimization Success"] = bool(status.loc[portfolio_name, "Success"])
            row["Optimization Message"] = status.loc[portfolio_name, "Message"]
        rows.append(row)
    return pd.DataFrame(rows)


def _run_latest_from_context(
    context: Project2Context,
    output_dir: Path,
    top_pct: float,
    risk_window: int,
    cap: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dates = get_rebalance_dates(context.factor_data.monthly_signal)
    if not latest_dates:
        raise RuntimeError("No valid signal dates found for the latest Project 2 report.")
    latest_rebalance_date = latest_dates[-1]
    prepared = prepare_period(
        context=context,
        rebalance_date=latest_rebalance_date,
        next_rebalance_date=None,
        top_pct=top_pct,
        risk_window=risk_window,
    )
    if prepared is None:
        raise RuntimeError(
            f"Could not prepare the latest Project 2 report for {latest_rebalance_date.date().isoformat()}."
        )

    weights_by_portfolio, status_rows = _build_portfolio_weights(prepared, cap=cap)
    optimization_status = pd.DataFrame(status_rows)
    latest_artifacts: dict[str, dict[str, object]] = {}
    workbook_frames: dict[str, pd.DataFrame] = {
        "Optimization_Status": optimization_status,
    }

    for portfolio_name in PORTFOLIO_ORDER:
        artifacts = build_latest_risk_report(prepared, weights_by_portfolio[portfolio_name])
        latest_artifacts[portfolio_name] = artifacts
        portfolio_dir = output_dir / PORTFOLIO_SLUGS[portfolio_name]
        _export_latest_portfolio_report(portfolio_name, artifacts, portfolio_dir)

        prefix = PORTFOLIO_SLUGS[portfolio_name]
        workbook_frames[f"{prefix}_summary"] = artifacts["risk_summary"]
        workbook_frames[f"{prefix}_historical_var"] = artifacts["historical_var"]
        workbook_frames[f"{prefix}_normal_var"] = artifacts["normal_var"]
        workbook_frames[f"{prefix}_weights"] = artifacts["weights"]
        workbook_frames[f"{prefix}_factor_total"] = artifacts["factor_total_contribution"]
        workbook_frames[f"{prefix}_factor_proxy"] = artifacts["factor_proxy_active_contribution"]
        workbook_frames[f"{prefix}_top_total"] = artifacts["total_risk_contributors"]
        workbook_frames[f"{prefix}_top_proxy_te"] = artifacts["proxy_te_contributors"]

    latest_comparison = _build_latest_summary_comparison(latest_artifacts, optimization_status)
    latest_comparison.to_csv(output_dir / "latest_summary_comparison.csv", index=False)
    optimization_status.to_csv(output_dir / "optimization_status.csv", index=False)
    _export_latest_root_files(latest_artifacts, latest_comparison, output_dir)
    method_note_path = _write_methods_note(
        output_dir,
        "latest",
        prepared.rebalance_date,
        context.benchmark_weights,
    )
    workbook_frames = {"Latest_Comparison": latest_comparison, **workbook_frames}
    export_dataframe_bundle_excel(workbook_frames, output_dir / "project2_latest_report.xlsx")

    return {
        "prepared_period": prepared,
        "latest_artifacts": latest_artifacts,
        "latest_summary_comparison": latest_comparison,
        "optimization_status": optimization_status,
        "method_note_path": method_note_path,
        "output_dir": output_dir,
    }


def run_latest_project2_report(
    project1_dir: Path,
    output_dir: Path,
    factor_quantile: float = 0.2,
    top_pct: float = 0.05,
    risk_window: int = 252,
    cap: float = 0.1,
    risk_factor_model: str = "barra_proxy",
) -> dict[str, object]:
    context = prepare_project2_context(
        project1_dir,
        factor_quantile=factor_quantile,
        risk_factor_model=risk_factor_model,
    )
    return _run_latest_from_context(
        context=context,
        output_dir=output_dir,
        top_pct=top_pct,
        risk_window=risk_window,
        cap=cap,
    )


def run_full_project2_backtest(
    project1_dir: Path,
    output_dir: Path,
    factor_quantile: float = 0.2,
    top_pct: float = 0.05,
    risk_window: int = 252,
    cap: float = 0.1,
    risk_factor_model: str = "barra_proxy",
) -> dict[str, object]:
    context = prepare_project2_context(
        project1_dir,
        factor_quantile=factor_quantile,
        risk_factor_model=risk_factor_model,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = context.market_data.prices.columns
    previous_weights = {
        portfolio_name: pd.Series(0.0, index=all_tickers, dtype=float)
        for portfolio_name in PORTFOLIO_ORDER
    }
    portfolio_return_parts: dict[str, list[pd.Series]] = {name: [] for name in PORTFOLIO_ORDER}
    turnovers: dict[str, list[float]] = {name: [] for name in PORTFOLIO_ORDER}
    risk_snapshot_rows: list[dict[str, object]] = []
    factor_total_rows: list[dict[str, object]] = []
    factor_proxy_rows: list[dict[str, object]] = []
    stock_total_rows: list[dict[str, object]] = []
    stock_proxy_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    optimization_rows: list[dict[str, object]] = []
    valid_periods = 0

    rebalance_dates = get_rebalance_dates(context.factor_data.monthly_signal)
    for idx, rebalance_date in enumerate(rebalance_dates):
        next_rebalance_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None
        prepared = prepare_period(
            context=context,
            rebalance_date=rebalance_date,
            next_rebalance_date=next_rebalance_date,
            top_pct=top_pct,
            risk_window=risk_window,
        )
        if prepared is None:
            continue

        valid_periods += 1
        weights_by_portfolio, status_rows = _build_portfolio_weights(prepared, cap=cap)
        optimization_rows.extend(status_rows)

        for portfolio_name in PORTFOLIO_ORDER:
            weights = weights_by_portfolio[portfolio_name]
            full_weights = full_weight_series(all_tickers, weights)
            turnover = compute_turnover(full_weights, previous_weights[portfolio_name])
            previous_weights[portfolio_name] = full_weights
            turnovers[portfolio_name].append(turnover)

            portfolio_returns = compute_portfolio_daily_returns(prepared.holding_period_returns, weights)
            if not portfolio_returns.empty:
                portfolio_return_parts[portfolio_name].append(portfolio_returns.rename(portfolio_name))

            total_risk = portfolio_total_risk(weights, prepared.investable_fit)
            tracking_error = portfolio_tracking_error(weights, prepared.investable_fit)
            snapshot, factor_total, factor_proxy, stock_total, stock_proxy = build_risk_snapshot_rows(
                date=prepared.rebalance_date,
                portfolio_name=portfolio_name,
                total_risk=total_risk,
                tracking_error=tracking_error,
                turnover=turnover,
            )
            risk_snapshot_rows.extend(snapshot)
            factor_total_rows.extend(factor_total)
            factor_proxy_rows.extend(factor_proxy)
            stock_total_rows.extend(stock_total)
            stock_proxy_rows.extend(stock_proxy)

            for ticker, weight in weights.sort_values(ascending=False).items():
                weight_rows.append(
                    {
                        "Date": prepared.rebalance_date,
                        "Portfolio": portfolio_name,
                        "Ticker": ticker,
                        "Weight": weight,
                    }
                )

    if valid_periods == 0:
        raise RuntimeError("No valid rebalance periods found for the full Project 2 backtest.")

    portfolio_returns = pd.DataFrame(
        {
            name: pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
            for name, parts in portfolio_return_parts.items()
        }
    )
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="last")]
    historical_benchmark_returns = (
        context.market_data.benchmark["ben_daily_return"].reindex(portfolio_returns.index).fillna(0.0)
    )
    daily_returns_export = portfolio_returns.copy()
    daily_returns_export[HISTORICAL_BENCHMARK_NAME] = historical_benchmark_returns
    daily_returns_export = daily_returns_export.reset_index().rename(columns={"index": "Date"})

    summary_rows = []
    for portfolio_name in PORTFOLIO_ORDER:
        summary_rows.append(
            {
                "Portfolio": portfolio_name,
                **performance_summary(
                    portfolio_returns[portfolio_name],
                    historical_benchmark_returns,
                    turnovers[portfolio_name],
                ),
            }
        )
    summary_stats = pd.DataFrame(summary_rows)

    risk_snapshots = pd.DataFrame(risk_snapshot_rows)
    factor_total_contributions = pd.DataFrame(factor_total_rows)
    factor_proxy_active_contributions = pd.DataFrame(factor_proxy_rows)
    stock_total_contributions = pd.DataFrame(stock_total_rows)
    stock_proxy_te_contributions = pd.DataFrame(stock_proxy_rows)
    weights_by_rebalance = pd.DataFrame(weight_rows)
    optimization_status = pd.DataFrame(optimization_rows)

    daily_returns_export.to_csv(output_dir / "portfolio_daily_returns.csv", index=False)
    summary_stats.to_csv(output_dir / "summary_stats.csv", index=False)
    risk_snapshots.to_csv(output_dir / "risk_snapshots.csv", index=False)
    factor_total_contributions.to_csv(output_dir / "factor_total_contributions.csv", index=False)
    factor_proxy_active_contributions.to_csv(output_dir / "factor_proxy_active_contributions.csv", index=False)
    factor_proxy_active_contributions.to_csv(output_dir / "factor_te_contributions.csv", index=False)
    stock_total_contributions.to_csv(output_dir / "stock_total_risk_contributions.csv", index=False)
    stock_proxy_te_contributions.to_csv(output_dir / "stock_proxy_te_contributions.csv", index=False)
    weights_by_rebalance.to_csv(output_dir / "weights_by_rebalance.csv", index=False)
    optimization_status.to_csv(output_dir / "optimization_status.csv", index=False)

    plot_cumulative_returns(
        daily_returns_export.set_index("Date")[[*PORTFOLIO_ORDER, HISTORICAL_BENCHMARK_NAME]],
        output_dir / "portfolio_cumulative_returns.png",
        "Project 2 portfolio cumulative returns",
    )
    plot_risk_timeseries(
        risk_snapshots,
        "Predicted Total Risk",
        "Predicted total risk by rebalance date",
        output_dir / "predicted_total_risk.png",
    )
    plot_risk_timeseries(
        risk_snapshots,
        "Predicted Proxy Tracking Error",
        "Predicted proxy tracking error by rebalance date",
        output_dir / "predicted_proxy_tracking_error.png",
    )
    plot_risk_timeseries(
        risk_snapshots,
        "Predicted Proxy Tracking Error",
        "Predicted proxy tracking error by rebalance date",
        output_dir / "predicted_tracking_error.png",
    )

    method_note_path = _write_methods_note(
        output_dir,
        "full backtest",
        None,
        context.benchmark_weights,
    )
    latest_result = _run_latest_from_context(
        context=context,
        output_dir=output_dir / "latest_reports",
        top_pct=top_pct,
        risk_window=risk_window,
        cap=cap,
    )
    latest_result["latest_summary_comparison"].to_csv(output_dir / "latest_summary_comparison.csv", index=False)
    _export_latest_root_files(
        latest_result["latest_artifacts"],
        latest_result["latest_summary_comparison"],
        output_dir,
    )

    workbook_frames = {
        "Summary_Stats": summary_stats,
        "Daily_Returns": daily_returns_export,
        "Risk_Snapshots": risk_snapshots,
        "Factor_Total": factor_total_contributions,
        "Factor_Proxy_Active": factor_proxy_active_contributions,
        "Stock_Total": stock_total_contributions,
        "Stock_Proxy_TE": stock_proxy_te_contributions,
        "Weights": weights_by_rebalance,
        "Optimization_Status": optimization_status,
        "Latest_Comparison": latest_result["latest_summary_comparison"],
    }
    export_dataframe_bundle_excel(workbook_frames, output_dir / "project2_full_outputs.xlsx")

    return {
        "summary_stats": summary_stats,
        "portfolio_daily_returns": daily_returns_export,
        "risk_snapshots": risk_snapshots,
        "factor_total_contributions": factor_total_contributions,
        "factor_proxy_active_contributions": factor_proxy_active_contributions,
        "stock_total_risk_contributions": stock_total_contributions,
        "stock_proxy_te_contributions": stock_proxy_te_contributions,
        "weights_by_rebalance": weights_by_rebalance,
        "optimization_status": optimization_status,
        "latest_result": latest_result,
        "method_note_path": method_note_path,
        "output_dir": output_dir,
    }
