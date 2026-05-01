from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketData


@dataclass
class FactorData:
    factor_scores: dict[str, pd.DataFrame]
    composite_signal: pd.DataFrame
    monthly_signal: pd.DataFrame
    factor_returns: pd.DataFrame


def standardize_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    row_mean = frame.mean(axis=1)
    row_std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(row_mean, axis=0).div(row_std, axis=0)


def compute_factor_data(
    market_data: MarketData,
    top_bottom_quantile: float = 0.2,
    risk_factor_model: str = "barra_proxy",
    project2_data_dir: Path | None = None,
) -> FactorData:
    prices = market_data.prices
    opens = market_data.opens
    volume = market_data.volume
    daily_rets = market_data.returns

    raw_gap = (opens / prices.shift(1) - 1.0).rolling(252).sum()
    raw_rel = daily_rets.sub(daily_rets.mean(axis=1), axis=0).rolling(252).sum()
    raw_vol = volume.rolling(20).mean() / volume.rolling(252).mean()

    factor_scores = {
        "Market": pd.DataFrame(index=daily_rets.index, columns=[]),
        "Mom_Gap": standardize_cross_section(raw_gap),
        "Rel_Strength": standardize_cross_section(raw_rel),
        "Vol_Trend": standardize_cross_section(raw_vol),
    }

    composite_signal = (
        factor_scores["Mom_Gap"].fillna(0.0)
        + factor_scores["Rel_Strength"].fillna(0.0)
        + factor_scores["Vol_Trend"].fillna(0.0)
    ) / 3.0

    nonzero_mask = (composite_signal != 0.0).any(axis=1)
    first_valid = nonzero_mask.idxmax()
    composite_signal = composite_signal.loc[first_valid:]
    monthly_signal = composite_signal.resample("ME").last()
    # Drop a trailing partial month. Resampling labels 2026-03-16 as
    # 2026-03-31 if March is incomplete, which would create a fake rebalance.
    # Keep normal month ends where the calendar month-end is a weekend/holiday.
    latest_observed_date = pd.Timestamp(composite_signal.index.max())
    if not monthly_signal.empty:
        latest_label = pd.Timestamp(monthly_signal.index.max())
        if latest_label > latest_observed_date and (latest_label - latest_observed_date).days > 3:
            monthly_signal = monthly_signal.drop(index=latest_label)

    if risk_factor_model == "project1":
        factor_returns = build_factor_mimicking_returns(
            factor_scores={
                key: value
                for key, value in factor_scores.items()
                if key != "Market"
            },
            stock_returns=daily_rets,
            benchmark_returns=market_data.benchmark["ben_daily_return"],
            quantile=top_bottom_quantile,
        )
    elif risk_factor_model == "barra_proxy":
        if project2_data_dir is None:
            raise ValueError("project2_data_dir is required for the Barra-style proxy risk model.")
        from .barra_proxy import build_barra_style_factor_returns

        factor_returns = build_barra_style_factor_returns(
            market_data=market_data,
            project2_data_dir=project2_data_dir,
            top_bottom_quantile=top_bottom_quantile,
        )
    else:
        raise ValueError(f"Unsupported risk_factor_model: {risk_factor_model}")

    return FactorData(
        factor_scores=factor_scores,
        composite_signal=composite_signal,
        monthly_signal=monthly_signal,
        factor_returns=factor_returns,
    )


def build_factor_mimicking_returns(
    factor_scores: dict[str, pd.DataFrame],
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    quantile: float = 0.2,
    min_names: int = 20,
) -> pd.DataFrame:
    lagged_scores = {name: scores.shift(1) for name, scores in factor_scores.items()}
    factor_return_map: dict[str, pd.Series] = {}

    for name, scores in lagged_scores.items():
        values: list[float] = []
        dates: list[pd.Timestamp] = []
        for date in stock_returns.index:
            if date not in scores.index:
                continue
            signal = scores.loc[date]
            next_ret = stock_returns.loc[date]
            eligible = signal.notna() & next_ret.notna()
            eligible_names = signal.index[eligible]
            if len(eligible_names) < min_names:
                values.append(np.nan)
                dates.append(date)
                continue

            score_slice = signal.loc[eligible_names].sort_values()
            bucket = max(min_names // 2, int(len(score_slice) * quantile))
            bucket = min(bucket, len(score_slice) // 2)
            if bucket == 0:
                values.append(np.nan)
                dates.append(date)
                continue

            short_names = score_slice.index[:bucket]
            long_names = score_slice.index[-bucket:]
            long_ret = next_ret.loc[long_names].mean()
            short_ret = next_ret.loc[short_names].mean()
            values.append(long_ret - short_ret)
            dates.append(date)

        factor_return_map[name] = pd.Series(values, index=pd.Index(dates, name="Date"))

    factor_returns = pd.DataFrame(factor_return_map).sort_index()
    factor_returns.insert(0, "Market", benchmark_returns.reindex(factor_returns.index))
    factor_returns = factor_returns.dropna()
    return factor_returns
