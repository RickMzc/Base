from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    raw: pd.DataFrame
    benchmark: pd.DataFrame
    prices: pd.DataFrame
    opens: pd.DataFrame
    highs: pd.DataFrame
    lows: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame


def load_market_data(project1_dir: Path) -> MarketData:
    csv_path = project1_dir / "SP500_Full_OHLCV_Final.csv"
    benchmark_path = project1_dir / "SP500.xlsx"

    raw = pd.read_csv(
        csv_path,
        parse_dates=["Date"],
        dtype={
            "Ticker": str,
            "Close": float,
            "High": float,
            "Low": float,
            "Open": float,
            "Volume": float,
        },
    )
    raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    raw["Suspended"] = np.where(raw["Volume"] == 0, 1, 0)
    raw = raw.dropna(subset=["Close", "High", "Low", "Open", "Volume"]).copy()
    raw["Return"] = raw.groupby("Ticker")["Close"].pct_change()

    benchmark = pd.read_excel(benchmark_path, parse_dates=["Date"])
    benchmark = benchmark.sort_values("Date").set_index("Date")
    benchmark["ben_daily_return"] = benchmark["Benchmark"].pct_change()
    benchmark["ben_daily_return"] = benchmark["ben_daily_return"].fillna(0.0)

    prices = raw.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    opens = raw.pivot(index="Date", columns="Ticker", values="Open").sort_index()
    highs = raw.pivot(index="Date", columns="Ticker", values="High").sort_index()
    lows = raw.pivot(index="Date", columns="Ticker", values="Low").sort_index()
    volume = raw.pivot(index="Date", columns="Ticker", values="Volume").sort_index()
    returns = prices.pct_change(fill_method=None)

    common_dates = prices.index.intersection(benchmark.index)
    prices = prices.loc[common_dates]
    opens = opens.loc[common_dates]
    highs = highs.loc[common_dates]
    lows = lows.loc[common_dates]
    volume = volume.loc[common_dates]
    returns = returns.loc[common_dates]
    benchmark = benchmark.loc[common_dates]

    return MarketData(
        raw=raw,
        benchmark=benchmark,
        prices=prices,
        opens=opens,
        highs=highs,
        lows=lows,
        volume=volume,
        returns=returns,
    )


def full_weight_series(index: pd.Index, weights: pd.Series | None = None) -> pd.Series:
    series = pd.Series(0.0, index=index, dtype=float)
    if weights is None:
        return series
    aligned = weights.reindex(index).fillna(0.0)
    series.loc[:] = aligned.values
    return series
