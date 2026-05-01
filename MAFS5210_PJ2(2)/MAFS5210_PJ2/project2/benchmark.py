from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BENCHMARK_SOURCE_NAME = "IVV ETF holdings proxy"
BENCHMARK_TICKER_ALIASES = {
    "BRKB": "BRK-B",
    "BFB": "BF-B",
    "FB": "META",
    "ANTM": "ELV",
    "CBS": "PARA",
}


@dataclass
class BenchmarkWeightsData:
    raw_weights: pd.DataFrame
    summary: pd.DataFrame
    by_rebalance: dict[pd.Timestamp, pd.Series]
    holdings_date_by_rebalance: dict[pd.Timestamp, pd.Timestamp]
    coverage_ratio_by_rebalance: pd.Series
    source_name: str = BENCHMARK_SOURCE_NAME


def benchmark_data_dir(project1_dir: Path) -> Path:
    return project1_dir.parent / "Project2_data"


def benchmark_weights_path(project1_dir: Path) -> Path:
    return benchmark_data_dir(project1_dir) / "ivv_benchmark_weights.csv"


def benchmark_summary_path(project1_dir: Path) -> Path:
    return benchmark_data_dir(project1_dir) / "ivv_benchmark_rebalance_summary.csv"


def normalize_benchmark_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    return BENCHMARK_TICKER_ALIASES.get(clean, clean)


def load_benchmark_weights(
    project1_dir: Path,
    required: bool = True,
) -> BenchmarkWeightsData | None:
    weights_path = benchmark_weights_path(project1_dir)
    summary_path = benchmark_summary_path(project1_dir)
    if not weights_path.exists() or not summary_path.exists():
        if required:
            raise FileNotFoundError(
                "Missing cached IVV benchmark weights. "
                "Run `python download_ivv_benchmark_weights.py` first and include the "
                "`Project2_data` files when reproducing Project 2 results."
            )
        return None

    raw_weights = pd.read_csv(
        weights_path,
        parse_dates=["RebalanceDate", "RequestedDate", "HoldingsDate"],
    )
    summary = pd.read_csv(
        summary_path,
        parse_dates=["RebalanceDate", "RequestedDate", "HoldingsDate"],
    )

    usable = raw_weights[raw_weights["UseInRiskModel"] == 1].copy()
    by_rebalance = {
        pd.Timestamp(date): group.set_index("Ticker")["LocalNormalizedWeight"].astype(float)
        for date, group in usable.groupby("RebalanceDate")
    }
    holdings_date_by_rebalance = {
        pd.Timestamp(row["RebalanceDate"]): pd.Timestamp(row["HoldingsDate"])
        for _, row in summary.iterrows()
    }
    coverage_ratio = (
        summary.set_index("RebalanceDate")["LocalWeightCoverageRatio"].astype(float).rename("LocalWeightCoverageRatio")
    )

    return BenchmarkWeightsData(
        raw_weights=raw_weights,
        summary=summary,
        by_rebalance=by_rebalance,
        holdings_date_by_rebalance=holdings_date_by_rebalance,
        coverage_ratio_by_rebalance=coverage_ratio,
    )
