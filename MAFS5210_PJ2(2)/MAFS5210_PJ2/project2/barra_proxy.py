from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketData


BARRA_PROXY_METADATA_FILE = "barra_proxy_metadata.csv"


def _clean_factor_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "Unknown"


def load_barra_proxy_metadata(project2_data_dir: Path) -> pd.DataFrame:
    metadata_path = project2_data_dir / BARRA_PROXY_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing Barra-style proxy metadata: {metadata_path}. "
            "Run `python download_barra_proxy_data.py` first."
        )

    metadata = pd.read_csv(metadata_path, dtype={"Ticker": str})
    if "Ticker" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a Ticker column.")
    metadata["Ticker"] = metadata["Ticker"].astype(str)
    return metadata.drop_duplicates("Ticker").set_index("Ticker")


def _standardize_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    row_mean = frame.mean(axis=1)
    row_std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(row_mean, axis=0).div(row_std, axis=0)


def _cross_sectional_residual(
    target: pd.DataFrame,
    regressors: list[pd.DataFrame],
    include_intercept: bool = True,
    min_names: int = 20,
) -> pd.DataFrame:
    residuals = pd.DataFrame(np.nan, index=target.index, columns=target.columns, dtype=float)

    for date in target.index:
        y = pd.to_numeric(target.loc[date], errors="coerce")
        x_parts = []
        for idx, reg in enumerate(regressors):
            x_parts.append(pd.to_numeric(reg.loc[date], errors="coerce").rename(f"x{idx}"))
        x = pd.concat(x_parts, axis=1)

        valid = y.notna()
        for column in x.columns:
            valid &= x[column].notna()
        valid_names = y.index[valid]

        min_required = max(min_names, x.shape[1] + 2)
        if len(valid_names) < min_required:
            continue

        y_values = y.loc[valid_names].to_numpy(dtype=float)
        x_values = x.loc[valid_names].to_numpy(dtype=float)
        if include_intercept:
            x_values = np.column_stack([np.ones(len(valid_names), dtype=float), x_values])

        coeff = np.linalg.pinv(x_values) @ y_values
        residual_values = y_values - x_values @ coeff
        residuals.loc[date, valid_names] = residual_values

    return residuals


def _constant_exposure_frame(
    values: pd.Series,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    aligned = values.reindex(columns).astype(float)
    return pd.DataFrame(
        np.tile(aligned.to_numpy(dtype=float), (len(index), 1)),
        index=index,
        columns=columns,
    )


def _impute_numeric_exposures(metadata: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    numeric_columns = [
        "SharesOutstanding",
        "BookToPrice",
        "EarningsYield",
        "Leverage",
        "DividendYield",
    ]
    exposures = pd.DataFrame(index=columns)
    for column in numeric_columns:
        if column in metadata.columns:
            series = pd.to_numeric(metadata[column], errors="coerce").reindex(columns)
        else:
            series = pd.Series(np.nan, index=columns, dtype=float)
        median_value = float(series.median()) if series.notna().any() else 0.0
        exposures[column] = series.fillna(median_value).astype(float)
    return exposures


def _residual_volatility(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    benchmark = benchmark_returns.reindex(stock_returns.index).fillna(0.0)
    benchmark_variance = benchmark.rolling(window).var().replace(0.0, np.nan)
    stock_variance = stock_returns.rolling(window).var()
    stock_market_covariance = stock_returns.rolling(window).cov(benchmark)
    residual_variance = stock_variance - stock_market_covariance.pow(2).div(benchmark_variance, axis=0)
    return np.sqrt(residual_variance.clip(lower=0.0))


def _long_short_factor_returns(
    factor_scores: dict[str, pd.DataFrame],
    stock_returns: pd.DataFrame,
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
            next_return = stock_returns.loc[date]
            eligible = signal.notna() & next_return.notna()
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
            values.append(float(next_return.loc[long_names].mean() - next_return.loc[short_names].mean()))
            dates.append(date)

        factor_return_map[name] = pd.Series(values, index=pd.Index(dates, name="Date"))

    return pd.DataFrame(factor_return_map).sort_index()


def _sector_factor_returns(
    metadata: pd.DataFrame,
    stock_returns: pd.DataFrame,
    min_sector_names: int = 5,
) -> pd.DataFrame:
    if "Sector" not in metadata.columns:
        return pd.DataFrame(index=stock_returns.index)

    sector_map = metadata["Sector"].reindex(stock_returns.columns).fillna("Unknown")
    universe_return = stock_returns.mean(axis=1)
    factor_returns: dict[str, pd.Series] = {}
    for sector_name, sector_tickers in sector_map.groupby(sector_map).groups.items():
        tickers = pd.Index(sector_tickers).intersection(stock_returns.columns)
        if len(tickers) < min_sector_names or str(sector_name).lower() == "unknown":
            continue
        factor_name = f"Sector_{_clean_factor_name(sector_name)}"
        factor_returns[factor_name] = stock_returns.loc[:, tickers].mean(axis=1) - universe_return
    return pd.DataFrame(factor_returns).sort_index()


def build_barra_style_factor_returns(
    market_data: MarketData,
    project2_data_dir: Path,
    top_bottom_quantile: float = 0.2,
) -> pd.DataFrame:
    metadata = load_barra_proxy_metadata(project2_data_dir)
    prices = market_data.prices
    volume = market_data.volume
    stock_returns = market_data.returns
    benchmark_returns = market_data.benchmark["ben_daily_return"]
    numeric_exposures = _impute_numeric_exposures(metadata, prices.columns)

    shares = numeric_exposures["SharesOutstanding"].replace(0.0, np.nan)
    market_cap = prices.mul(shares, axis=1)
    size = np.log(market_cap.replace(0.0, np.nan))
    standardized_size = _standardize_cross_section(size)
    nonlinear_size_raw = standardized_size.pow(3)
    nonlinear_size_residual = _cross_sectional_residual(
        target=nonlinear_size_raw,
        regressors=[standardized_size],
        include_intercept=True,
    )

    momentum = prices.shift(21) / prices.shift(252) - 1.0
    residual_vol = _residual_volatility(stock_returns, benchmark_returns)
    dollar_volume = (prices * volume).rolling(20).mean()
    liquidity = np.log(dollar_volume.replace(0.0, np.nan))

    constant_frames = {
        "Book_to_Price": _constant_exposure_frame(numeric_exposures["BookToPrice"], prices.index, prices.columns),
        "Earnings_Yield": _constant_exposure_frame(numeric_exposures["EarningsYield"], prices.index, prices.columns),
        "Leverage": _constant_exposure_frame(numeric_exposures["Leverage"], prices.index, prices.columns),
        "Dividend_Yield": _constant_exposure_frame(numeric_exposures["DividendYield"], prices.index, prices.columns),
    }

    style_scores = {
        "Size": standardized_size,
        "Nonlinear_Size": _standardize_cross_section(nonlinear_size_residual),
        "Momentum": _standardize_cross_section(momentum),
        "Residual_Volatility": _standardize_cross_section(residual_vol),
        "Liquidity": _standardize_cross_section(liquidity),
        **{name: _standardize_cross_section(frame) for name, frame in constant_frames.items()},
    }

    style_returns = _long_short_factor_returns(
        style_scores,
        stock_returns=stock_returns,
        quantile=top_bottom_quantile,
    )
    sector_returns = _sector_factor_returns(metadata, stock_returns)

    factor_returns = pd.concat(
        [
            benchmark_returns.rename("Market"),
            sector_returns,
            style_returns,
        ],
        axis=1,
        join="inner",
    )
    factor_returns = factor_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if factor_returns.empty:
        raise ValueError("Barra-style proxy factor return matrix is empty after cleaning.")
    return factor_returns
