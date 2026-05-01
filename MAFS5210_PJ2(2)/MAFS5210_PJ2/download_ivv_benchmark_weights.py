from __future__ import annotations

import csv
import io
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from project2.backtest import get_rebalance_dates, prepare_project2_context
from project2.benchmark import (
    BENCHMARK_SOURCE_NAME,
    benchmark_data_dir,
    benchmark_summary_path,
    benchmark_weights_path,
    normalize_benchmark_ticker,
)


IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund&asOfDate={as_of_date}"
)


def trading_date_on_or_before(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    eligible = index[index <= date]
    if eligible.empty:
        raise ValueError(f"No trading date on or before {date}.")
    return pd.Timestamp(eligible.max())


def parse_ishares_holdings_csv(text: str) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    clean_text = text.replace("\ufeff", "")
    lines = clean_text.splitlines()
    if len(lines) < 10:
        raise ValueError("Unexpected iShares holdings response.")

    holdings_date = None
    as_of_line = lines[1]
    as_of_value = next(csv.reader([as_of_line.split(",", 1)[1]]))[0]
    if as_of_value and as_of_value != "-":
        holdings_date = pd.to_datetime(as_of_value)

    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Ticker,"))
    header = next(csv.reader([lines[header_idx]]))
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        if not line or line == "\xa0":
            break
        if line.startswith('"The content contained herein is owned or licensed by BlackRock'):
            break
        rows.append(next(csv.reader([line])))

    frame = pd.DataFrame(rows, columns=header)
    numeric_columns = ["Market Value", "Weight (%)", "Notional Value", "Quantity", "Price", "FX Rate"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ""), errors="coerce")
    return holdings_date, frame


def download_holdings_on_or_before(
    session: requests.Session,
    date: pd.Timestamp,
    max_lookback_days: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    for offset in range(max_lookback_days + 1):
        requested_date = pd.Timestamp(date - timedelta(days=offset))
        url = IVV_HOLDINGS_URL.format(as_of_date=requested_date.strftime("%Y%m%d"))
        response = session.get(url, timeout=30)
        response.raise_for_status()
        holdings_date, frame = parse_ishares_holdings_csv(response.text)
        if holdings_date is None or frame.empty:
            continue
        return requested_date, holdings_date, frame
    raise RuntimeError(f"Could not find valid IVV holdings on or before {date.date().isoformat()}.")


def main() -> None:
    project1_dir = ROOT / "Project1"
    context = prepare_project2_context(
        project1_dir=project1_dir,
        require_benchmark_weights=False,
    )
    rebalance_dates = get_rebalance_dates(context.factor_data.monthly_signal)
    local_universe = set(context.market_data.prices.columns)
    output_dir = benchmark_data_dir(project1_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    weight_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for rebalance_date in rebalance_dates:
        requested_anchor = trading_date_on_or_before(context.market_data.returns.index, rebalance_date)
        requested_date, holdings_date, raw_frame = download_holdings_on_or_before(session, requested_anchor)
        equities = raw_frame[raw_frame["Asset Class"] == "Equity"].copy()
        equities = equities[equities["Ticker"].astype(str).str.strip() != "-"].copy()
        equities["OriginalTicker"] = equities["Ticker"].astype(str).str.strip().str.upper()
        equities["Ticker"] = equities["OriginalTicker"].map(normalize_benchmark_ticker)
        equities["RawWeight"] = equities["Weight (%)"] / 100.0
        equities["UseInRiskModel"] = equities["Ticker"].isin(local_universe).astype(int)

        matched = equities[equities["UseInRiskModel"] == 1].copy()
        matched_weight = float(matched["RawWeight"].sum())
        if matched_weight <= 0.0:
            raise RuntimeError(f"No overlapping IVV holdings and local universe on {rebalance_date.date().isoformat()}.")
        equities["LocalNormalizedWeight"] = 0.0
        equities.loc[matched.index, "LocalNormalizedWeight"] = matched["RawWeight"] / matched_weight

        summary_rows.append(
            {
                "RebalanceDate": rebalance_date,
                "RequestedDate": requested_date,
                "HoldingsDate": holdings_date,
                "Source": BENCHMARK_SOURCE_NAME,
                "EquityCount": int(len(equities)),
                "MatchedTickerCount": int(len(matched)),
                "RawMatchedWeight": matched_weight,
                "LocalWeightCoverageRatio": matched_weight / float(equities["RawWeight"].sum()),
            }
        )

        for _, row in equities.iterrows():
            weight_rows.append(
                {
                    "RebalanceDate": rebalance_date,
                    "RequestedDate": requested_date,
                    "HoldingsDate": holdings_date,
                    "Source": BENCHMARK_SOURCE_NAME,
                    "OriginalTicker": row["OriginalTicker"],
                    "Ticker": row["Ticker"],
                    "Name": row["Name"],
                    "Sector": row["Sector"],
                    "RawWeight": row["RawWeight"],
                    "LocalNormalizedWeight": row["LocalNormalizedWeight"],
                    "UseInRiskModel": int(row["UseInRiskModel"]),
                }
            )

        print(
            f"{rebalance_date.date().isoformat()} <- {holdings_date.date().isoformat()} "
            f"coverage={matched_weight:.4%} matched={len(matched)}/{len(equities)}"
        )

    weights_frame = pd.DataFrame(weight_rows)
    summary_frame = pd.DataFrame(summary_rows)
    weights_frame.to_csv(benchmark_weights_path(project1_dir), index=False)
    summary_frame.to_csv(benchmark_summary_path(project1_dir), index=False)

    print(f"Saved weights to: {benchmark_weights_path(project1_dir)}")
    print(f"Saved summary to: {benchmark_summary_path(project1_dir)}")


if __name__ == "__main__":
    main()
