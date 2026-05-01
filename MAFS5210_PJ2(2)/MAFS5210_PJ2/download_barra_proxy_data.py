from __future__ import annotations

import sys
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


PROJECT1_DIR = ROOT / "Project1"
PROJECT2_DATA_DIR = ROOT / "Project2_data"
OUTPUT_PATH = PROJECT2_DATA_DIR / "barra_proxy_metadata.csv"


def _to_yahoo_ticker(ticker: str) -> str:
    return ticker.replace(".", "-")


def _safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_project_tickers() -> list[str]:
    csv_path = PROJECT1_DIR / "SP500_Full_OHLCV_Final.csv"
    raw = pd.read_csv(csv_path, usecols=["Ticker"], dtype={"Ticker": str})
    return sorted(raw["Ticker"].dropna().unique())


def _load_wikipedia_sp500() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    except Exception:
        return pd.DataFrame()
    if not tables:
        return pd.DataFrame()
    frame = tables[0].copy()
    if "Symbol" not in frame.columns:
        return pd.DataFrame()
    frame["Ticker"] = frame["Symbol"].astype(str).str.replace(".", "-", regex=False)
    columns = ["Ticker"]
    rename_map = {}
    if "GICS Sector" in frame.columns:
        columns.append("GICS Sector")
        rename_map["GICS Sector"] = "WikiSector"
    if "GICS Sub-Industry" in frame.columns:
        columns.append("GICS Sub-Industry")
        rename_map["GICS Sub-Industry"] = "WikiIndustry"
    if "CIK" in frame.columns:
        columns.append("CIK")
        rename_map["CIK"] = "CIK"
    return frame[columns].rename(columns=rename_map).drop_duplicates("Ticker")


def _fetch_yfinance_info(ticker: str) -> dict[str, object]:
    import yfinance as yf

    yahoo_ticker = _to_yahoo_ticker(ticker)
    info: dict[str, object] = {}
    try:
        info = yf.Ticker(yahoo_ticker).get_info()
    except Exception:
        try:
            info = yf.Ticker(yahoo_ticker).info
        except Exception:
            info = {}

    current_price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    market_cap = _safe_float(info.get("marketCap"))
    shares_outstanding = _safe_float(info.get("sharesOutstanding"))
    if np.isnan(shares_outstanding) and not np.isnan(market_cap) and current_price > 0:
        shares_outstanding = market_cap / current_price

    price_to_book = _safe_float(info.get("priceToBook"))
    book_to_price = 1.0 / price_to_book if price_to_book > 0 else np.nan

    trailing_eps = _safe_float(info.get("trailingEps"))
    trailing_pe = _safe_float(info.get("trailingPE"))
    if current_price > 0 and not np.isnan(trailing_eps):
        earnings_yield = trailing_eps / current_price
    elif trailing_pe > 0:
        earnings_yield = 1.0 / trailing_pe
    else:
        earnings_yield = np.nan

    dividend_yield = _safe_float(info.get("dividendYield"))
    if dividend_yield > 1.0:
        dividend_yield = dividend_yield / 100.0

    debt_to_equity = _safe_float(info.get("debtToEquity"))
    leverage = debt_to_equity / 100.0 if not np.isnan(debt_to_equity) else np.nan

    return {
        "Ticker": ticker,
        "YahooTicker": yahoo_ticker,
        "YFinanceSector": info.get("sector"),
        "YFinanceIndustry": info.get("industry"),
        "MarketCap": market_cap,
        "SharesOutstanding": shares_outstanding,
        "BookToPrice": book_to_price,
        "EarningsYield": earnings_yield,
        "Leverage": leverage,
        "DividendYield": dividend_yield,
        "RevenueGrowth": _safe_float(info.get("revenueGrowth")),
        "DataSource": "yfinance_info",
    }


def build_barra_proxy_metadata(sleep_seconds: float = 0.05) -> pd.DataFrame:
    tickers = _load_project_tickers()
    wikipedia = _load_wikipedia_sp500()
    rows = []
    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index:03d}/{len(tickers):03d}] fetching {ticker}")
        rows.append(_fetch_yfinance_info(ticker))
        if sleep_seconds > 0:
            sleep(sleep_seconds)

    metadata = pd.DataFrame(rows)
    if not wikipedia.empty:
        metadata = metadata.merge(wikipedia, on="Ticker", how="left")
    else:
        metadata["WikiSector"] = np.nan
        metadata["WikiIndustry"] = np.nan
        metadata["CIK"] = np.nan

    metadata["Sector"] = metadata["YFinanceSector"].fillna(metadata["WikiSector"]).fillna("Unknown")
    metadata["Industry"] = metadata["YFinanceIndustry"].fillna(metadata["WikiIndustry"]).fillna("Unknown")
    metadata["DataSource"] = metadata["DataSource"] + "+wikipedia_sp500"
    return metadata


def main() -> None:
    PROJECT2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = build_barra_proxy_metadata()
    metadata.to_csv(OUTPUT_PATH, index=False)
    coverage = metadata[["Sector", "SharesOutstanding", "BookToPrice", "EarningsYield"]].notna().mean()
    print(f"Saved Barra-style proxy metadata to: {OUTPUT_PATH}")
    print("Coverage:")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
