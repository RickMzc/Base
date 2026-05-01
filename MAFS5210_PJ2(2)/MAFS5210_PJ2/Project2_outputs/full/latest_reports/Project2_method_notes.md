# Project 2 Method Notes

## Scope
- Output mode: latest.
- Latest risk snapshot date: 2026-02-28.
- Historical performance and historical tracking error are measured against the S&P500 index return in `Project1/SP500.xlsx`.
- Rebalance dates use completed month-end signals only. A trailing partial month is excluded to avoid labeling an incomplete signal as a month-end rebalance.
- Historical performance uses completed holding-period returns only.
- Predicted tracking error is labelled as proxy tracking error because benchmark constituent weights come from IVV ETF holdings, not the licensed S&P 500 index file.

## Proxy Benchmark For Risk Model
- At each rebalance date, benchmark weights are taken from official iShares IVV holdings on the rebalance trading date or the closest prior available holdings date.
- Because the local stock panel does not contain every historical S&P 500 constituent, IVV holdings are restricted to the observable local universe and renormalized within that local universe.
- The stock-selection and optimization universe is also restricted to names present in the same IVV holdings proxy at that rebalance date.
- Minimum tracking error and predicted active risk are optimized and reported relative to this IVV-holdings proxy benchmark, not relative to the licensed full S&P500 constituent-weight file.


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
- Benchmark constituent weights and historical investable-universe membership are proxied by IVV ETF holdings restricted to the observable local universe, so forecast tracking error remains a proxy estimate.
- Barra-style proxy sector and fundamental exposures come from public data cached in `Project2_data/barra_proxy_metadata.csv`; this is not licensed MSCI Barra data.
- Sector classifications and fundamentals are public-data proxies and are not guaranteed to be fully historical point-in-time.
- Historical return comparison remains against the S&P500 index itself.
