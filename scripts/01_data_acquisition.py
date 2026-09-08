"""
Step 1: Data Acquisition
GARCH Volatility Forecasting Project

Pulls daily price history for SPY, computes daily log returns,
and saves both to a CSV for use in later steps (baseline vol,
GARCH fitting, evaluation).
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Config ----
TICKER = "SPY"
START_DATE = "2010-01-01"   # long history for GARCH to fit on
# yfinance treats end as exclusive. Freezing it keeps the published sample
# reproducible and includes observations through 2026-08-28.
END_DATE = "2026-08-29"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "spy_returns.csv"
OUTPUT_PLOT = OUTPUT_DIR / "spy_returns_plot.png"

def fetch_price_data(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV data and return it with a clean DatetimeIndex."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range/network.")
    # yfinance returns MultiIndex columns (e.g. ('Close', 'SPY')) even for a
    # single ticker as of recent versions; flatten to plain column names.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna()
    return data

def compute_log_returns(prices: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Daily log return: ln(P_t / P_{t-1}).
    Log returns are used instead of simple returns because they are
    additive across time and are the standard input for GARCH models.
    """
    close = prices[price_col]
    log_returns = np.log(close / close.shift(1))
    return log_returns.dropna()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching {TICKER} price history from {START_DATE}...")
    prices = fetch_price_data(TICKER, START_DATE, END_DATE)
    print(f"Retrieved {len(prices)} trading days.")

    log_returns = compute_log_returns(prices)

    out = pd.DataFrame({
        "Close": prices["Close"].loc[log_returns.index],
        "LogReturn": log_returns
    })
    out.to_csv(OUTPUT_CSV, lineterminator="\n")
    print(f"Saved {len(out)} rows to {OUTPUT_CSV}")

    print("\n--- Summary statistics: daily log returns ---")
    print(f"Mean:      {log_returns.mean():.6f}")
    print(f"Std Dev:   {log_returns.std():.6f}")
    print(f"Skew:      {log_returns.skew():.4f}")
    print(f"Kurtosis:  {log_returns.kurt():.4f}  (excess kurtosis; 0 = normal-ish tails)")

    # Price and return series, for a visual check of volatility clustering
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(prices["Close"].loc[log_returns.index], linewidth=0.8)
    axes[0].set_title(f"{TICKER} Adjusted Close Price")
    axes[0].set_ylabel("Price ($)")

    axes[1].plot(log_returns, linewidth=0.6, color="darkorange")
    axes[1].set_title(f"{TICKER} Daily Log Returns")
    axes[1].set_ylabel("Log Return")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved diagnostic plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
