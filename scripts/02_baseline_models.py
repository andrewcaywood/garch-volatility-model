"""
Step 2: Baseline Volatility Models
GARCH Volatility Forecasting Project

Reads the log returns produced in step 1 (spy_returns.csv) and builds
two baseline volatility measures:
  1. Rolling realized volatility (trailing N-day standard deviation of returns)
  2. A naive forecast (tomorrow's vol = today's trailing rolling vol)

These serve as the comparison point for GARCH in later steps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
INPUT_CSV = "spy_returns.csv"
WINDOW = 30             # trailing window length in trading days
TRADING_DAYS = 252      # annualization factor for daily vol
OUTPUT_CSV = "spy_baseline_vol.csv"

def load_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "LogReturn" not in df.columns:
        raise ValueError(f"Expected a 'LogReturn' column in {path}, found: {df.columns.tolist()}")
    return df

def compute_rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling realized volatility: standard deviation of returns over a
    trailing window, annualized by sqrt(252) (daily variance scales
    linearly with time, so daily std scales with sqrt of trading days/year).
    """
    daily_vol = returns.rolling(window=window).std()
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS)
    return annualized_vol.dropna()

def compute_naive_forecast(realized_vol: pd.Series) -> pd.Series:
    """Naive forecast: tomorrow's predicted volatility = today's trailing realized volatility."""
    return realized_vol.shift(1).dropna()

def main():
    df = load_returns(INPUT_CSV)
    returns = df["LogReturn"]

    realized_vol = compute_rolling_volatility(returns, WINDOW)
    naive_forecast = compute_naive_forecast(realized_vol)

    out = pd.DataFrame({
        "RealizedVol": realized_vol,
        "NaiveForecast": naive_forecast
    }).dropna()

    out.to_csv(OUTPUT_CSV)
    print(f"Saved {len(out)} rows to {OUTPUT_CSV}")

    print("\n--- Summary: annualized realized volatility ---")
    print(f"Mean:   {realized_vol.mean():.4f}  ({realized_vol.mean()*100:.1f}%)")
    print(f"Min:    {realized_vol.min():.4f}  ({realized_vol.min()*100:.1f}%)")
    print(f"Max:    {realized_vol.max():.4f}  ({realized_vol.max()*100:.1f}%)")

    # Realized vol vs. naive forecast (naive is realized vol shifted by one day)
    plt.figure(figsize=(10, 4.5))
    plt.plot(realized_vol, label=f"{WINDOW}-day Realized Volatility (annualized)", linewidth=0.9)
    plt.plot(naive_forecast, label="Naive Forecast (yesterday's realized vol)",
              linewidth=0.9, linestyle="--", alpha=0.8)
    plt.title(f"SPY: {WINDOW}-Day Rolling Realized Volatility vs. Naive Forecast")
    plt.ylabel("Annualized Volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spy_baseline_vol_plot.png", dpi=150)
    print("\nSaved plot to spy_baseline_vol_plot.png")

if __name__ == "__main__":
    main()
