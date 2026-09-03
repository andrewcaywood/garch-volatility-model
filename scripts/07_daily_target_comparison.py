"""
Step 7: Daily-Target Comparison
GARCH Volatility Forecasting Project

Step 6 scored GARCH, EGARCH, and the naive baseline against a smoothed
30-day realized-volatility target. The naive baseline performed best on
that target, largely because it is derived directly from the same 30-day
rolling average it is being scored against.

This step re-scores all three models against an unsmoothed daily
volatility proxy: each day's absolute return, annualized. This is the
timescale GARCH is designed to react to. Comparing both results shows
whether model ranking depends on the evaluation target.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
RETURNS_CSV = "spy_returns.csv"
OOS_EVAL_CSV = "spy_oos_evaluation.csv"   # forecasts produced in step 6
TRADING_DAYS = 252

def compute_daily_vol_proxy(returns: pd.Series) -> pd.Series:
    """
    A single day's volatility can't be measured with a standard deviation
    (that requires multiple observations), so the standard proxy in the
    literature is the day's absolute return, annualized. It's a noisy but
    unbiased estimate of that day's true (unobservable) volatility.
    """
    return returns.abs() * np.sqrt(TRADING_DAYS)

def compute_rmse(forecast: pd.Series, actual: pd.Series) -> float:
    aligned = pd.concat([forecast.rename("forecast"), actual.rename("actual")],
                         axis=1, join="inner")
    return np.sqrt(((aligned["forecast"] - aligned["actual"]) ** 2).mean())

def main():
    returns_df = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    oos = pd.read_csv(OOS_EVAL_CSV, index_col=0, parse_dates=True)

    daily_target = compute_daily_vol_proxy(returns_df["LogReturn"])
    daily_target = daily_target.loc[oos.index]

    print(f"Test period: {oos.index.min().date()} to {oos.index.max().date()} "
          f"({len(oos)} days)\n")

    rmse_naive_daily = compute_rmse(oos["NaiveForecast"], daily_target)
    rmse_garch_daily = compute_rmse(oos["GARCHForecast"], daily_target)
    rmse_egarch_daily = compute_rmse(oos["EGARCHForecast"], daily_target)

    rmse_naive_smooth = compute_rmse(oos["NaiveForecast"], oos["RealizedVol"])
    rmse_garch_smooth = compute_rmse(oos["GARCHForecast"], oos["RealizedVol"])
    rmse_egarch_smooth = compute_rmse(oos["EGARCHForecast"], oos["RealizedVol"])

    print("--- RMSE vs. smoothed 30-day realized volatility (from step 6) ---")
    print(f"Naive:   {rmse_naive_smooth:.4f}")
    print(f"GARCH:   {rmse_garch_smooth:.4f}")
    print(f"EGARCH:  {rmse_egarch_smooth:.4f}")

    print("\n--- RMSE vs. unsmoothed daily volatility proxy (this step) ---")
    print(f"Naive:   {rmse_naive_daily:.4f}")
    print(f"GARCH:   {rmse_garch_daily:.4f}")
    print(f"EGARCH:  {rmse_egarch_daily:.4f}")

    best_smooth = min([("Naive", rmse_naive_smooth), ("GARCH", rmse_garch_smooth),
                        ("EGARCH", rmse_egarch_smooth)], key=lambda x: x[1])
    best_daily = min([("Naive", rmse_naive_daily), ("GARCH", rmse_garch_daily),
                       ("EGARCH", rmse_egarch_daily)], key=lambda x: x[1])

    print(f"\nBest on smoothed target: {best_smooth[0]}")
    print(f"Best on daily target:    {best_daily[0]}")
    if best_smooth[0] != best_daily[0]:
        print("\nRanking changes depending on the evaluation target. No model is "
              "'best' in an absolute sense here -- performance depends on the "
              "timescale being forecast.")
    else:
        print(f"\n{best_daily[0]} wins on both targets.")

    out = pd.DataFrame({
        "DailyVolTarget": daily_target,
        "SmoothedVolTarget": oos["RealizedVol"],
        "NaiveForecast": oos["NaiveForecast"],
        "GARCHForecast": oos["GARCHForecast"],
        "EGARCHForecast": oos["EGARCHForecast"],
    })
    out.to_csv("spy_daily_target_comparison.csv")
    print("\nSaved detailed results to spy_daily_target_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Naive", "GARCH(1,1)", "EGARCH(1,1)"]
    smooth_scores = [rmse_naive_smooth, rmse_garch_smooth, rmse_egarch_smooth]
    daily_scores = [rmse_naive_daily, rmse_garch_daily, rmse_egarch_daily]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, smooth_scores, width, label="vs. Smoothed 30-day target")
    ax.bar(x + width/2, daily_scores, width, label="vs. Unsmoothed daily target")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("RMSE (lower = better)")
    ax.set_title("Model Ranking by Evaluation Target")
    ax.legend()
    plt.tight_layout()
    plt.savefig("spy_daily_target_comparison_plot.png", dpi=150)
    print("Saved plot to spy_daily_target_comparison_plot.png")

if __name__ == "__main__":
    main()
