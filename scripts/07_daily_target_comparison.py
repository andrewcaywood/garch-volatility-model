"""Step 7: secondary volatility-level forecast comparison."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RETURNS_CSV = OUTPUT_DIR / "spy_returns.csv"
OOS_EVAL_CSV = OUTPUT_DIR / "spy_oos_evaluation.csv"
OUTPUT_CSV = OUTPUT_DIR / "spy_daily_target_comparison.csv"
OUTPUT_PLOT = OUTPUT_DIR / "spy_daily_target_comparison_plot.png"
TRADING_DAYS = 252


def compute_daily_vol_proxy(returns: pd.Series) -> pd.Series:
    """Bias-adjusted |return| proxy under a zero-mean Gaussian assumption."""
    normal_bias_correction = np.sqrt(np.pi / 2)
    return returns.abs() * normal_bias_correction * np.sqrt(TRADING_DAYS)


def compute_rmse(forecast: pd.Series, actual: pd.Series) -> float:
    aligned = pd.concat(
        [forecast.rename("forecast"), actual.rename("actual")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("Forecast and actual series have no overlapping observations")
    return float(np.sqrt(((aligned["forecast"] - aligned["actual"]) ** 2).mean()))


def main():
    returns = pd.read_csv(
        RETURNS_CSV, index_col=0, parse_dates=True
    )["LogReturn"]
    oos = pd.read_csv(OOS_EVAL_CSV, index_col=0, parse_dates=True)
    daily_target = compute_daily_vol_proxy(returns).reindex(oos.index)

    print(f"Test period: {oos.index.min().date()} to {oos.index.max().date()} "
          f"({len(oos)} days)\n")
    print("This is a secondary check: the absolute-return proxy is unbiased for "
          "volatility only under a zero-mean Gaussian assumption.\n")

    scores = pd.Series({
        name: compute_rmse(oos[f"{name}Forecast"], daily_target)
        for name in ["Naive", "GARCH", "EGARCH"]
    }, name="RMSE")
    print("--- RMSE vs. bias-adjusted absolute-return proxy ---")
    print(scores.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest on this secondary target: {scores.idxmin()}")

    out = oos[["NaiveForecast", "GARCHForecast", "EGARCHForecast"]].copy()
    out.insert(0, "BiasAdjustedAbsReturn", daily_target)
    out.to_csv(OUTPUT_CSV, lineterminator="\n")
    print(f"\nSaved detailed results to {OUTPUT_CSV}")

    fig, ax = plt.subplots(figsize=(8, 5))
    scores.plot.bar(ax=ax)
    ax.set_ylabel("RMSE (lower = better)")
    ax.set_title("Secondary Volatility-Level Evaluation")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
