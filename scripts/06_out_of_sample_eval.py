"""Step 6: target-aligned one-step-ahead variance evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RETURNS_CSV = OUTPUT_DIR / "spy_returns.csv"
BASELINE_CSV = OUTPUT_DIR / "spy_baseline_vol.csv"
OUTPUT_CSV = OUTPUT_DIR / "spy_oos_evaluation.csv"
SCORES_CSV = OUTPUT_DIR / "spy_oos_scores.csv"
OUTPUT_PLOT = OUTPUT_DIR / "spy_oos_evaluation_plot.png"
TRADING_DAYS = 252
TRAIN_END = "2023-12-31"
REFIT_EVERY = 63


def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "LogReturn" not in df.columns:
        raise ValueError(f"Expected a 'LogReturn' column in {path}")
    return df["LogReturn"].dropna().sort_index()


def rolling_variance_forecast(
    returns: pd.Series,
    vol_model: str,
    train_end: str,
    refit_every: int,
) -> pd.Series:
    """Return target-aligned one-step-ahead daily variance forecasts."""
    if refit_every <= 0:
        raise ValueError("refit_every must be positive")
    if vol_model not in {"GARCH", "EGARCH"}:
        raise ValueError("vol_model must be 'GARCH' or 'EGARCH'")

    scaled = returns.sort_index() * 100
    test_dates = scaled.index[scaled.index > train_end]
    if test_dates.empty:
        raise ValueError("No observations occur after train_end")

    forecasts: dict[pd.Timestamp, float] = {}
    for block_start in test_dates[::refit_every]:
        start_pos = scaled.index.get_loc(block_start)
        history = scaled.iloc[:start_pos]
        if history.empty:
            raise ValueError("Training sample is empty")

        model_kwargs = {
            "vol": vol_model,
            "p": 1,
            "o": 1 if vol_model == "EGARCH" else 0,
            "q": 1,
            "dist": "normal",
        }
        fitted = arch_model(history, **model_kwargs).fit(disp="off")

        # Target alignment places the forecast made through t-1 on date t.
        fixed = arch_model(scaled, **model_kwargs).fix(fitted.params)
        end_pos = min(start_pos + refit_every, len(scaled))
        block_dates = scaled.index[start_pos:end_pos]
        block = fixed.forecast(
            horizon=1,
            start=start_pos - 1,
            align="target",
            reindex=True,
        ).variance["h.1"].loc[block_dates]

        # The model uses percentage returns; restore decimal-return variance.
        forecasts.update((block / 100**2).to_dict())

    result = pd.Series(forecasts, name=f"{vol_model}Variance").sort_index()
    if result.isna().any() or not result.index.equals(test_dates):
        raise RuntimeError("Forecast output is incomplete or misaligned")
    return result


def _aligned(forecast: pd.Series, actual: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [forecast.rename("forecast"), actual.rename("actual")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("Forecast and actual series have no overlapping observations")
    if (aligned["forecast"] <= 0).any():
        raise ValueError("Variance forecasts must be strictly positive")
    return aligned


def compute_mse(forecast: pd.Series, actual: pd.Series) -> float:
    aligned = _aligned(forecast, actual)
    return float(((aligned["forecast"] - aligned["actual"]) ** 2).mean())


def compute_qlike(forecast: pd.Series, actual: pd.Series) -> float:
    """QLIKE loss, omitting constants that do not affect model rankings."""
    aligned = _aligned(forecast, actual)
    return float((aligned["actual"] / aligned["forecast"]
                  + np.log(aligned["forecast"])).mean())


def main():
    returns = load_returns(RETURNS_CSV)
    baseline = pd.read_csv(BASELINE_CSV, index_col=0, parse_dates=True)
    test_dates = returns.index[returns.index > TRAIN_END]

    print(f"Training period: through {TRAIN_END}")
    print(f"Test period: {test_dates.min().date()} to {test_dates.max().date()} "
          f"({len(test_dates)} days)")
    print(f"Re-estimating parameters every {REFIT_EVERY} trading days\n")

    print("Running target-aligned GARCH(1,1) forecasts...")
    garch_variance = rolling_variance_forecast(
        returns, "GARCH", TRAIN_END, REFIT_EVERY
    )
    print("Running target-aligned EGARCH(1,1) forecasts...")
    egarch_variance = rolling_variance_forecast(
        returns, "EGARCH", TRAIN_END, REFIT_EVERY
    )

    naive_variance = (
        baseline.loc[test_dates, "NaiveForecast"].pow(2) / TRADING_DAYS
    ).rename("NaiveVariance")
    squared_return = returns.loc[test_dates].pow(2).rename("SquaredReturn")

    forecasts = {
        "Naive": naive_variance,
        "GARCH": garch_variance,
        "EGARCH": egarch_variance,
    }
    scores = pd.DataFrame({
        name: {
            "MSE": compute_mse(forecast, squared_return),
            "QLIKE": compute_qlike(forecast, squared_return),
        }
        for name, forecast in forecasts.items()
    }).T

    print("\n--- One-step-ahead variance forecast losses (lower = better) ---")
    print(scores.to_string(float_format=lambda value: f"{value:.8g}"))
    print(f"\nBest by MSE:   {scores['MSE'].idxmin()}")
    print(f"Best by QLIKE: {scores['QLIKE'].idxmin()}")

    results = pd.DataFrame({
        "SquaredReturn": squared_return,
        "Trailing30DayVol": baseline.loc[test_dates, "RealizedVol"],
        "NaiveVariance": naive_variance,
        "GARCHVariance": garch_variance,
        "EGARCHVariance": egarch_variance,
    })
    for name in ["Naive", "GARCH", "EGARCH"]:
        results[f"{name}Forecast"] = np.sqrt(
            results[f"{name}Variance"] * TRADING_DAYS
        )
    results.to_csv(OUTPUT_CSV, lineterminator="\n")
    scores.to_csv(SCORES_CSV, lineterminator="\n")
    print(f"\nSaved forecasts to {OUTPUT_CSV}")
    print(f"Saved loss scores to {SCORES_CSV}")

    plt.figure(figsize=(11, 5))
    plt.plot(results["Trailing30DayVol"],
             label="Trailing 30-day volatility (context)",
             linewidth=1.0, color="black", alpha=0.65)
    for name in ["Naive", "GARCH", "EGARCH"]:
        plt.plot(results[f"{name}Forecast"],
                 label=f"{name} one-day forecast", linewidth=0.8, alpha=0.8)
    plt.title("Target-Aligned One-Step-Ahead Volatility Forecasts")
    plt.ylabel("Annualized volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
