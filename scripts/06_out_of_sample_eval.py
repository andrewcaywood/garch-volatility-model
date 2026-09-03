"""
Step 6: Out-of-Sample Evaluation
GARCH Volatility Forecasting Project

Splits the data into a training period and a test period, fits GARCH(1,1)
and EGARCH(1,1) on the training period only, then produces rolling
one-step-ahead forecasts through the test period. Parameters are
re-estimated periodically (not daily) to keep runtime reasonable, but
forecasts are still updated every day using the actual returns observed,
so they're comparable to the daily-updating naive baseline. Compares
GARCH, EGARCH, and the naive baseline against realized volatility using
RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# ---- Config ----
RETURNS_CSV = "spy_returns.csv"
BASELINE_CSV = "spy_baseline_vol.csv"
TRADING_DAYS = 252
TRAIN_END = "2023-12-31"   # everything up to here = training data
REFIT_EVERY = 63           # re-estimate parameters roughly quarterly
WINDOW = 30                # must match the window used in step 2's realized vol

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["LogReturn"]

def rolling_forecast(returns: pd.Series, vol_model: str, train_end: str, refit_every: int):
    """
    Produce one-step-ahead annualized volatility forecasts for the test
    period, updated daily.

    Parameters are re-estimated every `refit_every` trading days rather
    than daily, for compute cost. Between re-estimations, forecasts still
    update daily: a model is fit on the full return series with parameters
    fixed at their most recent estimate, and its forecast is generated
    using arch's fixed-parameter forecasting, which runs the volatility
    recursion against actual observed returns each day. Only the
    parameters (omega, alpha, beta, etc.) stay constant between
    re-estimations -- the forecast itself reacts to new data daily.
    """
    scaled = returns * 100
    test_dates = scaled.index[scaled.index > train_end]

    forecasts = {}
    refit_points = test_dates[::refit_every]

    for block_start in refit_points:
        # Estimate parameters using all data strictly before this block starts
        history = scaled.loc[:block_start].iloc[:-1]
        model = arch_model(history, vol=vol_model,
                            p=1, o=(1 if vol_model == "EGARCH" else 0), q=1,
                            dist="normal")
        fitted = model.fit(disp="off")

        # Forecast one-step-ahead for every day in this block, using the
        # full series and fixed parameters, so the recursion reflects real
        # returns as they occur.
        full_model = arch_model(scaled, vol=vol_model,
                                 p=1, o=(1 if vol_model == "EGARCH" else 0), q=1,
                                 dist="normal")
        fixed_res = full_model.fix(fitted.params)

        block_end_idx = min(scaled.index.get_loc(block_start) + refit_every, len(scaled))
        block_forecast = fixed_res.forecast(horizon=1, start=block_start, reindex=False)

        block_dates = scaled.index[scaled.index.get_loc(block_start):block_end_idx]
        daily_var = block_forecast.variance.values[:len(block_dates), 0]
        daily_vol = np.sqrt(daily_var) / 100
        annualized = daily_vol * np.sqrt(TRADING_DAYS)

        for d, v in zip(block_dates, annualized):
            if d in test_dates:
                forecasts[d] = v

    return pd.Series(forecasts).sort_index()

def compute_rmse(forecast: pd.Series, actual: pd.Series) -> float:
    aligned = pd.concat([forecast.rename("forecast"), actual.rename("actual")],
                         axis=1).dropna()
    return np.sqrt(((aligned["forecast"] - aligned["actual"]) ** 2).mean())

def main():
    returns = load_returns(RETURNS_CSV)
    baseline = pd.read_csv(BASELINE_CSV, index_col=0, parse_dates=True)

    print(f"Training period: through {TRAIN_END}")
    print(f"Test period: {TRAIN_END} onward ({(returns.index > TRAIN_END).sum()} days)")
    print(f"Re-estimating parameters every {REFIT_EVERY} trading days\n")

    realized = baseline["RealizedVol"]

    print("Running GARCH(1,1) rolling forecast (this may take a minute)...")
    garch_forecast = rolling_forecast(returns, "GARCH", TRAIN_END, REFIT_EVERY)

    print("Running EGARCH(1,1) rolling forecast (this may take a minute)...")
    egarch_forecast = rolling_forecast(returns, "EGARCH", TRAIN_END, REFIT_EVERY)

    naive_forecast = baseline["NaiveForecast"]
    naive_forecast = naive_forecast[naive_forecast.index > TRAIN_END]

    rmse_garch = compute_rmse(garch_forecast, realized)
    rmse_egarch = compute_rmse(egarch_forecast, realized)
    rmse_naive = compute_rmse(naive_forecast, realized)

    print("\n--- Out-of-sample RMSE vs. realized volatility (lower = better) ---")
    print(f"Naive baseline: {rmse_naive:.4f}")
    print(f"GARCH(1,1):     {rmse_garch:.4f}")
    print(f"EGARCH(1,1):    {rmse_egarch:.4f}")

    best = min([("Naive", rmse_naive), ("GARCH", rmse_garch), ("EGARCH", rmse_egarch)],
               key=lambda x: x[1])
    print(f"\nBest overall (lowest RMSE): {best[0]}")

    results = pd.DataFrame({
        "RealizedVol": realized[realized.index > TRAIN_END],
        "NaiveForecast": naive_forecast,
        "GARCHForecast": garch_forecast,
        "EGARCHForecast": egarch_forecast,
    })
    results.to_csv("spy_oos_evaluation.csv")
    print("\nSaved detailed results to spy_oos_evaluation.csv")

    plt.figure(figsize=(11, 5))
    plt.plot(results["RealizedVol"], label="Realized Volatility (actual)",
              linewidth=1.1, color="black")
    plt.plot(results["NaiveForecast"], label=f"Naive (RMSE={rmse_naive:.4f})",
              linewidth=0.8, alpha=0.7)
    plt.plot(results["GARCHForecast"], label=f"GARCH(1,1) (RMSE={rmse_garch:.4f})",
              linewidth=0.8, alpha=0.8)
    plt.plot(results["EGARCHForecast"], label=f"EGARCH(1,1) (RMSE={rmse_egarch:.4f})",
              linewidth=0.8, alpha=0.8)
    plt.title(f"Out-of-Sample Volatility Forecasts vs. Realized ({TRAIN_END} onward)")
    plt.ylabel("Annualized Volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spy_oos_evaluation_plot.png", dpi=150)
    print("Saved plot to spy_oos_evaluation_plot.png")

if __name__ == "__main__":
    main()
