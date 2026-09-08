"""
Step 4: Residual Diagnostics
GARCH Volatility Forecasting Project

Checks whether the fitted GARCH(1,1) model removed volatility clustering
from the returns, or whether structure is still left over.

Method: standardize the model's residuals by the fitted conditional
volatility, then run a Ljung-Box test on the squared standardized
residuals. A low p-value indicates left-over autocorrelation, i.e.
clustering the model failed to capture.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path

# ---- Config ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RETURNS_CSV = OUTPUT_DIR / "spy_returns.csv"
OUTPUT_PLOT = OUTPUT_DIR / "spy_residual_diagnostics_plot.png"
LAGS = 20  # standard choice for daily data (~1 trading month)

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["LogReturn"]

def fit_garch(returns: pd.Series):
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal")
    return model.fit(disp="off")

def main():
    returns = load_returns(RETURNS_CSV)
    fitted = fit_garch(returns)

    # Standardized residuals: model residual divided by its forecasted
    # conditional volatility. If the model fits well, these should behave
    # like unstructured noise.
    std_resid = fitted.std_resid.dropna()

    print(f"Standardized residuals: {len(std_resid)} observations")
    print(f"Mean:     {std_resid.mean():.4f}  (expected close to 0)")
    print(f"Std Dev:  {std_resid.std():.4f}  (expected close to 1)")

    # Ljung-Box test on squared standardized residuals.
    # Null hypothesis: no autocorrelation remains. A low p-value rejects
    # that, i.e. clustering structure is still present.
    squared_resid = std_resid ** 2
    lb_result = acorr_ljungbox(squared_resid, lags=[LAGS], return_df=True)

    print(f"\n--- Ljung-Box test on squared standardized residuals (lag={LAGS}) ---")
    print(lb_result)

    p_value = lb_result["lb_pvalue"].iloc[0]
    print(f"\np-value: {p_value:.4f}")
    if p_value < 0.05:
        print("p < 0.05: reject the null of 'no remaining structure.'")
        print("Some volatility clustering is still present after GARCH(1,1) -- "
              "a legitimate limitation to note, not a coding error.")
    else:
        print("p >= 0.05: no strong evidence of remaining clustering.")
        print("At this lag choice, the test finds no strong evidence of remaining "
              "squared-residual autocorrelation. This is not proof that all "
              "conditional-variance structure has been removed.")

    # Same test on raw returns, for comparison / context
    raw_squared = (returns - returns.mean()) ** 2
    lb_raw = acorr_ljungbox(raw_squared, lags=[LAGS], return_df=True)
    print(f"\n--- Same test on raw returns (no model) ---")
    print(lb_raw)
    print("(Expected to be near zero, confirming clustering in the raw data.)")

    # ACF plots: squared residuals before and after fitting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    plot_acf(squared_resid, lags=LAGS, ax=axes[0])
    axes[0].set_title("ACF: Squared Standardized Residuals (post-GARCH)")

    plot_acf(raw_squared, lags=LAGS, ax=axes[1])
    axes[1].set_title("ACF: Squared Raw Returns (pre-model)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
