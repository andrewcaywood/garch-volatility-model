"""
Step 3: GARCH(1,1) Model Fitting
GARCH Volatility Forecasting Project

Fits a GARCH(1,1) model to the SPY log returns using the `arch` package,
prints the fitted parameters with interpretation, and plots the model's
fitted (in-sample) conditional volatility against the realized volatility
baseline from step 2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# ---- Config ----
RETURNS_CSV = "spy_returns.csv"
BASELINE_CSV = "spy_baseline_vol.csv"
TRADING_DAYS = 252
OUTPUT_CSV = "spy_garch_fitted.csv"

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["LogReturn"]

def fit_garch(returns: pd.Series):
    """
    Fit a GARCH(1,1) model. Returns are scaled by 100 before fitting, which
    is the standard convention for the `arch` package -- it improves the
    optimizer's numerical stability. The scaling is undone afterward.
    """
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")
    return fitted

def main():
    returns = load_returns(RETURNS_CSV)
    print(f"Fitting GARCH(1,1) on {len(returns)} daily log returns...\n")

    fitted = fit_garch(returns)
    print(fitted.summary())

    # Parameter interpretation
    params = fitted.params
    omega = params["omega"]     # long-run variance level
    alpha = params["alpha[1]"]  # weight on yesterday's shock (squared residual)
    beta = params["beta[1]"]    # weight on yesterday's forecasted variance

    print("\n--- Parameter interpretation ---")
    print(f"omega (long-run variance level): {omega:.5f}")
    print(f"alpha (weight on yesterday's shock):    {alpha:.4f}")
    print(f"beta  (weight on yesterday's forecast):  {beta:.4f}")
    print(f"alpha + beta = {alpha + beta:.4f}  (persistence of volatility shocks; "
          f"closer to 1 = slower decay)")

    if alpha + beta >= 1:
        print("Note: alpha + beta >= 1 indicates near-unit-root behavior in variance "
              "(shocks barely decay). Flag this as a limitation in the write-up.")

    # Fitted conditional volatility (in-sample), annualized for comparison with step 2
    daily_vol = fitted.conditional_volatility / 100
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS)
    annualized_vol.index = returns.index[-len(annualized_vol):]

    out = pd.DataFrame({"GARCH_ConditionalVol": annualized_vol})
    out.to_csv(OUTPUT_CSV)
    print(f"\nSaved fitted conditional volatility to {OUTPUT_CSV}")

    # GARCH fitted vol vs. realized vol baseline
    try:
        baseline = pd.read_csv(BASELINE_CSV, index_col=0, parse_dates=True)
        plt.figure(figsize=(10, 4.5))
        plt.plot(baseline["RealizedVol"], label="30-day Realized Volatility",
                  linewidth=0.8, alpha=0.7)
        plt.plot(annualized_vol, label="GARCH(1,1) Fitted Conditional Volatility",
                  linewidth=0.9)
        plt.title("SPY: GARCH(1,1) Fitted Volatility vs. Realized Volatility")
        plt.ylabel("Annualized Volatility")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig("spy_garch_fit_plot.png", dpi=150)
        print("Saved plot to spy_garch_fit_plot.png")
    except FileNotFoundError:
        print(f"\n({BASELINE_CSV} not found -- skipping comparison plot. Run step 2 first.)")

if __name__ == "__main__":
    main()
