"""
Step 5: EGARCH Model Fitting (Comparison Specification)
GARCH Volatility Forecasting Project

Fits an EGARCH(1,1) model, which allows negative and positive return
shocks to affect future volatility differently -- the "leverage effect"
observed in equity markets, where a price drop tends to raise future
volatility more than an equal-sized gain lowers it. Compares EGARCH's
fitted volatility against GARCH's from step 3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path

# ---- Config ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RETURNS_CSV = OUTPUT_DIR / "spy_returns.csv"
GARCH_FITTED_CSV = OUTPUT_DIR / "spy_garch_fitted.csv"
TRADING_DAYS = 252
OUTPUT_CSV = OUTPUT_DIR / "spy_egarch_fitted.csv"
OUTPUT_PLOT = OUTPUT_DIR / "spy_garch_vs_egarch_plot.png"

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["LogReturn"]

def fit_egarch(returns: pd.Series):
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="EGARCH", p=1, o=1, q=1, dist="normal")
    fitted = model.fit(disp="off")
    return fitted

def main():
    returns = load_returns(RETURNS_CSV)
    print(f"Fitting EGARCH(1,1) on {len(returns)} daily log returns...\n")

    fitted = fit_egarch(returns)
    print(fitted.summary())

    params = fitted.params
    gamma = params.get("gamma[1]", None)  # asymmetry (leverage) term

    print("\n--- Parameter interpretation ---")
    print(f"omega:    {params['omega']:.5f}  (log-variance equation intercept)")
    print(f"alpha[1]: {params['alpha[1]']:.4f}  (reaction to shock size)")
    if gamma is not None:
        print(f"gamma[1]: {gamma:.4f}  (asymmetry / leverage term)")
        if gamma < 0:
            print("  Negative gamma is consistent with the leverage effect: negative "
                  "shocks increase future volatility more than positive shocks of the "
                  "same size.")
        elif gamma > 0:
            print("  Positive gamma is atypical for equities -- worth double-checking "
                  "and noting as an unexpected result.")
    print(f"beta[1]:  {params['beta[1]']:.4f}  (persistence -- weight on prior forecast)")

    # Fitted conditional volatility (in-sample), annualized
    daily_vol = fitted.conditional_volatility / 100
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS)
    annualized_vol.index = returns.index[-len(annualized_vol):]

    out = pd.DataFrame({"EGARCH_ConditionalVol": annualized_vol})
    out.to_csv(OUTPUT_CSV, lineterminator="\n")
    print(f"\nSaved fitted conditional volatility to {OUTPUT_CSV}")

    # Model comparison via AIC/BIC (lower = better, penalized for complexity)
    print("\n--- Model comparison ---")
    print(f"EGARCH  AIC: {fitted.aic:.2f}   BIC: {fitted.bic:.2f}")
    print("Compare against GARCH(1,1) AIC/BIC from step 3's output. Lower values "
          "indicate better relative in-sample fit among these specifications; "
          "they do not establish out-of-sample superiority.")

    # EGARCH vs GARCH fitted volatility
    try:
        garch = pd.read_csv(GARCH_FITTED_CSV, index_col=0, parse_dates=True)
        plt.figure(figsize=(10, 4.5))
        plt.plot(garch["GARCH_ConditionalVol"], label="GARCH(1,1)", linewidth=0.9, alpha=0.8)
        plt.plot(annualized_vol, label="EGARCH(1,1)", linewidth=0.9, alpha=0.8)
        plt.title("SPY: GARCH(1,1) vs. EGARCH(1,1) Fitted Volatility")
        plt.ylabel("Annualized Volatility")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150)
        print(f"Saved comparison plot to {OUTPUT_PLOT}")
    except FileNotFoundError:
        print(f"\n({GARCH_FITTED_CSV} not found -- skipping comparison plot. Run step 3 first.)")

if __name__ == "__main__":
    main()
