# GARCH Volatility Forecasting on SPY

A from-scratch implementation and honest out-of-sample evaluation of GARCH-family
volatility models on S&P 500 (SPY) data, built as an independent quantitative
finance project.

## Question

Does GARCH volatility clustering actually improve volatility forecasts over a
naive baseline, or does added model complexity fail to pay off in practice?

## Data

- **Instrument:** SPY (SPDR S&P 500 ETF Trust)
- **Period:** January 2010 - August 2026 (4,188 trading days)
- **Source:** Yahoo Finance via the `yfinance` package
- **Training window:** through Dec 31, 2023
- **Test window:** Jan 2024 - Aug 2026 (667 trading days, out-of-sample)

## Method

1. Pull daily prices, compute log returns
2. Build two baselines: 30-day rolling realized volatility, and a naive
   next-day forecast
3. Fit GARCH(1,1) via the `arch` package
4. Run residual diagnostics (Ljung-Box test on squared standardized residuals)
   to confirm the model actually removes volatility clustering
5. Fit EGARCH(1,1) to test for the equity "leverage effect" (asymmetric
   response to negative vs. positive shocks)
6. Evaluate all models out-of-sample against realized volatility using RMSE
7. Re-evaluate against an unsmoothed daily volatility proxy, to test whether
   model ranking depends on the evaluation target

## Key results

| Finding | Result |
|---|---|
| Volatility persistence (GARCH) | alpha + beta ~ 0.965 |
| Residual clustering removed | Ljung-Box p = 0.82 (post-GARCH) vs. p ~ 0.0 (raw) |
| Leverage effect (EGARCH) | gamma ~ -0.172 (confirmed, negative) |
| EGARCH vs. GARCH fit | AIC 10,564.6 vs. 10,756.9 (EGARCH better) |
| OOS RMSE, smoothed target | Naive 0.0099 vs. GARCH 0.0525 (naive wins) |
| OOS RMSE, daily target | GARCH 0.0994 vs. Naive 0.1276 (GARCH wins) |

The full discussion, including why the model ranking reverses across targets
and what it means, is in [`WRITEUP.md`](WRITEUP.md).

## Repository structure

```
scripts/
  01_data_acquisition.py        - Pull SPY data, compute log returns
  02_baseline_models.py         - Rolling realized vol + naive forecast
  03_garch_fit.py                - Fit GARCH(1,1)
  04_residual_diagnostics.py    - Ljung-Box test on residuals
  05_egarch_fit.py               - Fit EGARCH(1,1), leverage effect
  06_out_of_sample_eval.py      - OOS RMSE vs. smoothed target
  07_daily_target_comparison.py - OOS RMSE vs. daily target
outputs/                         - Generated CSVs and plots
WRITEUP.md                       - Full research-style write-up
requirements.txt
README.md
```

## Running it

```bash
pip install -r requirements.txt
python scripts/01_data_acquisition.py
python scripts/02_baseline_models.py
python scripts/03_garch_fit.py
python scripts/04_residual_diagnostics.py
python scripts/05_egarch_fit.py
python scripts/06_out_of_sample_eval.py
python scripts/07_daily_target_comparison.py
```

Each script reads the CSV(s) produced by earlier steps, so they should be run
in order the first time.

## Limitations

GARCH assumes normally distributed shocks despite SPY's fat-tailed return
distribution; parameters are re-estimated quarterly rather than daily during
out-of-sample evaluation; and the test period contains only one major
volatility event. See [`WRITEUP.md`](WRITEUP.md) for the full discussion.
