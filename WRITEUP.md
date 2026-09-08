# Volatility Clustering in SPY: A GARCH-Based Forecasting Comparison

## Research question

Volatility in equity markets clusters: large movements tend to be followed by
large movements and quiet periods tend to persist. This project asks whether a
GARCH(1,1) model captures that conditional-variance structure in SPY returns and
whether GARCH-family one-day-ahead forecasts outperform a simple trailing-window
benchmark out of sample.

## Data and methodology

The sample contains 4,188 daily SPY log returns from January 2010 through
August 28, 2026. The download end date is frozen for reproducibility. Returns
are expressed in percentage units while fitting the models and converted back
to decimal units for evaluation.

The analysis first calculates a 30-day trailing realized-volatility series and
a naive forecast equal to the preceding day's trailing volatility. It then fits
constant-mean GARCH(1,1) and EGARCH(1,1) models with Gaussian innovations. The
full-sample fits are used only for parameter interpretation and diagnostics.

The genuine out-of-sample exercise begins on January 2, 2024. Model parameters
are estimated using observations strictly before each forecast block and are
re-estimated every 63 trading days. Between refits, the conditional-variance
recursion updates daily with newly observed returns. Forecasts are explicitly
target-aligned: the forecast indexed by date *t* uses information only through
date *t-1*.

Because the models produce one-day conditional-variance forecasts, the primary
evaluation uses each day's squared return as a noisy proxy for that day's latent
variance. Models are ranked using both mean squared error (MSE) and QLIKE. The
backward-looking 30-day volatility series is retained for visual context but is
not treated as the realization of a one-day forecast.

A secondary analysis converts forecasts to annualized volatility and compares
them with absolute returns multiplied by √(π/2) and √252. This corrects the
downward bias of raw absolute returns under a zero-mean Gaussian assumption.
Since that assumption is restrictive, the secondary RMSE is supporting evidence
rather than the headline comparison.

## Full-sample model results

The GARCH estimate has α + β ≈ 0.965, indicating persistent but mean-reverting
conditional variance. Its implied long-run annualized volatility is about
16.84%; ω itself is the variance-equation intercept, not the long-run variance.

At lag 20, the Ljung–Box p-value for squared standardized GARCH residuals is
approximately 0.82, compared with effectively zero for squared raw returns.
This means the test finds no strong evidence of remaining squared-residual
autocorrelation at the selected lags. It does not prove that every form of
conditional heteroskedasticity has been removed.

The EGARCH asymmetry coefficient is approximately -0.172 and is statistically
significant in the fitted specification, consistent with negative shocks having
a larger volatility effect than equal-sized positive shocks. EGARCH also has a
lower full-sample AIC than GARCH (10,564.6 versus 10,756.9), indicating better
relative in-sample fit among these two Gaussian specifications.

## Corrected out-of-sample results

Lower loss is better:

| Model | Variance MSE | QLIKE |
|---|---:|---:|
| Naive trailing-window forecast | 1.9909e-7 | -8.3742 |
| GARCH(1,1) | 1.8461e-7 | -8.4963 |
| EGARCH(1,1) | **1.7011e-7** | **-8.5634** |

EGARCH ranks first under both primary loss functions. The secondary,
Gaussian-dependent volatility-level comparison produces the same ranking:

| Model | RMSE vs. bias-adjusted absolute return |
|---|---:|
| Naive | 0.1482 |
| GARCH(1,1) | 0.1390 |
| EGARCH(1,1) | **0.1332** |

These values replace the earlier results, which accidentally assigned
origin-aligned forecasts to the dates on which they were produced rather than
to the following dates they forecast. That one-day error allowed each day's
return to influence the forecast scored against the same day's proxy.

## Limitations

- Squared daily returns are unbiased for the conditional second moment when the
  conditional mean is negligible, but they are extremely noisy variance proxies.
  Intraday realized variance would provide a stronger evaluation target.
- Gaussian innovations do not describe the fat tails in SPY returns well. A
  Student's-t specification should be tested as a robustness check.
- Quarterly parameter refitting is computationally convenient but may respond
  more slowly to structural changes than daily or monthly refitting.
- The out-of-sample period is short and contains relatively few major volatility
  episodes, so the ranking may not generalize to other market regimes.
- MSE and QLIKE agree here, but forecast rankings can depend on the loss function
  and the economic use of the forecast.
- Yahoo Finance data may be revised by the provider even though the requested
  sample dates and package versions are fixed.

## Conclusion

The corrected analysis supports three measured conclusions. First, GARCH(1,1)
captures much of the squared-return autocorrelation detected by the selected
diagnostic. Second, the negative EGARCH asymmetry coefficient is consistent with
the equity leverage effect. Third, EGARCH provides the lowest out-of-sample MSE
and QLIKE among the three forecasts tested. These are conditional findings for
this sample and specification, not proof that EGARCH is universally superior.
