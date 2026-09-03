# Volatility Clustering in SPY: A GARCH-Based Forecasting Comparison

**Research question**

Volatility in equity markets is not constant over time. Instead, high-volatility periods tend to cluster together, while low-volatility periods tend to be prolonged. This project tries to answer two specific questions: (1) whether a GARCH(1,1) model can capture such clustering of volatility when applied to SPY return data, and (2) whether it forecasts better than a naive alternative.

**Data and methodology**

Daily adjusted-close prices for SPY were obtained between 2010 and 2026 (4,188 trading days). Using the daily prices, daily log returns were then generated. Two baseline measures were created: a 30-day rolling realized volatility (a backward-looking measure of how volatile SPY actually was over the trailing month) and a naive forecast built from it (tomorrow's volatility = yesterday's 30-day realized volatility). Both exist to provide a benchmark, something to be beaten by more complicated models.

A GARCH(1,1) model was used, in which today's volatility is a weighted average of the long-term historical average, yesterday's forecast, and the size of yesterday's shock. The GARCH model was compared against an EGARCH(1,1) model, which is capable of capturing the leverage effect; both were first fit on the full 2010-2026 dataset to examine their parameters and overall fit. Separately, both models were re-estimated using only data through the end of 2023, then used to produce forecasts for 2024 to 2026 (updated daily using real, not future, returns), so their forecasts could be rated by RMSE against a naive baseline.

**Results**

The GARCH(1,1) model produced alpha + beta ~ 0.965, implying that shocks to volatility decay gradually and that a volatile period should stay elevated for some time before returning to normal. When Ljung-Box tests were carried out on the squared standardized residuals from the model, the p-value was ~0.0 for the raw returns vs. 0.82 for the GARCH residuals, confirming the model successfully removed the clustering of volatility present in the raw data. The EGARCH model's asymmetry term came out negative and statistically significant (gamma ~ -0.172), confirming the leverage effect: negative shocks raise future volatility more than equivalent positive shocks do. The AIC goodness-of-fit statistic was also smaller for the EGARCH model than for plain GARCH (10,564.6 vs. 10,756.9, respectively), meaning it fit the data better overall.

Two out-of-sample tests were carried out: one against the smoothed 30-day realized-volatility target, and one against a noisier daily volatility proxy. Against the smoothed target, the naive baseline led the way with the lowest RMSE (0.0099 vs. 0.0525 for GARCH and 0.0624 for EGARCH), but against the noisier daily measure, GARCH came out ahead (0.0994 vs. 0.1276 for naive and 0.0999 for EGARCH).

**Limitations**

The evaluation period (2024 to 2026) used in this project contains only one significant volatility shock (the April 2025 tariffs-related selloff). As such, any conclusions made about fat-tail behavior rest on very few episodes. Second, the GARCH model is based on the assumption of normally distributed innovations, which is not consistent with the fact that SPY shows clear excess kurtosis; a Student's-t distribution would probably fit better. Third, re-estimation was only performed on roughly a quarterly basis rather than daily, as would be customary in practice. Lastly, the switch in the best-performing model depending on whether the target was the smoothed 30-day realized volatility or the noisier daily measure is an indication that the choice of evaluation target - not just the choice of model - can significantly influence model ranking.
