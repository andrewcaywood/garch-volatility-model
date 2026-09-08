import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


oos = load_script("oos_eval", "06_out_of_sample_eval.py")
daily = load_script("daily_eval", "07_daily_target_comparison.py")


class ForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        innovations = rng.standard_normal(380)
        variance = np.empty(380)
        variance[0] = 0.0001
        returns = np.empty(380)
        returns[0] = np.sqrt(variance[0]) * innovations[0]
        for i in range(1, 380):
            variance[i] = 0.000002 + 0.08 * returns[i - 1] ** 2 + 0.90 * variance[i - 1]
            returns[i] = np.sqrt(variance[i]) * innovations[i]
        index = pd.bdate_range("2020-01-01", periods=380)
        self.returns = pd.Series(returns, index=index)
        self.train_end = str(index[299].date())

    def test_forecasts_are_target_aligned_and_complete(self):
        forecast = oos.rolling_variance_forecast(
            self.returns, "GARCH", self.train_end, 40
        )
        expected_index = self.returns.index[self.returns.index > self.train_end]
        self.assertTrue(forecast.index.equals(expected_index))
        self.assertTrue((forecast > 0).all())

    def test_first_forecast_does_not_use_first_test_return(self):
        original = oos.rolling_variance_forecast(
            self.returns, "GARCH", self.train_end, 40
        )
        changed = self.returns.copy()
        changed.loc[original.index[0]] = 0.50
        perturbed = oos.rolling_variance_forecast(
            changed, "GARCH", self.train_end, 40
        )
        self.assertAlmostEqual(original.iloc[0], perturbed.iloc[0], places=14)

    def test_loss_functions(self):
        forecast = pd.Series([1.0, 2.0])
        actual = pd.Series([2.0, 4.0])
        self.assertAlmostEqual(oos.compute_mse(forecast, actual), 2.5)
        expected = np.mean(actual / forecast + np.log(forecast))
        self.assertAlmostEqual(oos.compute_qlike(forecast, actual), expected)

    def test_absolute_return_proxy_bias_correction(self):
        rng = np.random.default_rng(11)
        sigma = 0.01
        returns = pd.Series(rng.normal(0, sigma, 500_000))
        estimated_daily_sigma = (
            daily.compute_daily_vol_proxy(returns).mean() / np.sqrt(252)
        )
        self.assertAlmostEqual(estimated_daily_sigma, sigma, places=4)


if __name__ == "__main__":
    unittest.main()
