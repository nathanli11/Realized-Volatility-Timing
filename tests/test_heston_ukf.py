import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from investment_lab.heston_ukf import (
    HestonParams,
    HestonUKF,
    VolatilityTiming,
    build_timing_positions,
)


def test_compute_signal_linear_is_bounded_and_keeps_index():
    idx = pd.bdate_range("2024-01-01", periods=8)
    spread = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02, -0.04, 0.08, -0.10], index=idx)

    timer = VolatilityTiming(scaling="linear", lookback=5, max_leverage=1.5)
    signal = timer.compute_signal(spread)

    assert signal.index.equals(idx)
    assert signal.iloc[:4].isna().all()
    assert signal.dropna().abs().max() <= 1.5


def test_apply_timing_uses_business_day_lag_and_neutral_before_first_execution():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df_positions = pd.DataFrame({"date": idx, "weight": [1.0] * len(idx), "mid": [1.0] * len(idx)})
    signal = pd.Series([0.5, 2.0], index=idx[:2], name="timing_signal")

    timer = VolatilityTiming()
    timed = timer.apply_timing(df_positions, signal, lag_bdays=1)

    expected = pd.Series(
        [1.0, 0.5, 2.0, 1.0, 1.0],
        index=pd.Index(idx, name="date"),
        name="weight",
    )
    got = timed.set_index("date")["weight"]

    pd.testing.assert_series_equal(got, expected, check_freq=False)


def test_filter_with_rolling_params_returns_series_aligned_on_rolling_index(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams())
    ukf._params = HestonParams()
    ukf._rolling_params = pd.DataFrame(
        {
            "kappa": [1.0, 1.1, 1.2],
            "theta": [0.04, 0.05, 0.06],
            "xi": [0.3, 0.3, 0.3],
            "rho": [-0.7, -0.7, -0.7],
            "mu": [0.0, 0.0, 0.0],
        },
        index=idx[-3:],
    )

    monkeypatch.setattr("investment_lab.heston_ukf._build_ukf_core", lambda *args, **kwargs: object())
    monkeypatch.setattr(ukf, "_update_core_functions", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ukf,
        "_step",
        lambda _core, params, _r: {
            "v_hat": params.theta,
            "sigma_hat": np.sqrt(params.theta),
            "expected_return": 0.0,
            "innovation": 0.0,
            "innovation_var": 1.0,
            "std_innovation": 0.0,
            "kalman_gain": 0.0,
            "loglik": 0.0,
        },
    )

    v_hat = ukf.filter(returns)

    assert v_hat.index.equals(idx[-3:])
    np.testing.assert_allclose(v_hat.values, [0.04, 0.05, 0.06])


def test_implied_realized_spread_reindexes_on_sigma_hat():
    idx_hat = pd.bdate_range("2024-01-03", periods=3)
    idx_iv = pd.bdate_range("2024-01-01", periods=5)

    ukf = HestonUKF()
    ukf._v_filtered = pd.Series([0.04, 0.09, 0.16], index=idx_hat, name="v_hat")
    sigma_iv = pd.Series([0.30, 0.31, 0.32, 0.33, 0.34], index=idx_iv)

    spread = ukf.implied_realized_spread(sigma_iv)

    expected = sigma_iv.reindex(idx_hat) - np.sqrt(ukf._v_filtered)
    assert spread.index.equals(idx_hat)
    pd.testing.assert_series_equal(spread, expected.rename("iv_rv_spread"))


def test_build_timing_positions_propagates_signal_lag(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=6)
    log_returns = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
    sigma_iv = pd.Series(0.25, index=idx)
    df_positions = pd.DataFrame({"date": idx, "weight": [1.0] * len(idx), "mid": [1.0] * len(idx)})

    captured = {}

    def fake_fit(self, log_returns, window=252, use_cache=True, save_every=10):
        self._params = HestonParams()
        self._rolling_params = pd.DataFrame(
            {
                "kappa": [1.0] * len(log_returns[1:]),
                "theta": [0.04] * len(log_returns[1:]),
                "xi": [0.3] * len(log_returns[1:]),
                "rho": [-0.7] * len(log_returns[1:]),
                "mu": [0.0] * len(log_returns[1:]),
            },
            index=log_returns.index[1:],
        )
        return self

    def fake_filter(self, log_returns):
        self._v_filtered = pd.Series(0.04, index=log_returns.index[1:], name="v_hat")
        return self._v_filtered

    def fake_compute_signal(self, spread):
        return pd.Series(1.25, index=spread.index, name="timing_signal")

    def fake_apply_timing(self, df_positions, signal, lag_bdays=1):
        captured["lag_bdays"] = lag_bdays
        out = df_positions.copy()
        out["weight"] = out["weight"] * 2.0
        return out

    monkeypatch.setattr(HestonUKF, "fit", fake_fit)
    monkeypatch.setattr(HestonUKF, "filter", fake_filter)
    monkeypatch.setattr(
        HestonUKF,
        "implied_realized_spread",
        lambda self, sigma_iv: pd.Series(0.1, index=self.sigma_hat.index, name="iv_rv_spread"),
    )
    monkeypatch.setattr(VolatilityTiming, "compute_signal", fake_compute_signal)
    monkeypatch.setattr(VolatilityTiming, "apply_timing", fake_apply_timing)

    df_timed, spread, signal = build_timing_positions(
        df_positions=df_positions,
        log_returns=log_returns,
        sigma_iv=sigma_iv,
        fit_window=3,
        scaling="linear",
        signal_lag_bdays=2,
    )

    assert captured["lag_bdays"] == 2
    assert spread.name == "iv_rv_spread"
    assert signal.name == "timing_signal"
    assert (df_timed["weight"] == 2.0).all()


def test_log_likelihood_uses_step_diagnostics(monkeypatch):
    ukf = HestonUKF(initial_params=HestonParams())
    params = HestonParams()
    sample = np.array([0.01, -0.02, 0.015, 0.005, -0.01])

    monkeypatch.setattr("investment_lab.heston_ukf._build_ukf_core", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ukf,
        "_step",
        lambda _core, _params, _r: {
            "v_hat": 0.04,
            "sigma_hat": 0.2,
            "expected_return": 0.0,
            "innovation": 0.0,
            "innovation_var": 1.0,
            "std_innovation": 0.0,
            "kalman_gain": 0.0,
            "loglik": -1.5,
        },
    )

    ll = ukf._log_likelihood(params, sample)
    assert ll == pytest.approx(-1.5 * len(sample))


def test_fit_stores_rolling_window_diagnostics(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    class DummyResult:
        success = True
        fun = 1.0
        status = 0
        message = "ok"
        nfev = 3
        nit = 2
        x = np.array([1.5, 0.05, 0.25, -0.6, 0.01])

    monkeypatch.setattr(
        "investment_lab.heston_ukf.minimize",
        lambda *args, **kwargs: DummyResult(),
    )
    monkeypatch.setattr(
        HestonUKF,
        "_log_likelihood",
        lambda self, params, sample: float(np.sum(sample)),
    )

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=None)
    ukf.fit(returns, window=3, use_cache=False)

    diag = ukf.fit_diagnostics
    assert len(diag) == len(returns) - 3
    assert {"window_start", "window_end", "window_size", "start_loglik", "final_loglik"}.issubset(diag.columns)
    assert (diag["window_size"] == 3).all()
    assert diag.index.min() == idx[3]
    assert diag.loc[idx[3], "window_start"] == idx[0]
    assert diag.loc[idx[3], "window_end"] == idx[2]
    assert bool(diag.loc[idx[3], "optimizer_success"]) is True


def test_fit_restores_fully_from_parquet_artifact(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=tmp_path)
    cache_path = ukf._cache_path(returns, window=3)
    assert cache_path is not None

    artifact = pd.DataFrame(
        {
            "date": idx[3:],
            "kappa": [1.1] * 5,
            "theta": [0.04] * 5,
            "xi": [0.3] * 5,
            "rho": [-0.7] * 5,
            "mu": [0.01] * 5,
            "window_start": idx[:5],
            "window_end": idx[2:7],
            "window_size": [3] * 5,
            "start_loglik": [-10.0] * 5,
            "final_loglik": [-8.0] * 5,
            "loglik_improvement": [2.0] * 5,
            "feller_violation": [0.0] * 5,
            "objective_value": [8.0] * 5,
            "optimizer_success": [True] * 5,
            "optimizer_status": [0] * 5,
            "optimizer_message": ["ok"] * 5,
            "nfev": [3] * 5,
            "nit": [2] * 5,
        }
    )
    artifact.to_parquet(cache_path, index=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("minimize should not be called when the parquet artifact is complete")

    monkeypatch.setattr("investment_lab.heston_ukf.minimize", fail_if_called)

    ukf.fit(returns, window=3, use_cache=True)

    assert ukf.rolling_params.index.equals(idx[3:])
    assert ukf.fit_diagnostics.index.equals(idx[3:])
    assert ukf.params.kappa == pytest.approx(1.1)


def test_cache_path_is_human_readable_and_stable():
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=Path("/tmp"))
    cache_path = ukf._cache_path(returns, window=3)

    assert cache_path is not None
    assert cache_path.name.startswith("rolling_window3_n5_20240101_20240105_")
    assert cache_path.suffix == ".parquet"


def test_cache_path_can_include_artifact_label():
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=Path("/tmp"), artifact_label="SPY")
    cache_path = ukf._cache_path(returns, window=3)

    assert cache_path is not None
    assert cache_path.name.startswith("spy_rolling_window3_n5_20240101_20240105_")
