"""
Tests unitaires pour le pipeline HestonUKF.

Tous les tests utilisent les vraies implémentations sur de petites séries
synthétiques — pas de monkeypatching.

Couverture :
    - VolatilityTiming.compute_signal  : bornage et index du signal
    - VolatilityTiming.apply_timing    : lag d'exécution et valeur neutre
    - HestonUKF.filter                 : alignement sur l'index rolling
    - HestonUKF.implied_realized_spread: réindexation et formule s_t
    - HestonUKF._log_likelihood        : retourne un float fini pour des params valides
    - HestonUKF.fit                    : structure et contenu des diagnostics rolling
    - build_timing_positions           : lag propagé jusqu'aux poids finaux
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from investment_lab.stochastic.heston_ukf import HestonUKF
from investment_lab.stochastic.heston import HestonParams
from investment_lab.metrics.signal import VolatilityTiming
from investment_lab.timing import build_timing_positions

RNG = np.random.default_rng(42)


def test_compute_signal_linear_is_bounded_and_keeps_index():
    """Le signal linéaire doit rester dans [-max_leverage, +max_leverage] et conserver l'index."""
    idx    = pd.bdate_range("2024-01-01", periods=8)
    spread = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02, -0.04, 0.08, -0.10], index=idx)

    timer  = VolatilityTiming(scaling="linear", lookback=5, max_leverage=1.5)
    signal = timer.compute_signal(spread)

    assert signal.index.equals(idx)
    assert signal.iloc[:4].isna().all()        # pas assez d'historique avant lookback
    assert signal.dropna().abs().max() <= 1.5  # signal toujours borné


def test_apply_timing_uses_business_day_lag_and_neutral_before_first_execution():
    """Le signal doit être décalé de lag_bdays jours ouvrés. Les dates sans signal reçoivent un poids neutre (1.0)."""
    idx          = pd.bdate_range("2024-01-01", periods=5)
    df_positions = pd.DataFrame({"date": idx, "weight": [1.0] * len(idx), "mid": [1.0] * len(idx)})
    signal       = pd.Series([0.5, 2.0], index=idx[:2], name="timing_signal")

    timer = VolatilityTiming()
    timed = timer.apply_timing(df_positions, signal, lag_bdays=1)

    # signal[0]=0.5 appliqué en j+1, signal[1]=2.0 appliqué en j+2, j+3 et j+4 neutres
    expected = pd.Series(
        [1.0, 0.5, 2.0, 1.0, 1.0],
        index=pd.Index(idx, name="date"),
        name="weight",
    )
    pd.testing.assert_series_equal(timed.set_index("date")["weight"], expected, check_freq=False)


def test_filter_with_rolling_params_returns_series_aligned_on_rolling_index():
    """filter() doit retourner une série alignée sur l'index de rolling_params, pas sur l'index complet des returns."""
    idx     = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.Series(RNG.normal(0, 0.01, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=None)
    ukf._params = HestonParams()
    # On injecte 5 dates de paramètres rolling (les 5 dernières de la série)
    ukf._rolling_params = pd.DataFrame(
        {"kappa": [2.0]*5, "theta": [0.04]*5, "xi": [0.3]*5, "rho": [-0.7]*5, "mu": [0.0]*5},
        index=idx[-5:],
    )

    v_hat = ukf.filter(returns)

    assert v_hat.index.equals(idx[-5:])  # aligné sur rolling_params, pas sur returns
    assert (v_hat > 0).all()             # la variance reste strictement positive


def test_implied_realized_spread_reindexes_on_sigma_hat():
    """implied_realized_spread doit réindexer sigma_iv sur l'index de sigma_hat avant de calculer s_t = σ_IV − σ̂."""
    idx_hat = pd.bdate_range("2024-01-03", periods=3)
    idx_iv  = pd.bdate_range("2024-01-01", periods=5)  # plage plus large que sigma_hat

    ukf = HestonUKF()
    ukf._v_filtered = pd.Series([0.04, 0.09, 0.16], index=idx_hat, name="v_hat")
    sigma_iv = pd.Series([0.30, 0.31, 0.32, 0.33, 0.34], index=idx_iv)

    spread   = ukf.implied_realized_spread(sigma_iv)
    expected = sigma_iv.reindex(idx_hat) - np.sqrt(ukf._v_filtered)

    assert spread.index.equals(idx_hat)
    pd.testing.assert_series_equal(spread, expected.rename("iv_rv_spread"))


def test_log_likelihood_returns_finite_for_valid_params():
    """_log_likelihood doit retourner un float fini (et supérieur à -inf) pour des paramètres valides."""
    ukf    = HestonUKF(initial_params=HestonParams())
    params = HestonParams()
    sample = RNG.normal(0, 0.01, 30)

    ll = ukf._log_likelihood(params, sample)

    # La log-vraisemblance gaussienne peut être positive quand S_t << 1
    # (le terme −½ log(2π S_t) devient grand positif). On vérifie seulement
    # qu'elle est finie et strictement supérieure à la valeur de rejet (−inf).
    assert np.isfinite(ll)
    assert ll > -np.inf


def test_fit_stores_rolling_window_diagnostics():
    """fit() doit produire un diagnostic par date de calibration, avec les bonnes fenêtres temporelles."""
    idx     = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.Series(RNG.normal(0, 0.01, len(idx)), index=idx)

    ukf = HestonUKF(initial_params=HestonParams(), cache_dir=None)
    ukf.fit(returns, window=3, use_cache=False)

    diag = ukf.fit_diagnostics
    assert len(diag) == len(returns) - 3   # une ligne par date hors fenêtre initiale
    assert {"window_start", "window_end", "window_size", "start_loglik", "final_loglik"}.issubset(diag.columns)
    assert (diag["window_size"] == 3).all()
    # La première calibration (idx[3]) utilise la fenêtre [idx[0], idx[2]]
    assert diag.index.min() == idx[3]
    assert diag.loc[idx[3], "window_start"] == idx[0]
    assert diag.loc[idx[3], "window_end"]   == idx[2]


def test_build_timing_positions_lag_shifts_weights():
    """Les poids de la stratégie timée ne doivent changer qu'à partir de lag+1 jours après le premier signal."""
    idx          = pd.bdate_range("2024-01-01", periods=12)
    returns      = pd.Series(RNG.normal(0, 0.01, len(idx)), index=idx)
    sigma_iv     = pd.Series(0.20, index=idx)
    df_positions = pd.DataFrame({"date": idx, "weight": [1.0] * len(idx), "mid": [1.0] * len(idx)})

    df_timed, spread, signal = build_timing_positions(
        df_positions=df_positions,
        log_returns=returns,
        sigma_iv=sigma_iv,
        fit_window=5,
        scaling="linear",
        signal_lag_bdays=1,
    )

    # Le signal n'est disponible qu'après fit_window jours, exécuté au jour suivant.
    # Avant la première date d'exécution, les poids doivent être identiques à la base.
    first_signal_date = signal.dropna().index.min()
    first_exec_date   = first_signal_date + pd.offsets.BDay(1)

    base_weights  = df_positions.set_index("date")["weight"]
    timed_weights = df_timed.set_index("date")["weight"]

    pre_exec_dates = timed_weights.index[timed_weights.index < first_exec_date]
    pd.testing.assert_series_equal(
        timed_weights.loc[pre_exec_dates],
        base_weights.loc[pre_exec_dates],
        check_names=False,
    )
