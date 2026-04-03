from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from investment_lab.stochastic.heston import HestonParams
from investment_lab.stochastic.heston_ukf import HestonUKF
from investment_lab.metrics.signal import VolatilityTiming

# =============================================================================
# build_timing_positions — helper end-to-end
# =============================================================================

def build_timing_positions(
    df_positions: pd.DataFrame,
    log_returns: pd.Series,
    sigma_iv: pd.Series,
    fit_window: int = 252,
    scaling: str = "linear",
    lookback: int = 63,
    max_leverage: float = 2.0,
    threshold: float = 0.02,
    signal_lag_bdays: int = 1,
    dt: float = 1.0 / 252.0,
    initial_params: Optional[HestonParams] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline complet : calibration UKF → spread → timing → positions dynamiques.

    Enchaîne les 5 étapes du workflow dans l'ordre correct :
        1. HestonUKF.fit()                        calibration MLE roulante
        2. HestonUKF.filter()                     estimation de v̂_t
        3. HestonUKF.implied_realized_spread()    calcul de s_t = σ_IV − σ̂_t
        4. VolatilityTiming.compute_signal()      normalisation du signal
        5. VolatilityTiming.apply_timing()        application aux poids avec lag d'exécution

    Paramètres
    ----------
    df_positions    : pd.DataFrame  Positions de base (sortie de generate_trades).
    log_returns     : pd.Series     Log-returns journaliers du sous-jacent.
    sigma_iv        : pd.Series     Volatilité implicite ATM annualisée.
    fit_window      : int           Fenêtre MLE roulante (défaut 252 = 1 an).
    scaling         : str           Mode de timing : "linear", "rank", "threshold".
    lookback        : int           Fenêtre de normalisation du signal (défaut 63).
    max_leverage    : float         Multiplicateur maximum |w_t / w_base| (défaut 2.0).
    signal_lag_bdays: int           Lag d'exécution du signal (défaut 1 jour ouvré).
    dt              : float         Pas de temps en années (défaut 1/252).
    initial_params  : HestonParams  Point de départ MLE (défaut : valeurs typiques equity).

    Retourne
    --------
    df_timed_positions : pd.DataFrame  Positions avec poids dynamiques.
    spread             : pd.Series     s_t = σ_IV,t − σ̂_t.
    signal             : pd.Series     Signal de timing normalisé.

    Exemple d'usage dans un notebook
    ---------------------------------
    >>> from investment_lab.heston_ukf import build_timing_positions
    >>> from investment_lab.metrics.util import levels_to_returns
    >>>
    >>> log_returns = levels_to_returns(df_spot["spot"])
    >>> sigma_iv    = df_atm["implied_volatility"]
    >>>
    >>> df_timed, spread, signal = build_timing_positions(
    ...     df_positions = df_base,
    ...     log_returns  = log_returns,
    ...     sigma_iv     = sigma_iv,
    ...     fit_window   = 252,
    ...     scaling      = "linear",
    ... )
    >>> backtest_timed = StrategyBacktester(df_timed).compute_backtest()
    >>> backtest_base  = StrategyBacktester(df_base).compute_backtest()
    >>>
    >>> # Comparaison NAV
    >>> backtest_base.nav["NAV"].plot(label="Base")
    >>> backtest_timed.nav["NAV"].plot(label="Timed (UKF)")
    """
    logging.info("Démarrage du pipeline HestonUKF timing.")

    # Étapes 1 & 2 : calibration des paramètres puis filtrage de v̂_t
    ukf = HestonUKF(initial_params=initial_params, dt=dt)
    ukf.fit(log_returns, window=fit_window)
    ukf.filter(log_returns)

    # Étape 3 : spread  s_t = σ_IV,t − σ̂_t
    spread = ukf.implied_realized_spread(sigma_iv)

    # Étapes 4 & 5 : normalisation du signal et application aux poids
    timer = VolatilityTiming(
        scaling=scaling,
        lookback=lookback,
        max_leverage=max_leverage,
        threshold=threshold, 
    )
    signal   = timer.compute_signal(spread)
    df_timed = timer.apply_timing(df_positions, signal, lag_bdays=signal_lag_bdays)

    return df_timed, spread, signal
