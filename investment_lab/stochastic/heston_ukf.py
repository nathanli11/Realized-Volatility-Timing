"""
heston_ukf.py
=============
Unscented Kalman Filter (UKF) appliqué au modèle de Heston (1993).

Modèle continu (slide)
-----------------------
    dS_t  = μ S_t dt  +  S_t √v_t  dW_{1,t}
    dv_t  = κ(θ - v_t) dt  +  ξ √v_t  dW_{2,t}
    dW_{1,t} · dW_{2,t} = rho dt

Discrétisation Euler-Maruyama (pas de temps Δt = 1/252)
---------------------------------------------------------
Équation d'état  (variance latente v_t) :
    v_{t+1} = v_t + κ(θ - v_t)Δt  +  ξ√v_t √Δt · ε_t^(2)
    Bruit de processus :  Q_t = ξ^2 v_t Δt

Équation d'observation  (log-return r_t via formule d'Itô sur log S_t) :
    r_t = (μ - v_t/2) Δt  +  √(v_t Δt) · ε_t^(1)
    Bruit de mesure :  R_t = v_t Δt

Corrélation des bruits (terme ρ — spécifique à Heston)
-------------------------------------------------------
    Cov(ε_t^(1), ε_t^(2)) = ρ
    ⟹ Cov(v_{t+1} − v̄, r_t − r̄) = ρ · ξ · v_t · Δt

    filterpy calcule P_{xz} uniquement depuis les sigma-points,
    ce qui donne P_{xz} ≈ 0 car les bruits ne sont pas représentés.
    HestonUKFCore surcharge update() pour injecter ce terme manquant.

Pipeline
--------
    1. HestonUKF.fit(log_returns, window)         → calibration MLE roulante
    2. HestonUKF.filter(log_returns)              → v̂_t (variance estimée)
    3. HestonUKF.implied_realized_spread(σ_IV)    → s_t = rho_IV,t - sigma_t
    4. VolatilityTiming.compute_signal(s_t)       → signal d'allocation
    5. VolatilityTiming.apply_timing(positions)   → poids dynamiques

API publique
------------
    HestonParams           dataclass  (κ, θ, ξ, ρ, μ)
    HestonUKFCore          sous-classe filterpy avec correction ρ dans update()
    HestonUKF              fit / filter / spread
    VolatilityTiming       allocation dynamique depuis s_t
    build_timing_positions helper end-to-end

Dépendances
-----------
    filterpy >= 1.4   pip install filterpy
    scipy, numpy, pandas
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.stochastic.heston import HestonParams
from investment_lab.stochastic.kalman import HestonUKFCore, build_ukf_core

# =============================================================================
# HestonUKF — classe principale : fit / filter / spread
# =============================================================================

class HestonUKF:
    """UKF sur la dynamique de Heston avec correction exacte de ρ.

    Workflow
    --------
    1. fit(log_returns, window=252)
       Calibre (κ, θ, ξ, ρ, μ) par MLE sur la fenêtre roulante spécifiée.
       Utilise L-BFGS-B avec une pénalité douce sur la condition de Feller.

    2. filter(log_returns)
       Estime v̂_t via le filtre UKF avec les paramètres calibrés.
       La correction ρ·ξ·v_t·Δt est injectée à chaque pas dans P_{xz}.

    3. implied_realized_spread(sigma_iv)
       Calcule s_t = σ_IV,t − σ̂_t  où  σ̂_t = √v̂_t.
       s_t > 0 : vol implicite > vol réalisée estimée → carry positif.

    Paramètres
    ----------
    initial_params : HestonParams   Point de départ pour l'optimiseur MLE.
    dt             : float          Pas de temps en années (défaut 1/252).
    """

    def __init__(
        self,
        initial_params: Optional[HestonParams] = None,
        dt: float = 1.0 / 252.0,
        cache_dir: Optional[str | Path] = ".cache/heston_ukf",
        optimizer_maxiter: int = 300,
    ) -> None:
        # Point de départ de l'optimiseur (valeurs typiques equity si non fourni)
        self.initial_params = initial_params or HestonParams()
        # Pas de temps : 1/252 pour des données journalières
        self.dt = dt
        # Répertoire de cache pour les calibrations rolling
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Nombre maximal d'itérations pour L-BFGS-B
        self.optimizer_maxiter = optimizer_maxiter
        # Paramètres calibrés (None tant que fit() n'a pas été appelé)
        self._params: Optional[HestonParams] = None
        # Historique des paramètres calibrés date par date
        self._rolling_params: Optional[pd.DataFrame] = None
        # Diagnostics complets de calibration rolling
        self._fit_diagnostics: Optional[pd.DataFrame] = None
        # Série temporelle de la variance filtrée v̂_t (None avant filter())
        self._v_filtered: Optional[pd.Series] = None
        # DataFrame complet des diagnostics du filtre (v̂_t, innovation, gain, loglik) après filter_with_diagnostics()
        self._filter_diagnostics: Optional[pd.DataFrame] = None

    @staticmethod
    def _params_to_record(params: HestonParams) -> dict[str, float]:
        """Sérialise HestonParams pour stockage dans un DataFrame."""
        return {
            "kappa": params.kappa,
            "theta": params.theta,
            "xi": params.xi,
            "rho": params.rho,
            "mu": params.mu,
        }

    @staticmethod
    def _record_to_params(record: pd.Series) -> HestonParams:
        """Reconstruit HestonParams depuis une ligne de calibration rolling."""
        return HestonParams(
            kappa=float(record["kappa"]),
            theta=float(record["theta"]),
            xi=float(record["xi"]),
            rho=float(record["rho"]),
            mu=float(record["mu"]),
        )

    def _update_core_functions(self, ukf: HestonUKFCore, params: HestonParams) -> None:
        """Met à jour les fonctions du filtre pour refléter les paramètres du jour."""

        def fx(v: np.ndarray, dt: float) -> np.ndarray:
            v_val = max(float(v[0]), 1e-8)
            v_next = v_val + params.kappa * (params.theta - v_val) * dt
            return np.array([max(v_next, 1e-8)])

        def hx(v: np.ndarray) -> np.ndarray:
            v_val = max(float(v[0]), 1e-8)
            return np.array([(params.mu - 0.5 * v_val) * self.dt])

        ukf.fx = fx
        ukf.hx = hx

    def _cache_key(self, returns: pd.Series, window: int) -> str:
        """Construit une clé de cache déterministe pour une calibration rolling."""
        hasher = hashlib.sha256()
        hasher.update(str(window).encode("utf-8"))
        hasher.update(str(self.dt).encode("utf-8"))
        hasher.update(str(self.optimizer_maxiter).encode("utf-8"))
        hasher.update(np.asarray(self.initial_params.to_array(), dtype=np.float64).tobytes())
        hasher.update(np.asarray(returns.index.view("i8"), dtype=np.int64).tobytes())
        hasher.update(np.asarray(returns.values, dtype=np.float64).tobytes())
        return hasher.hexdigest()

    def _cache_path(self, returns: pd.Series, window: int) -> Optional[Path]:
        """Retourne le chemin du cache rolling pour la série fournie."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"rolling_{self._cache_key(returns, window)}.pkl"

    @staticmethod
    def _cache_columns() -> list[str]:
        """Colonnes de paramètres attendues dans le cache rolling."""
        return ["kappa", "theta", "xi", "rho", "mu"]

    # ------------------------------------------------------------------
    # Méthode interne : un pas predict-update avec mise à jour des matrices
    # ------------------------------------------------------------------

    def _step(
        self, ukf: HestonUKFCore, params: HestonParams, r: float
    ) -> dict[str, float]:
        """Exécute un cycle predict → update et retourne les diagnostics du pas (v̂_t, contribution LL).

        Les matrices Q, R et la correction ρ doivent être mises à jour à chaque
        pas car elles dépendent de v_t (bruit multiplicatif dans Heston).

        Paramètres
        ----------
        ukf    : HestonUKFCore  Filtre en cours d'exécution.
        params : HestonParams   Paramètres courants.
        r      : float          Log-return observé à ce pas.

        Retourne
        --------
        v_updated : float  Variance filtrée après mise à jour.
        innov_ll  : float  Contribution à la log-vraisemblance (innovation).
        """
        # Variance courante (avant predict) — clampée pour stabilité numérique
        v_pred = max(float(ukf.x[0]), 1e-6)

        # Mise à jour du bruit de processus Q_t = ξ² v_t Δt
        # (bruit multiplicatif : Q dépend de l'état courant v_t)
        ukf.Q = np.array([[params.xi ** 2 * v_pred * self.dt]])

        # Mise à jour du bruit de mesure R_t = v_t Δt
        # (hétéroscédasticité : la variance du return dépend de v_t)
        ukf.R = np.array([[v_pred * self.dt]])

        # Mise à jour de la correction de cross-covariance ρ·ξ·v_t·Δt
        # Ce terme sera injecté dans P_{xz} lors du update() ci-dessous
        ukf._rho_xi_vt_dt = params.rho * params.xi * v_pred * self.dt

        # Étape de prédiction : propage les sigma-points à travers fx
        # Met à jour x_{t|t-1} et P_{t|t-1} via la transformée non-parfumée
        ukf.predict()

        # Calcul de l'innovation ν_t = r_t − E[r_t | F_{t-1}]
        # On utilise l'état prédit après predict() pour être cohérent
        v_after_predict = max(float(ukf.x[0]), 1e-8)
        expected_r = (params.mu - 0.5 * v_after_predict) * self.dt

        # Étape de mise à jour : intègre l'observation r_t
        # Appelle HestonUKFCore.update() qui injecte la correction ρ dans P_{xz}
        ukf.update(np.array([r]))

        # Diagnostics venant du filtre
        innovation = float(ukf._innovation)
        innovation_var = float(ukf.S[0, 0]) if ukf.S is not None else v_pred * self.dt
        innovation_var = max(innovation_var, 1e-6)
        std_innovation = innovation / np.sqrt(innovation_var)
        kalman_gain = float(ukf.K[0, 0]) if ukf.K is not None else np.nan

        # Contribution à la log-vraisemblance (décomposition en innovations) :
        # log p(r_t | F_{t-1}) = −½ [log(2π S_t) + ν_t² / S_t]
        innov_ll = -0.5 * (np.log(2.0 * np.pi * innovation_var) + innovation ** 2 / innovation_var)

        v_updated = max(float(ukf.x[0]), 1e-6)
        sigma_updated = np.sqrt(v_updated)
        return {
            "v_hat": v_updated,
            "sigma_hat": sigma_updated,
            "expected_return": expected_r,
            "innovation": innovation,
            "innovation_var": innovation_var,
            "std_innovation": std_innovation,
            "kalman_gain": kalman_gain,
            "loglik": float(innov_ll),
        }

    # ------------------------------------------------------------------
    # Log-vraisemblance (décomposition en innovations, UKF)
    # ------------------------------------------------------------------

    def _log_likelihood(self, params: HestonParams, log_returns: np.ndarray) -> float:
        """Log-vraisemblance via la décomposition en innovations du UKF.

        Formule (prediction-error decomposition) :
            log p(r_{1:T} | params) = −½ Σ_t [ log(2π S_t) + ν_t² / S_t ]

        où ν_t = r_t − E[r_t | F_{t-1}] est l'innovation du filtre
        et S_t = Var(r_t | F_{t-1}) est sa variance (covariance d'innovation).

        La correction ρ est active via HestonUKFCore dans chaque _step().

        Paramètres
        ----------
        params      : HestonParams   Paramètres à évaluer.
        log_returns : np.ndarray     Série des log-returns (fenêtre roulante).
        """
        if len(log_returns) < 5:
            return -np.inf

        # Initialiser le filtre avec theta comme variance de départ
        v0 = max(params.theta, 1e-6)
        ukf = build_ukf_core(params, self.dt, v0)
        ll = 0.0

        try:
            for r in log_returns:
                step_diag = self._step(ukf, params, float(r))
                ll += float(step_diag["loglik"])
        except Exception:
            return -np.inf

        return ll if np.isfinite(ll) else -np.inf

    # ------------------------------------------------------------------
    # fit() — calibration MLE roulante
    # ------------------------------------------------------------------

    def fit(
        self,
        log_returns: pd.Series,
        window: int = 252,
        use_cache: bool = True,
        save_every: int = 10,
        refit_every: int = 1,
    ) -> "HestonUKF":
        """Calibre les paramètres de Heston par MLE sur la fenêtre roulante.

        On maximise log p(r_{t-W+1:t} | κ, θ, ξ, ρ, μ) via L-BFGS-B.
        Une pénalité douce assure que la condition de Feller (2κθ > ξ²) est
        approximativement respectée, garantissant v_t > 0 p.s.

        Paramètres
        ----------
        log_returns : pd.Series  Log-returns journaliers (index = dates).
        window      : int        Taille de la fenêtre roulante (défaut 252 = 1 an).
        use_cache   : bool       Active le chargement / la sauvegarde sur disque.
        save_every  : int        Fréquence de checkpoint du cache partiel.
        refit_every : int        Fréquence de recalibration MLE en jours ouvrés.
                                 refit_every=1 (défaut) → recalibration quotidienne.
                                 refit_every=5 → recalibration hebdomadaire (5× plus rapide).
                                 Entre deux recalibrations, les paramètres précédents
                                 sont conservés (warm start naturel).

        Retourne
        --------
        self  (chaînage de méthodes)
        """
        logging.info(
            "Fitting HestonUKF : n=%d observations, window=%d.", len(log_returns), window
        )
        returns = log_returns.dropna()
        n = len(returns)

        if n < 6:
            raise ValueError("Au moins 6 observations sont nécessaires pour la calibration.")

        # Garder au moins une observation hors fenêtre pour filtrer sans look-ahead.
        if n <= window:
            new_window = n - 1
            logging.warning(
                "Série trop courte (%d <= %d). Fenêtre réduite à %d pour garder "
                "une observation hors-fenêtre.",
                n,
                window,
                new_window,
            )
            window = new_window

        cache_path = self._cache_path(returns, window)
        if use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        rolling_df: Optional[pd.DataFrame] = None
        if use_cache and cache_path is not None and cache_path.exists():
            try:
                loaded = pd.read_pickle(cache_path)
                if isinstance(loaded, pd.DataFrame) and all(
                    col in loaded.columns for col in self._cache_columns()
                ):
                    rolling_df = loaded.sort_index()
                    logging.info(
                        "Cache rolling chargé depuis %s (%d lignes).",
                        cache_path,
                        len(rolling_df),
                    )
            except Exception as exc:
                logging.warning("Impossible de charger le cache rolling %s (%s).", cache_path, exc)

        x0 = self.initial_params.to_array()
        rolling_records: list[dict[str, float | pd.Timestamp]] = []
        diagnostics_records: list[dict[str, object]] = []
        start_end = window

        if rolling_df is not None and len(rolling_df) > 0:
            valid_index = rolling_df.index.intersection(returns.index[window:])
            if len(valid_index) > 0:
                rolling_df = rolling_df.loc[valid_index]
                rolling_records = [
                    {"date": idx, **self._params_to_record(self._record_to_params(rolling_df.loc[idx]))}
                    for idx in rolling_df.index
                ]
                diagnostic_cols = [
                    "window_start",
                    "window_end",
                    "window_size",
                    "start_loglik",
                    "final_loglik",
                    "loglik_improvement",
                    "feller_violation",
                    "objective_value",
                    "optimizer_success",
                    "optimizer_status",
                    "optimizer_message",
                    "nfev",
                    "nit",
                ]
                available_diag_cols = [col for col in diagnostic_cols if col in rolling_df.columns]
                if available_diag_cols:
                    diagnostics_records = [
                        {"date": idx, **rolling_df.loc[idx, available_diag_cols].to_dict()}
                        for idx in rolling_df.index
                    ]
                last_cached_date = rolling_df.index[-1]
                last_cached_pos = returns.index.get_loc(last_cached_date)
                start_end = int(last_cached_pos) + 1
                x0 = rolling_df.iloc[-1][self._cache_columns()].to_numpy(dtype=np.float64)
                logging.info(
                    "Reprise de la calibration rolling après %s.",
                    last_cached_date,
                )

        if start_end >= n and rolling_df is not None and len(rolling_df) == n - window:
            self._rolling_params = rolling_df[self._cache_columns()].copy()
            diagnostic_cols = [
                "window_start",
                "window_end",
                "window_size",
                "start_loglik",
                "final_loglik",
                "loglik_improvement",
                "feller_violation",
                "objective_value",
                "optimizer_success",
                "optimizer_status",
                "optimizer_message",
                "nfev",
                "nit",
            ]
            available_diag_cols = [col for col in diagnostic_cols if col in rolling_df.columns]
            self._fit_diagnostics = (
                rolling_df[available_diag_cols].copy() if available_diag_cols else None
            )
            self._params = self._record_to_params(self._rolling_params.iloc[-1])
            logging.info("Calibration rolling entièrement restaurée depuis le cache.")
            return self

        for end in range(start_end, n):
            fit_date = returns.index[end]
            sample_series = returns.iloc[end - window:end]
            sample = sample_series.values
            sample_start_date = sample_series.index[0]
            sample_end_date = sample_series.index[-1]

            # refit_every : on ne relance l'optimiseur que tous les refit_every jours.
            # Les jours intermédiaires réutilisent les paramètres précédents (x0 inchangé).
            # Cela réduit le temps de calcul d'un facteur refit_every sans perte significative
            # car les paramètres Heston varient lentement d'un jour à l'autre.
            steps_since_start = end - start_end
            should_refit = (refit_every <= 1) or (steps_since_start % refit_every == 0)

            x_start = x0.copy()
            start_params = HestonParams.from_array(x_start)
            start_loglik = self._log_likelihood(start_params, sample)
            start_feller_violation = max(
                0.0, start_params.xi ** 2 - 2.0 * start_params.kappa * start_params.theta
            )
            start_obj = -start_loglik + 1e4 * start_feller_violation

            if should_refit:
                def neg_ll(x: np.ndarray) -> float:
                    p = HestonParams.from_array(x)
                    feller_violation = max(0.0, p.xi ** 2 - 2.0 * p.kappa * p.theta)
                    penalty = 1e4 * feller_violation
                    return -self._log_likelihood(p, sample) + penalty

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        neg_ll,
                        x_start,
                        method="L-BFGS-B",
                        bounds=HestonParams.bounds(),
                        options={"maxiter": self.optimizer_maxiter, "ftol": 1e-10, "gtol": 1e-7},
                    )

                if result.success or (result.fun < start_obj):
                    fitted = HestonParams.from_array(result.x)
                    x0 = result.x
                else:
                    logging.warning(
                        "MLE non convergé au %s (%s). Paramètres précédents conservés.",
                        fit_date,
                        result.message,
                    )
                    fitted = HestonParams.from_array(x_start)
            else:
                # Pas de recalibration ce jour : on réutilise les paramètres précédents.
                # On crée un faux résultat pour les diagnostics.
                fitted = HestonParams.from_array(x_start)
                result = type("_FakeResult", (), {
                    "success": True, "fun": -start_loglik,
                    "status": 0, "message": "skipped (refit_every)",
                    "nfev": 0, "nit": 0,
                })()

            final_loglik = self._log_likelihood(fitted, sample)
            feller_violation = max(0.0, fitted.xi ** 2 - 2.0 * fitted.kappa * fitted.theta)

            row = {"date": fit_date}
            row.update(self._params_to_record(fitted))
            rolling_records.append(row)
            diagnostics_records.append(
                {
                    "date": fit_date,
                    # La calibration à la date t repose sur la fenêtre passée [t-window, t).
                    "window_start": sample_start_date,
                    "window_end": sample_end_date,
                    "window_size": int(len(sample)),
                    "start_loglik": float(start_loglik),
                    "final_loglik": float(final_loglik),
                    "loglik_improvement": float(final_loglik - start_loglik),
                    "feller_violation": float(feller_violation),
                    "objective_value": float(result.fun if np.isfinite(result.fun) else np.nan),
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(getattr(result, "nfev", -1)),
                    "nit": int(getattr(result, "nit", -1)),
                }
            )

            should_checkpoint = (
                use_cache
                and cache_path is not None
                and (
                    len(rolling_records) % max(save_every, 1) == 0
                    or end == n - 1
                )
            )
            if should_checkpoint:
                rolling_params_df = pd.DataFrame(rolling_records).set_index("date")
                diagnostics_df = pd.DataFrame(diagnostics_records).set_index("date")
                rolling_params_df.join(diagnostics_df, how="left").to_pickle(cache_path)

        self._rolling_params = pd.DataFrame(rolling_records).set_index("date")
        self._fit_diagnostics = pd.DataFrame(diagnostics_records).set_index("date")
        if use_cache and cache_path is not None:
            self._rolling_params.join(self._fit_diagnostics, how="left").to_pickle(cache_path)
        self._params = self._record_to_params(self._rolling_params.iloc[-1])
        logging.info(
            "Rolling MLE terminé : %d jeux de paramètres calibrés. Dernier jeu : "
            "kappa=%.3f theta=%.4f xi=%.3f rho=%.3f mu=%.4f",
            len(self._rolling_params),
            self._params.kappa,
            self._params.theta,
            self._params.xi,
            self._params.rho,
            self._params.mu,
        )

        return self

    # ---------------------------------------------------------------------
    # filter_with_diagnostics() — estimation de l'état v̂_t + diagnostic
    # ---------------------------------------------------------------------

    def filter_with_diagnostics(self, log_returns: pd.Series) -> pd.DataFrame:
        """ Estime la variance instantanée v̂_t par UKF sur toute la série.
        Filtre la série et retourne un DataFrame complet de diagnostics.

        À chaque pas, la correction ρ·ξ·v_t·Δt est injectée dans P_{xz}
        via HestonUKFCore.update(), donnant le gain de Kalman exact sous
        la dynamique de Heston avec Browniens corrélés.

        Paramètres
        ----------
        log_returns : pd.Series  Log-returns journaliers (index = dates).

        Retourne
        --------
        pd.DataFrame  Variance filtrée v̂_t + diagnostics
        """

        if self._params is None:
            raise RuntimeError("Appeler fit() avant filter_with_diagnostics().")

        returns = log_returns.dropna()

        if self._rolling_params is None:
            params = self._params
            v0 = max(params.theta, 1e-6)
            ukf = build_ukf_core(params, self.dt, v0)
            iter_dates = returns.index
        else:
            iter_dates = self._rolling_params.index.intersection(returns.index)
            if len(iter_dates) == 0:
                raise RuntimeError("Aucune date commune entre les paramètres rolling et les returns.")
            first_params = self._record_to_params(self._rolling_params.loc[iter_dates[0]])
            v0 = max(first_params.theta, 1e-6)
            ukf = build_ukf_core(first_params, self.dt, v0)

        rows = []
        for date in iter_dates:
            if self._rolling_params is not None:
                params = self._record_to_params(self._rolling_params.loc[date])
                self._update_core_functions(ukf, params)

            r = returns.loc[date]
            # _step() : predict → update avec correction ρ et matrices Q, R state-dependent
            out = self._step(ukf, params, float(r))
            out["date"] = date
            rows.append(out)

        df_diag = pd.DataFrame(rows).set_index("date")
        df_diag.index.name = iter_dates.name or "date"

        # Stocker le df filtré pour les propriétés dérivées (sigma_hat, spread)
        self._v_filtered = df_diag["v_hat"].rename("v_hat")
        self._filter_diagnostics = df_diag
        return df_diag

    # ------------------------------------------------------------------
    # filter() — estimation de l'état v̂_t
    # ------------------------------------------------------------------

    def filter(self, log_returns: pd.Series) -> pd.Series:
        """Estime la variance instantanée v̂_t par UKF sur toute la série.

        À chaque pas, la correction ρ·ξ·v_t·Δt est injectée dans P_{xz}
        via HestonUKFCore.update(), donnant le gain de Kalman exact sous
        la dynamique de Heston avec Browniens corrélés.

        Paramètres
        ----------
        log_returns : pd.Series  Log-returns journaliers (index = dates).

        Retourne
        --------
        pd.Series  Variance filtrée v̂_t (même index que log_returns).
        """
        df_diag = self.filter_with_diagnostics(log_returns)
        return df_diag["v_hat"].rename("v_hat")

    # ------------------------------------------------------------------
    # Propriétés dérivées
    # ------------------------------------------------------------------

    @property
    def sigma_hat(self) -> pd.Series:
        """Volatilité annualisée estimée : σ̂_t = √v̂_t.

        C'est la réalised volatility estimée par le filtre, à comparer
        à la volatilité implicite σ_IV pour construire le spread s_t.
        """
        if self._v_filtered is None:
            raise RuntimeError("Appeler filter() d'abord.")
        return np.sqrt(self._v_filtered).rename("sigma_hat")

    def implied_realized_spread(self, sigma_iv: pd.Series) -> pd.Series:
        """Calcule le spread IV-RV  s_t = σ_IV,t − σ̂_t.

        Interprétation du signal :
            s_t > 0  →  IV > RV estimée  →  vol "chère"       →  short vol profitable
            s_t < 0  →  IV < RV estimée  →  vol "bon marché"  →  réduire/fermer

        Paramètres
        ----------
        sigma_iv : pd.Series  Volatilité implicite ATM annualisée (même index).

        Retourne
        --------
        pd.Series  Spread s_t aligné sur l'index de sigma_hat.
        """
        # Réindexer sigma_iv sur l'index du filtre (gestion des jours manquants)
        spread = sigma_iv.reindex(self.sigma_hat.index) - self.sigma_hat
        spread.name = "iv_rv_spread"
        return spread

    @property
    def params(self) -> Optional[HestonParams]:
        """Paramètres calibrés par fit() (None si fit() non encore appelé)."""
        return self._params

    @property
    def rolling_params(self) -> pd.DataFrame:
        """Historique des paramètres calibrés date par date."""
        if self._rolling_params is None:
            raise RuntimeError("Appeler fit() d'abord.")
        return self._rolling_params.copy()

    @property
    def fit_diagnostics(self) -> pd.DataFrame:
        """Diagnostics complets de calibration rolling après fit()."""
        if self._fit_diagnostics is None:
            raise RuntimeError("Aucun diagnostic de calibration disponible. Appeler fit() d'abord.")
        return self._fit_diagnostics.copy()

    @property
    def filter_diagnostics(self) -> pd.DataFrame:
        """DataFrame complet des diagnostics du filtre après filter_with_diagnostics()."""
        if self._filter_diagnostics is None:
            raise RuntimeError("Appeler filter_with_diagnostics() d'abord.")
        return self._filter_diagnostics.copy()    

    # ------------------------------------------------------------------
    # forecast_variance() — forecast ponctuelle de variance
    # ------------------------------------------------------------------

    def forecast_variance(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """Prévision de variance conditionnelle E[v_{t+h} | v_t].
        
        Utilise la formule analytique de Heston pour la prévision de variance
        à horizon h (en jours): E[v_{t+h} | v_t] = θ+(v_t​−θ)e^−κu
        """
        p = params or self._params
        if p is None:
            raise RuntimeError("Paramètres non calibrés.")

        v_t = max(float(v_t), 1e-6)
        tau = horizon * self.dt
        v_forecast = p.theta + (v_t - p.theta) * np.exp(-p.kappa * tau)
        return max(float(v_forecast), 1e-6)

    # ------------------------------------------------------------------
    # forecast_average_variance() — forecast moyenne de variance
    # ------------------------------------------------------------------

    def forecast_average_variance(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """Prévision de variance moyenne sur l'horizon [t, t+h]."""
        p = params or self._params
        if p is None:
            raise RuntimeError("Paramètres non calibrés.")

        v_t = max(float(v_t), 1e-6)
        tau = horizon * self.dt

        if p.kappa < 1e-10:
            v_avg = v_t
        else:
            decay_term = (1.0 - np.exp(-p.kappa * tau)) / (p.kappa * tau)
            v_avg = p.theta + (v_t - p.theta) * decay_term
        return max(float(v_avg), 1e-6)

    # ----------------------------------------------------------------------------
    # forecast_average_volatility() — helper pour forecast moyenne de volatilité
    # ----------------------------------------------------------------------------

    def forecast_average_volatility(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """Prévision de volatilité annualisée moyenne sur horizon."""
        return np.sqrt(self.forecast_average_variance(v_t, horizon=horizon, params=params))






