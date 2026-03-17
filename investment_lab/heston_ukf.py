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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import dot
from numpy.linalg import inv
import pandas as pd
from scipy.optimize import minimize

try:
    # On utilise uniquement MerweScaledSigmaPoints — l'API publique stable de filterpy.
    # UnscentedKalmanFilter n'est pas importé pour éviter les attributs internes
    # instables (_UT, _num_sigmas, sigmas_f) qui varient selon la version installée.
    from filterpy.kalman import MerweScaledSigmaPoints
except ImportError as exc:
    raise ImportError("filterpy est requis : pip install filterpy") from exc



# HestonParams — dataclass des parametres
@dataclass
class HestonParams:
    """Parametres du modèle de Heston.

    kappa : float  Vitesse de retour à la moyenne de v_t.     
    theta : float  Variance long terme (moyenne de v_t).      
    xi    : float  Volatilité de la variance (vol-of-vol).    
    rho   : float  Corrélation dW_1 / dW_2.                 
    mu    : float  Drift du log-prix.                       
    """

    # De facon arbitraire
    kappa: float = 2.0
    theta: float = 0.04
    xi:    float = 0.3
    rho:   float = -0.7
    mu:    float = 0.0

    def feller_satisfied(self) -> bool:
        """Condition de Feller : 2κθ > ξ²  →  v_t reste strictement positif."""
        return 2.0 * self.kappa * self.theta > self.xi ** 2

    def to_array(self) -> np.ndarray:
        """Series des 5 paramètres en vecteur numpy pour l'optimiseur."""
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.mu])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        """Désérialise un vecteur numpy en HestonParams."""
        return cls(kappa=x[0], theta=x[1], xi=x[2], rho=x[3], mu=x[4])

    @staticmethod
    def bounds() -> list[tuple]:
        """Bornes pour L-BFGS-B —."""
        return [
            (1e-3, 20.0),        # kappa  : retour à la moyenne positif
            (1e-4,  1.0),        # theta  : variance long terme > 0
            (1e-3,  5.0),        # xi     : vol-of-vol positif
            (-0.999, 0.999),     # rho    : corrélation strictement dans (−1, 1)
            (-1.0,   1.0),       # mu     : drift borné
        ]



# HestonUKFCore — implémentation autonome du UKF Heston
#
# On n'hérite PAS de filterpy.UnscentedKalmanFilter pour rester indépendant
# de ses attributs internes (_UT, _num_sigmas, sigmas_f…) dont les noms
# varient selon la version installée.
# On utilise uniquement MerweScaledSigmaPoints (API publique stable) pour
# calculer les sigma-points et leurs poids, puis on implémente le cycle
# predict / update à la main — 20 lignes, zéro attribut privé.

class HestonUKFCore:
    """UKF de Heston entièrement autonome avec correction ρ·ξ·v_t·Δt.

    Seule dépendance filterpy : MerweScaledSigmaPoints (API publique stable).
    Toute la logique predict / update est réimplémentée ici sans recourir à
    aucun attribut interne de UnscentedKalmanFilter (_UT, sigmas_f, etc.).

    Attributs principaux
    --------------------
    x  : np.ndarray (1,)   État courant v̂_t
    P  : np.ndarray (1,1)  Covariance de l'état
    Q  : np.ndarray (1,1)  Bruit de processus  (mis à jour à chaque step)
    R  : np.ndarray (1,1)  Bruit de mesure     (mis à jour à chaque step)
    S  : np.ndarray (1,1)  Covariance d'innovation (après update)
    K  : np.ndarray (1,1)  Gain de Kalman      (après update)
    _rho_xi_vt_dt : float  Correction ρ·ξ·v_t·Δt (mis à jour à chaque step)
    """

    def __init__(self, fx, hx, x0: float, P0: float, Q0: float, R0: float,
                 dt: float, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        # Fonctions d'état et d'observation
        self.fx = fx
        self.hx = hx
        self.dt = dt

        # État initial et covariance
        self.x = np.array([x0])
        self.P = np.array([[P0]])

        # Matrices de bruit (state-dependent — mises à jour dans _step())
        self.Q = np.array([[Q0]])
        self.R = np.array([[R0]])

        # Correction ρ — mise à jour dans _step() avant chaque update()
        self._rho_xi_vt_dt: float = 0.0

        # Diagnostics
        self.S  = np.array([[R0]])
        self.K  = np.zeros((1, 1))

        # Générateur de sigma-points Merwe — seule API filterpy utilisée
        # alpha=1e-3 : points proches de la moyenne (évite v_t < 0)
        # beta=2     : optimal pour gaussienne
        # kappa=0    : standard pour n=1
        self._sp = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)

    def predict(self) -> None:
        """Étape de prédiction : propage les sigma-points à travers fx.

        v̂_{t|t-1} = Σ Wm_i · f(σ_i)
        P_{t|t-1}  = Σ Wc_i · (f(σ_i) − v̂)² + Q_t
        """
        # Génération des 2n+1 = 3 sigma-points autour de l'état courant
        sigmas = self._sp.sigma_points(self.x, self.P)
        Wm, Wc = self._sp.Wm, self._sp.Wc

        # Propagation déterministe à travers f (drift de Heston)
        sigmas_f = np.array([self.fx(s, self.dt) for s in sigmas])

        # Moyenne prédite : v̂_{t|t-1}
        self.x = np.sum(Wm[:, None] * sigmas_f, axis=0)

        # Covariance prédite : P_{t|t-1} + Q_t
        P_pred = np.zeros((1, 1))
        for i in range(len(Wm)):
            d = (sigmas_f[i] - self.x).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)
        self.P = P_pred + self.Q

        # Clamp : variance prédite strictement positive
        self.x[0] = max(float(self.x[0]), 1e-8)
        self.P    = np.maximum(self.P, 1e-10 * np.eye(1))

        # Sauvegarder les sigma-points propagés pour update()
        self._sigmas_f = sigmas_f
        self._Wm = Wm
        self._Wc = Wc

    def update(self, z: float) -> None:
        """Étape de mise à jour avec correction ρ·ξ·v_t·Δt dans P_{xz}.

        Étapes :
          1. Propager les sigma-points prédits à travers hx
          2. Calculer ẑ et S (covariance d'innovation)
          3. Calculer P_{xz} depuis les sigma-points
          4. AJOUTER la correction ρ·ξ·v_t·Δt à P_{xz}
          5. Gain de Kalman K = P_{xz} · S⁻¹
          6. Mise à jour de x et P
        """
        sigmas_f = self._sigmas_f
        Wm, Wc   = self._Wm, self._Wc

        # --- Étape 1 : sigma-points dans l'espace d'observation ---
        # h(σ_i) = E[r_t | σ_i] = (μ − σ_i/2) Δt
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])

        # --- Étape 2 : moyenne prédite ẑ et covariance d'innovation S ---
        # ẑ = Σ Wm_i · h(σ_i)
        # S = Σ Wc_i · (h(σ_i) − ẑ)² + R_t
        zp = float(np.sum(Wm * sigmas_h.flatten()))
        S_val = float(np.sum(Wc * (sigmas_h.flatten() - zp) ** 2)) + float(self.R[0, 0])
        S_val = max(S_val, 1e-12)

        # --- Étape 3 : cross-covariance P_{xz} depuis les sigma-points ---
        # Cov(v_{t+1}, r_t) approchée par la transformée non-parfumée seule
        Pxz = 0.0
        for i in range(len(Wm)):
            dx = float(sigmas_f[i][0]) - float(self.x[0])
            dz = float(sigmas_h[i][0]) - zp
            Pxz += Wc[i] * dx * dz

        # --- Étape 4 : CORRECTION ρ (terme manquant dans filterpy standard) ---
        # Cov(ξ√v_t dW_2, √v_t dW_1) = ρ · ξ · v_t · Δt
        # Injecté ici avant le calcul du gain
        Pxz += self._rho_xi_vt_dt

        # --- Étape 5 : gain de Kalman K = P_{xz} / S ---
        K = Pxz / S_val

        # --- Étape 6 : mise à jour état et covariance ---
        innovation = float(z) - zp
        self.x[0] = max(float(self.x[0]) + K * innovation, 1e-8)
        self.P[0, 0] = max(self.P[0, 0] - K * S_val * K, 1e-10)

        # Sauvegarde pour diagnostics (log-vraisemblance)
        self.S   = np.array([[S_val]])
        self.K   = np.array([[K]])
        self.zp  = zp
        self._innovation = innovation


# =============================================================================
# Factory interne — construit un HestonUKFCore initialisé
# =============================================================================

def _build_ukf_core(params: HestonParams, dt: float, v0: float) -> HestonUKFCore:
    """Instancie et initialise un HestonUKFCore pour l'état scalaire v_t.

    Dimensions : dim_x = 1 (variance v_t),  dim_z = 1 (log-return r_t).

    Sigma-points (Merwe Scaled)
    ---------------------------
    Avec n=1, on génère 2n+1 = 3 points :
        σ_0 = v̂_t                         (centre, poids W_0)
        σ_1 = v̂_t + √((n+λ)·P)           (droite)
        σ_2 = v̂_t − √((n+λ)·P)           (gauche)
    Les paramètres α=1e-3, β=2, κ=0 sont les valeurs standards pour des
    distributions proches de la gaussienne avec état positif borné.

    Paramètres
    ----------
    params : HestonParams  Paramètres calibrés.
    dt     : float         Pas de temps en années (ex : 1/252).
    v0     : float         Variance initiale (typiquement params.theta).
    """
    # Générateur de sigma-points selon la méthode de Merwe et al. (2000)
    # alpha=1e-3 : points très proches de la moyenne (évite v_t < 0 lors de la propagation)
    # beta=2     : optimal pour gaussienne (annule l'erreur de Taylor d'ordre 4)
    # kappa=0    : scaling secondaire standard pour n=1
    # ------------------------------------------------------------------
    # Fonction d'état f(v_t) — partie déterministe de l'EDS de la variance
    # dv_t = κ(θ − v_t)dt  +  ξ√v_t dW_2,t
    # La partie stochastique (ξ√v_t dW_2) est capturée par Q_t = ξ²v_t Δt.
    # ------------------------------------------------------------------
    def fx(v: np.ndarray, dt: float) -> np.ndarray:
        v_val = max(float(v[0]), 1e-8)
        # Drift de mean-reversion : κ pousse v_t vers θ
        v_next = v_val + params.kappa * (params.theta - v_val) * dt
        # Clamp : la variance ne peut pas être négative (propriété de Heston)
        return np.array([max(v_next, 1e-8)])

    # ------------------------------------------------------------------
    # Fonction d'observation h(v_t) — espérance du log-return
    # De l'EDS de S_t via formule d'Itô sur log S_t :
    # d(log S_t) = (μ − v_t/2)dt + √v_t dW_{1,t}
    # ⟹ E[r_t | v_t] = (μ − v_t/2) Δt
    # Le terme −v_t/2 est la correction d'Itô (convexity adjustment).
    # ------------------------------------------------------------------
    def hx(v: np.ndarray) -> np.ndarray:
        v_val = max(float(v[0]), 1e-8)
        # Correction d'Itô : l'espérance du log-return dépend de v_t
        return np.array([(params.mu - 0.5 * v_val) * dt])

    # Construction du filtre avec le nouveau constructeur autonome
    ukf = HestonUKFCore(
        fx  = fx,
        hx  = hx,
        x0  = v0,
        P0  = v0,
        Q0  = params.xi ** 2 * v0 * dt,   # Bruit de processus Q_t = ξ² v_t Δt
        R0  = v0 * dt,                      # Bruit de mesure R_t = v_t Δt
        dt  = dt,
    )

    # Initialisation de la correction ρ (sera recalculée à chaque step)
    ukf._rho_xi_vt_dt = params.rho * params.xi * v0 * dt

    return ukf


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
        # Série temporelle de la variance filtrée v̂_t (None avant filter())
        self._v_filtered: Optional[pd.Series] = None

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
        """Colonnes attendues pour le cache des paramètres rolling."""
        return ["kappa", "theta", "xi", "rho", "mu"]

    # ------------------------------------------------------------------
    # Méthode interne : un pas predict-update avec mise à jour des matrices
    # ------------------------------------------------------------------

    def _step(
        self, ukf: HestonUKFCore, params: HestonParams, r: float
    ) -> tuple[float, float]:
        """Exécute un cycle predict → update et retourne (v̂_t, contribution LL).

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
        v_pred = max(float(ukf.x[0]), 1e-8)

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
        innovation = r - expected_r

        # Étape de mise à jour : intègre l'observation r_t
        # Appelle HestonUKFCore.update() qui injecte la correction ρ dans P_{xz}
        ukf.update(np.array([r]))

        # Covariance d'innovation S_t stockée par HestonUKFCore.update()
        S = float(ukf.S[0, 0]) if ukf.S is not None else v_pred * self.dt
        S = max(S, 1e-12)

        # Contribution à la log-vraisemblance (décomposition en innovations) :
        # log p(r_t | F_{t-1}) = −½ [log(2π S_t) + ν_t² / S_t]
        innov_ll = -0.5 * (np.log(2.0 * np.pi * S) + innovation ** 2 / S)

        v_updated = max(float(ukf.x[0]), 1e-8)
        return v_updated, innov_ll

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
        ukf = _build_ukf_core(params, self.dt, v0)
        ll = 0.0

        try:
            for r in log_returns:
                _, innov_ll = self._step(ukf, params, float(r))
                ll += innov_ll
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
        start_end = window

        if rolling_df is not None and len(rolling_df) > 0:
            valid_index = rolling_df.index.intersection(returns.index[window:])
            if len(valid_index) > 0:
                rolling_df = rolling_df.loc[valid_index]
                rolling_records = [
                    {"date": idx, **self._params_to_record(self._record_to_params(rolling_df.loc[idx]))}
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
            self._rolling_params = rolling_df
            self._params = self._record_to_params(self._rolling_params.iloc[-1])
            logging.info("Calibration rolling entièrement restaurée depuis le cache.")
            return self

        for end in range(start_end, n):
            fit_date = returns.index[end]
            sample = returns.iloc[end - window:end].values

            def neg_ll(x: np.ndarray) -> float:
                p = HestonParams.from_array(x)
                feller_violation = max(0.0, p.xi ** 2 - 2.0 * p.kappa * p.theta)
                penalty = 1e4 * feller_violation
                return -self._log_likelihood(p, sample) + penalty

            x_start = x0.copy()
            start_obj = neg_ll(x_start)

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

            row = {"date": fit_date}
            row.update(self._params_to_record(fitted))
            rolling_records.append(row)

            should_checkpoint = (
                use_cache
                and cache_path is not None
                and (
                    len(rolling_records) % max(save_every, 1) == 0
                    or end == n - 1
                )
            )
            if should_checkpoint:
                pd.DataFrame(rolling_records).set_index("date").to_pickle(cache_path)

        self._rolling_params = pd.DataFrame(rolling_records).set_index("date")
        if use_cache and cache_path is not None:
            self._rolling_params.to_pickle(cache_path)
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

    # ------------------------------------------------------------------
    # filter() — estimation de l'état v̂_t
    # ------------------------------------------------------------------

    def filter(self, log_returns: pd.Series) -> pd.Series:
        """Estime la variance instantanée v̂_t par UKF sur toute la série.

        Si fit() a produit une calibration rolling, chaque date t est filtrée
        avec les paramètres calibrés sur la fenêtre [t-window, t), sans utiliser
        le return du jour dans l'estimation des paramètres.

        Paramètres
        ----------
        log_returns : pd.Series  Log-returns journaliers (index = dates).

        Retourne
        --------
        pd.Series  Variance filtrée v̂_t sur les dates où les paramètres sont disponibles.
        """
        if self._params is None:
            raise RuntimeError("Appeler fit() avant filter().")

        returns = log_returns.dropna()

        if self._rolling_params is None:
            params = self._params
            v0 = max(params.theta, 1e-6)
            ukf = _build_ukf_core(params, self.dt, v0)

            v_estimates: list[float] = []
            for r in returns.values:
                v_upd, _ = self._step(ukf, params, float(r))
                v_estimates.append(v_upd)

            self._v_filtered = pd.Series(v_estimates, index=returns.index, name="v_hat")
            return self._v_filtered

        rolling_index = self._rolling_params.index.intersection(returns.index)
        if len(rolling_index) == 0:
            raise RuntimeError("Aucune date commune entre les paramètres rolling et les returns.")

        first_params = self._record_to_params(self._rolling_params.loc[rolling_index[0]])
        v0 = max(first_params.theta, 1e-6)
        ukf = _build_ukf_core(first_params, self.dt, v0)

        v_estimates: list[float] = []
        for date in rolling_index:
            params = self._record_to_params(self._rolling_params.loc[date])
            self._update_core_functions(ukf, params)
            v_upd, _ = self._step(ukf, params, float(returns.loc[date]))
            v_estimates.append(v_upd)

        self._v_filtered = pd.Series(v_estimates, index=rolling_index, name="v_hat")
        return self._v_filtered

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
    def rolling_params(self) -> Optional[pd.DataFrame]:
        """Historique des paramètres calibrés date par date."""
        return self._rolling_params


# =============================================================================
# VolatilityTiming — allocation dynamique depuis le spread s_t
# =============================================================================

class VolatilityTiming:
    """Convertit le spread s_t en signal d'allocation et l'applique aux positions.

    Intuition
    ---------
    Le spread s_t = σ_IV − σ̂_t mesure la "cherté" de la volatilité implicite
    par rapport à la volatilité réalisée estimée. On l'utilise pour scaler
    dynamiquement les poids de la stratégie de base :
        w_t = w_base × f(s_t)

    Modes de normalisation
    ----------------------
    "linear"    → signal = clip(s_t / rolling_std(s_t), ±max_leverage)
                  Proportionnel au spread normalisé par sa volatilité récente.

    "rank"      → signal = percentile roulant recentré dans [−max_leverage, +max_leverage]
                  Robuste aux outliers, agnostique à l'unité de mesure de s_t.

    "threshold" → signal = +1 si s_t > +seuil, −1 si s_t < −seuil, 0 sinon
                  Binaire, adapté à un sizing discret (full in / out).

    Paramètres
    ----------
    scaling      : str    Mode de normalisation (défaut "linear").
    lookback     : int    Fenêtre de normalisation en jours (défaut 63 ~ 3 mois).
    max_leverage : float  Multiplicateur maximum en valeur absolue (défaut 2.0).
    threshold    : float  Seuil en points de vol pour le mode "threshold" (défaut 0.02).
    """

    def __init__(
        self,
        scaling: str = "linear",
        lookback: int = 63,
        max_leverage: float = 2.0,
        threshold: float = 0.02,
    ) -> None:
        if scaling not in ("linear", "rank", "threshold"):
            raise ValueError("scaling doit être 'linear', 'rank' ou 'threshold'.")
        self.scaling = scaling
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.threshold = threshold

    def compute_signal(self, spread: pd.Series) -> pd.Series:
        """Normalise s_t en signal d'allocation dans [−max_leverage, +max_leverage].

        Paramètres
        ----------
        spread : pd.Series  s_t = σ_IV − σ̂_t

        Retourne
        --------
        pd.Series  Signal de timing (même index que spread.dropna()).
        """
        spread = spread.dropna()

        if self.scaling == "linear":
            # Écart-type roulant de s_t sur la fenêtre lookback
            s_std = (
                spread.rolling(self.lookback, min_periods=5)
                .std()
                .replace(0.0, np.nan)
            )
            # Signal = s_t normalisé par son écart-type récent, clipé à ±max_leverage
            signal = (spread / s_std).clip(-self.max_leverage, self.max_leverage)

        elif self.scaling == "rank":
            # Rang percentile roulant de s_t (0 = min historique, 1 = max historique)
            rank = spread.rolling(self.lookback, min_periods=5).rank(pct=True)
            # Recentrage : rang 0.5 → 0 (neutre), 1.0 → +max_leverage, 0.0 → −max_leverage
            signal = ((rank - 0.5) * 2.0 * self.max_leverage).clip(
                -self.max_leverage, self.max_leverage
            )

        else:  # threshold
            # Signal binaire : ±1 au-delà des bandes ±threshold, 0 à l'intérieur
            signal = pd.Series(
                np.where(
                    spread >  self.threshold,  1.0,
                    np.where(spread < -self.threshold, -1.0, 0.0),
                ),
                index=spread.index,
            )

        signal.name = "timing_signal"
        return signal

    def apply_timing(
        self,
        df_positions: pd.DataFrame,
        signal: pd.Series,
        lag_bdays: int = 1,
    ) -> pd.DataFrame:
        """Multiplie les poids de la stratégie par le signal de timing.

        Le signal est décalé d'un jour ouvré par défaut : le poids appliqué à la
        date t est donc construit à partir de l'information disponible en t-1.
        Cela évite qu'une position prise à la date t utilise déjà r_t ou σ̂_t.

        Les dates sans signal disponible reçoivent un multiplicateur 1.0 (neutre),
        pour ne pas introduire de gaps dans la série de positions.

        Paramètres
        ----------
        df_positions : pd.DataFrame  Sortie de OptionTrade.generate_trades.
        signal       : pd.Series     Sortie de compute_signal (index = dates).
        lag_bdays    : int           Décalage d'exécution en jours ouvrés.

        Retourne
        --------
        pd.DataFrame  Identique à l'entrée avec la colonne 'weight' rescalée.
        """
        df = df_positions.copy()

        signal_exec = signal.copy()
        if lag_bdays > 0:
            signal_exec.index = signal_exec.index + pd.offsets.BDay(lag_bdays)

        # Préparer le signal pour le merge (reset_index() car index = dates)
        signal_df = (
            signal_exec.rename("timing_signal")
            .reset_index()
            .rename(columns={"index": "date"})
        )

        # Left join : toutes les lignes de positions sont conservées
        df = df.merge(signal_df, on="date", how="left")

        # Dates sans signal → multiplicateur neutre 1.0 (pas de levier forcé)
        df["timing_signal"] = df["timing_signal"].fillna(1.0)

        # Application du timing : w_t = w_base × f(s_t)
        df["weight"] = df["weight"] * df["timing_signal"]
        df = df.drop(columns=["timing_signal"])

        logging.info(
            "Volatility timing appliqué à %d lignes avec un lag de %d jour(s) ouvré(s).",
            len(df),
            lag_bdays,
        )
        return df


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
