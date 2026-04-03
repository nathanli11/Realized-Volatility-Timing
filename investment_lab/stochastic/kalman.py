"""
kalman.py
=========
Implémentation autonome de l'Unscented Kalman Filter (UKF) pour le modèle de Heston.

Pourquoi ne pas utiliser filterpy.UnscentedKalmanFilter directement ?
----------------------------------------------------------------------
filterpy.UnscentedKalmanFilter expose des attributs internes (_UT, _num_sigmas,
sigmas_f…) dont les noms varient selon la version installée. Cette instabilité
d'API rendrait le code fragile.

On utilise donc uniquement MerweScaledSigmaPoints (API publique et stable de
filterpy) pour générer les sigma-points et leurs poids, puis on réimplémente
le cycle predict / update à la main — 20 lignes suffisent.

Spécificité Heston — correction ρ
-----------------------------------
Dans le modèle de Heston, les deux bruits (état et observation) sont corrélés :
    Cov(ξ√v_t dW_2, √v_t dW_1) = ρ · ξ · v_t · Δt

Un UKF standard calcule P_{xz} uniquement depuis les sigma-points, ce qui donne
P_{xz} ≈ 0 puisqu'il n'y a pas de bruit commun dans la représentation état-espace
discrétisée. HestonUKFCore injecte ce terme manquant dans update() avant de
calculer le gain de Kalman.
"""

from __future__ import annotations
import numpy as np

try:
    # Seule l'API publique MerweScaledSigmaPoints est utilisée.
    from filterpy.kalman import MerweScaledSigmaPoints
except ImportError as exc:
    raise ImportError("filterpy est requis : pip install filterpy") from exc

from investment_lab.stochastic.heston import HestonParams


class HestonUKFCore:
    """UKF scalaire pour la variance de Heston, avec correction exacte de ρ.

    État   : x = v_t (variance instantanée, scalaire)
    Mesure : z = r_t (log-return journalier, scalaire)

    Attributs principaux
    --------------------
    x  : np.ndarray (1,)    Estimation courante de la variance v̂_t
    P  : np.ndarray (1,1)   Covariance de l'état (incertitude sur v̂_t)
    Q  : np.ndarray (1,1)   Bruit de processus Q_t = ξ² v_t Δt  (mis à jour à chaque step)
    R  : np.ndarray (1,1)   Bruit de mesure    R_t = v_t Δt     (mis à jour à chaque step)
    S  : np.ndarray (1,1)   Covariance d'innovation S_t          (après update)
    K  : np.ndarray (1,1)   Gain de Kalman                       (après update)
    _rho_xi_vt_dt : float   Terme de correction ρ·ξ·v_t·Δt      (mis à jour à chaque step)
    """

    def __init__(
        self,
        fx,
        hx,
        x0: float,
        P0: float,
        Q0: float,
        R0: float,
        dt: float,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        # Fonctions d'état f(v_t) et d'observation h(v_t)
        # Elles sont redéfinies à chaque pas dans HestonUKF._update_core_functions()
        # pour refléter les paramètres calibrés du jour.
        self.fx = fx
        self.hx = hx
        self.dt = dt

        # État initial et covariance initiale
        self.x = np.array([x0])
        self.P = np.array([[P0]])

        # Matrices de bruit — dépendent de v_t (bruit multiplicatif dans Heston)
        # Elles sont recalculées à chaque pas dans HestonUKF._step()
        self.Q = np.array([[Q0]])
        self.R = np.array([[R0]])

        # Correction ρ — sera recalculée avant chaque appel à update()
        self._rho_xi_vt_dt: float = 0.0

        # Diagnostics exposés après chaque update()
        self.S = np.array([[R0]])    # Covariance d'innovation
        self.K = np.zeros((1, 1))   # Gain de Kalman
        self._innovation: float = 0.0
        self.zp: float = 0.0        # Mesure prédite ẑ_t

        # Générateur de sigma-points selon Merwe et al. (2000)
        # alpha=1e-3 : points très proches de la moyenne (limite les sigma-points négatifs)
        # beta=2     : optimal pour les distributions gaussiennes
        # kappa=0    : scaling secondaire standard pour n=1
        self._sp = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)

    def predict(self) -> None:
        """Étape de prédiction de l'UKF — propagation des sigma-points.

        On génère 2n+1 = 3 sigma-points autour de l'état courant, on les
        propage à travers la fonction d'état f (drift de Heston), puis on
        reconstruit la moyenne et la covariance prédites.

        Mise à jour :
            v̂_{t|t-1} = Σ Wm_i · f(σ_i)
            P_{t|t-1}  = Σ Wc_i · (f(σ_i) − v̂)² + Q_t
        """
        # Génération des 3 sigma-points autour de l'état courant
        sigmas = self._sp.sigma_points(self.x, self.P)
        Wm, Wc = self._sp.Wm, self._sp.Wc

        # Propagation à travers f(v_t) = v_t + κ(θ − v_t)Δt  (partie déterministe)
        sigmas_f = np.array([self.fx(s, self.dt) for s in sigmas])

        # Moyenne prédite par la transformée non-parfumée (unscented transform)
        self.x = np.sum(Wm[:, None] * sigmas_f, axis=0)

        # Covariance prédite : Σ des déviations pondérées + bruit de processus Q_t
        P_pred = np.zeros((1, 1))
        for i in range(len(Wm)):
            d = (sigmas_f[i] - self.x).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)
        self.P = P_pred + self.Q

        # Plancher numérique : la variance ne peut pas être négative
        self.x[0] = max(float(self.x[0]), 1e-6)
        self.P    = np.maximum(self.P, 1e-10 * np.eye(1))

        # Sauvegarde pour l'étape update() qui suit immédiatement
        self._sigmas_f = sigmas_f
        self._Wm = Wm
        self._Wc = Wc

    def update(self, z: float) -> None:
        """Étape de mise à jour avec correction ρ·ξ·v_t·Δt dans la cross-covariance.

        C'est ici que réside la spécificité Heston par rapport à un UKF standard :
        la cross-covariance P_{xz} calculée depuis les sigma-points est incomplète
        car elle ignore la corrélation ρ entre les deux browniens. On l'injecte
        manuellement avant de calculer le gain de Kalman.

        Étapes :
          1. Propager les sigma-points prédits à travers h(v_t) = E[r_t | v_t]
          2. Calculer la mesure prédite ẑ et la variance d'innovation S
          3. Calculer P_{xz} depuis les sigma-points (terme UKF standard)
          4. Ajouter la correction ρ·ξ·v_t·Δt à P_{xz}  ← spécifique Heston
          5. Gain de Kalman K = P_{xz} / S
          6. Mise à jour de l'état et de la covariance
        """
        sigmas_f = self._sigmas_f
        Wm, Wc   = self._Wm, self._Wc

        # Étape 1 : sigma-points dans l'espace d'observation
        # h(v) = (μ − v/2) Δt  est l'espérance du log-return (formule d'Itô)
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])

        # Étape 2 : mesure prédite ẑ et variance d'innovation S = Var(r_t | F_{t-1})
        zp    = float(np.sum(Wm * sigmas_h.flatten()))
        S_val = float(np.sum(Wc * (sigmas_h.flatten() - zp) ** 2)) + float(self.R[0, 0])
        S_val = max(S_val, 1e-6)  # plancher pour éviter la division par zéro

        # Étape 3 : cross-covariance P_{xz} = Cov(v_{t+1}, r_t) depuis les sigma-points
        Pxz = 0.0
        for i in range(len(Wm)):
            dx = float(sigmas_f[i][0]) - float(self.x[0])
            dz = float(sigmas_h[i][0]) - zp
            Pxz += Wc[i] * dx * dz

        # Étape 4 : correction ρ — terme manquant dans un UKF standard
        # Cov(ξ√v_t dW_2, √v_t dW_1) = ρ · ξ · v_t · Δt
        # Ce terme capture la co-variation entre l'état (variance) et la mesure (return)
        # due à la corrélation entre les deux browniens dans le modèle de Heston.
        Pxz += self._rho_xi_vt_dt

        # Étape 5 : gain de Kalman K = P_{xz} / S
        # Clipping défensif pour éviter les explosions quand v_t est très petit
        K = np.clip(Pxz / S_val, -20.0, 20.0)

        # Étape 6 : mise à jour de l'état et de la covariance
        innovation = float(z) - zp
        self.x[0] = max(float(self.x[0]) + K * innovation, 1e-6)
        self.P[0, 0] = max(self.P[0, 0] - K * S_val * K, 1e-10)

        # Sauvegarde des diagnostics pour la log-vraisemblance et les plots
        self.S            = np.array([[S_val]])
        self.K            = np.array([[K]])
        self.zp           = zp
        self._innovation  = innovation


# =============================================================================
# Fonction de construction — instancie et initialise un HestonUKFCore
# =============================================================================

def build_ukf_core(params: HestonParams, dt: float, v0: float) -> HestonUKFCore:
    """Crée un HestonUKFCore initialisé avec les paramètres de Heston donnés.

    Cette fonction construit les deux fonctions f et h nécessaires au filtre,
    puis instancie HestonUKFCore avec les matrices de bruit initiales correctes.

    Paramètres
    ----------
    params : HestonParams
        Paramètres calibrés (κ, θ, ξ, ρ, μ) pour cette fenêtre temporelle.
    dt : float
        Pas de temps en années. Typiquement 1/252 pour des données journalières.
    v0 : float
        Variance initiale du filtre. On utilise typiquement params.theta
        (la variance long terme) comme point de départ neutre.

    Retourne
    --------
    HestonUKFCore prêt à l'emploi, initialisé à v̂_0 = v0.
    """

    def fx(v: np.ndarray, dt: float) -> np.ndarray:
        """Partie déterministe de l'équation d'état de Heston.

        Discrétisation Euler-Maruyama du drift de la variance :
            v_{t+1} ≈ v_t + κ(θ − v_t)Δt
        La partie stochastique ξ√v_t dW_2 est représentée par le bruit Q_t = ξ²v_t Δt.
        """
        v_val  = max(float(v[0]), 1e-6)
        v_next = v_val + params.kappa * (params.theta - v_val) * dt
        return np.array([max(v_next, 1e-6)])  # la variance reste positive

    def hx(v: np.ndarray) -> np.ndarray:
        """Espérance du log-return conditionnellement à la variance v_t.

        Découle de la formule d'Itô appliquée à log S_t :
            d(log S_t) = (μ − v_t/2) dt + √v_t dW_{1,t}
        Donc E[r_t | v_t] = (μ − v_t/2) Δt

        Le terme −v_t/2 est la correction de convexité d'Itô : l'espérance du
        log-return est inférieure à celle du return simple à cause de la variance.
        """
        v_val = max(float(v[0]), 1e-6)
        return np.array([(params.mu - 0.5 * v_val) * dt])

    # Matrices de bruit initiales — seront recalculées à chaque pas dans _step()
    # Q_0 = ξ² v_0 Δt  (bruit multiplicatif sur la variance)
    # R_0 = v_0 Δt     (variance du log-return conditionnellement à v_0)
    ukf = HestonUKFCore(
        fx  = fx,
        hx  = hx,
        x0  = v0,
        P0  = v0,
        Q0  = params.xi ** 2 * v0 * dt,
        R0  = v0 * dt,
        dt  = dt,
    )

    # Initialisation de la correction ρ (recalculée à chaque pas dans _step())
    ukf._rho_xi_vt_dt = params.rho * params.xi * v0 * dt

    return ukf
