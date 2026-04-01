from __future__ import annotations
import numpy as np
try:
    # On utilise uniquement MerweScaledSigmaPoints — l'API publique stable de filterpy.
    # UnscentedKalmanFilter n'est pas importé pour éviter les attributs internes
    # instables (_UT, _num_sigmas, sigmas_f) qui varient selon la version installée.
    from filterpy.kalman import MerweScaledSigmaPoints
except ImportError as exc:
    raise ImportError("filterpy est requis : pip install filterpy") from exc
from investment_lab.stochastic.heston import HestonParams

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
        self._innovation: float = 0.0
        self.zp: float = 0.0

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

        # Clamp : variance prédite strictement positive avec plancher numérique
        self.x[0] = max(float(self.x[0]), 1e-6)
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
        S_val = max(S_val, 1e-6)

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
        # Clipping défensif pour éviter les explosions numériques quand v_t approche 0
        K = np.clip(Pxz / S_val, -20.0, 20.0)

        # --- Étape 6 : mise à jour état et covariance ---
        innovation = float(z) - zp
        self.x[0] = max(float(self.x[0]) + K * innovation, 1e-6)
        self.P[0, 0] = max(self.P[0, 0] - K * S_val * K, 1e-10)

        # Sauvegarde pour diagnostics (log-vraisemblance)
        self.S   = np.array([[S_val]])
        self.K   = np.array([[K]])
        self.zp  = zp
        self._innovation = innovation


# =============================================================================
# Factory interne — construit un HestonUKFCore initialisé
# =============================================================================

def build_ukf_core(params: HestonParams, dt: float, v0: float) -> HestonUKFCore:
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
        v_val = max(float(v[0]), 1e-6)
        # Drift de mean-reversion : κ pousse v_t vers θ
        v_next = v_val + params.kappa * (params.theta - v_val) * dt
        # Clamp : la variance ne peut pas être négative (propriété de Heston)
        return np.array([max(v_next, 1e-6)])

    # ------------------------------------------------------------------
    # Fonction d'observation h(v_t) — espérance du log-return
    # De l'EDS de S_t via formule d'Itô sur log S_t :
    # d(log S_t) = (μ − v_t/2)dt + √v_t dW_{1,t}
    # ⟹ E[r_t | v_t] = (μ − v_t/2) Δt
    # Le terme −v_t/2 est la correction d'Itô (convexity adjustment).
    # ------------------------------------------------------------------
    def hx(v: np.ndarray) -> np.ndarray:
        v_val = max(float(v[0]), 1e-6)
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

