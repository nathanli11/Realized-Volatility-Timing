"""
heston.py
=========
Dataclass des paramètres du modèle de Heston (1993).

Le modèle de Heston décrit la dynamique conjointe du prix S_t et de sa variance
instantanée v_t sous la mesure risque-neutre :

    dS_t  = μ S_t dt  +  S_t √v_t  dW_{1,t}
    dv_t  = κ(θ - v_t) dt  +  ξ √v_t  dW_{2,t}
    dW_1 · dW_2 = ρ dt

Les 5 paramètres (κ, θ, ξ, ρ, μ) sont calibrés par MLE dans HestonUKF.fit().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HestonParams:
    """Paramètres du modèle de Heston (1993).

    Attributs
    ---------
    kappa : float
        Vitesse de retour à la moyenne de v_t vers θ.
        Plus κ est grand, plus la variance revient rapidement à sa moyenne.
    theta : float
        Variance long terme (niveau d'équilibre de v_t).
        La volatilité long terme associée est √θ.
    xi : float
        Vol-of-vol : amplitude des chocs sur la variance.
        Un xi élevé implique une volatilité très variable dans le temps.
    rho : float
        Corrélation entre les deux mouvements browniens.
        Typiquement négatif sur les actions (effet levier : quand le prix baisse, la vol monte).
    mu : float
        Drift du log-prix sous la mesure physique.
    """

    kappa: float = 2.0
    theta: float = 0.04
    xi:    float = 0.3
    rho:   float = -0.7
    mu:    float = 0.0

    def feller_satisfied(self) -> bool:
        """Vérifie la condition de Feller : 2κθ > ξ².

        Quand cette condition est respectée, le processus CIR de la variance
        ne peut pas atteindre zéro, ce qui garantit v_t > 0 presque sûrement.
        Si elle n'est pas satisfaite, v_t peut toucher zéro et les racines carrées
        dans le modèle deviennent problématiques numériquement.
        """
        return 2.0 * self.kappa * self.theta > self.xi ** 2

    def to_array(self) -> np.ndarray:
        """Sérialise les 5 paramètres en vecteur numpy pour l'optimiseur L-BFGS-B."""
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.mu])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        """Reconstruit un HestonParams depuis un vecteur numpy (inverse de to_array)."""
        return cls(kappa=x[0], theta=x[1], xi=x[2], rho=x[3], mu=x[4])

    @staticmethod
    def bounds() -> list[tuple]:
        """Bornes des paramètres pour l'optimiseur L-BFGS-B.

        Ces bornes empêchent l'optimiseur d'explorer des régions économiquement
        absurdes (variance négative, corrélation hors [-1,1], etc.).
        """
        return [
            (1e-3, 20.0),       # kappa  : vitesse de mean-reversion > 0
            (1e-4,  1.0),       # theta  : variance long terme > 0
            (1e-3,  5.0),       # xi     : vol-of-vol > 0
            (-0.999, 0.999),    # rho    : corrélation strictement dans (-1, 1)
            (-1.0,   1.0),      # mu     : drift borné (annualisé)
        ]
