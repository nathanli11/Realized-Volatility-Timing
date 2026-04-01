from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

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

    