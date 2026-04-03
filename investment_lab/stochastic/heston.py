"""
heston.py

Parameter container for the Heston stochastic volatility model.

This module defines a small dataclass used to store the five model parameters
required by the Heston variance process and the associated price dynamics.

The parameter set is used throughout the project, especially during rolling
maximum likelihood calibration and Unscented Kalman Filter state estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParams:
    """
    Store the parameters of the Heston stochastic volatility model.

    The class is a lightweight container used to pass model parameters between
    calibration, filtering, and forecasting components.

    Attributes:
    kappa : float, default 2.0
        Mean reversion speed of the variance process.
        A larger value means that variance returns more quickly toward its long-run level.

    theta : float, default 0.04
        Long-run variance level.
        This is the equilibrium level toward which variance mean reverts.

    xi : float, default 0.3
        Volatility of variance, often called vol of vol.
        A larger value means that variance itself moves more aggressively over time.

    rho : float, default -0.7
        Correlation between the Brownian motion driving the asset return and the Brownian motion driving the variance process.
        This parameter is often negative in equity markets.

    mu : float, default 0.0
        Drift parameter of the return process under the chosen measure.
        In this project, it is estimated jointly with the variance parameters.
    """

    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    mu: float = 0.0

    def feller_satisfied(self) -> bool:
        """
        Check whether the Feller condition is satisfied.

        The Feller condition is a standard stability condition for square-root variance processes. 
        When it is satisfied, the variance process is less likely to hit zero, which is desirable for both financial interpretation
        and numerical stability.

        Parameters:
        None

        Returns:
        bool
            True if the Feller condition is satisfied, otherwise False
        """
        return 2.0 * self.kappa * self.theta > self.xi**2

    def to_array(self) -> np.ndarray:
        """
        Convert the parameter set into a NumPy array.

        This helper is mainly used when passing parameters to numerical optimizers such as L-BFGS-B

        Parameters:
        None

        Returns:
        np.ndarray
            One-dimensional NumPy array containing the parameters in the following order: kappa, theta, xi, rho, mu
        """
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.mu])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        """
        Build a HestonParams instance from a NumPy array

        This is the inverse operation of "to_array". It is useful when an optimizer returns a parameter vector 
        that must be converted back into a structured parameter object.

        Parameters:
        x : np.ndarray
            One-dimensional NumPy array containing the parameters in the following order: kappa, theta, xi, rho, mu

        Returns:
        HestonParams
            New instance built from the input array
        """
        return cls(
            kappa=float(x[0]),
            theta=float(x[1]),
            xi=float(x[2]),
            rho=float(x[3]),
            mu=float(x[4]),
        )

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """
        Return parameter bounds for numerical optimization

        These bounds are designed to keep the optimizer inside economically meaningful and numerically stable regions. 
        They are used during rolling maximum likelihood calibration.

        Parameters:
        None

        Returns:
        list[tuple[float, float]]
            List of lower and upper bounds for each parameter, in the following order: kappa, theta, xi, rho, mu
        """
        return [
            (1e-3, 20.0),      # kappa must remain strictly positive
            (1e-4, 1.0),       # theta must remain strictly positive
            (1e-3, 5.0),       # xi must remain strictly positive
            (-0.999, 0.999),   # rho must stay strictly inside the correlation range
            (-1.0, 1.0),       # mu is kept inside a reasonable annualized range
        ]