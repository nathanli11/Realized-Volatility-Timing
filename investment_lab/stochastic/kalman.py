"""
kalman.py

Standalone Unscented Kalman Filter implementation for a Heston-type variance model.

This module does not rely on the full FilterPy UnscentedKalmanFilter class.
Instead, it only uses the public MerweScaledSigmaPoints helper to generate sigma points and their weights, 
and it implements the predict and update steps directly.

This design avoids dependence on internal FilterPy attributes whose names may change across versions.

The main Heston-specific feature of this implementation is the manual correction applied to the state-observation cross covariance during the update step.
In the Heston model, state noise and observation noise are correlated.
A standard Unscented Kalman Filter does not capture this term automatically in the discretized state-space representation, so it must be injected explicitly.
"""

from __future__ import annotations

import numpy as np

try:
    from filterpy.kalman import MerweScaledSigmaPoints
except ImportError as exc:
    raise ImportError("filterpy is required. Install it with: pip install filterpy") from exc

from investment_lab.stochastic.heston import HestonParams


class HestonUKFCore:
    """
    Scalar Unscented Kalman Filter core for Heston latent variance estimation

    This class implements a one-dimensional Unscented Kalman Filter where the
    hidden state is the instantaneous variance and the observation is the daily log return.

    Compared with a generic Unscented Kalman Filter, this implementation includes
    an explicit correction term in the update step to account for the correlation
    between state noise and observation noise in the Heston model.

    Attributes:
    fx : callable
        State transition function
    hx : callable
        Measurement function
    dt : float
        Time step expressed in years
    x : np.ndarray
        Current filtered state estimate
    P : np.ndarray
        Current state covariance matrix
    Q : np.ndarray
        Current process noise covariance matrix
    R : np.ndarray
        Current measurement noise covariance matrix
    S : np.ndarray
        Innovation covariance matrix from the latest update
    K : np.ndarray
        Kalman gain from the latest update
    zp : float
        Predicted observation from the latest update
    _innovation : float
        Innovation from the latest update
    _rho_xi_vt_dt : float
        Heston-specific correction term added to the state-observation cross covariance
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
    ) -> None:
        """
        Initialize the scalar Unscented Kalman Filter core

        Parameters:
        fx : callable
            State transition function. It must accept the current state and the
            time step, and return the predicted next state
        hx : callable
            Measurement function. It must accept the current state and return the expected observation
        x0 : float
            Initial state estimate
        P0 : float
            Initial state covariance
        Q0 : float
            Initial process noise covariance
        R0 : float
            Initial measurement noise covariance
        dt : float
            Time step expressed in years
        alpha : float, default 1e-3
            Spread parameter for sigma-point generation
        beta : float, default 2.0
            Prior-distribution shape parameter for sigma-point generation
            The default value is standard for Gaussian assumptions
        kappa : float, default 0.0
            Secondary scaling parameter for sigma-point generation

        Returns:
        None
        """
        # Store the state transition and measurement functions
        # In rolling calibration mode, these functions can be updated externally to reflect a new parameter set
        self.fx = fx
        self.hx = hx
        self.dt = dt

        # Initialize the filtered state and its covariance
        self.x = np.array([x0])
        self.P = np.array([[P0]])

        # Initialize process noise and measurement noise
        # In the Heston setting, both terms depend on the current variance level
        # and are therefore updated outside this core before each filter step
        self.Q = np.array([[Q0]])
        self.R = np.array([[R0]])

        # Initialize the Heston-specific correction term
        # This value is refreshed before each update step
        self._rho_xi_vt_dt: float = 0.0

        # Initialize diagnostic outputs exposed after each update step
        self.S = np.array([[R0]])
        self.K = np.zeros((1, 1))
        self._innovation: float = 0.0
        self.zp: float = 0.0

        # Build the sigma-point generator using the public FilterPy API only
        self._sp = MerweScaledSigmaPoints(
            n=1,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

    def predict(self) -> None:
        """
        Run the UKF prediction step

        This method generates sigma points around the current state estimate,
        propagates them through the state transition function, and reconstructs the predicted state mean and covariance.

        Parameters:
        None

        Returns:
        None
        """
        # Generate sigma points around the current filtered state
        sigmas = self._sp.sigma_points(self.x, self.P)
        Wm, Wc = self._sp.Wm, self._sp.Wc

        # Propagate sigma points through the state transition function
        sigmas_f = np.array([self.fx(s, self.dt) for s in sigmas])

        # Reconstruct the predicted state mean from propagated sigma points
        self.x = np.sum(Wm[:, None] * sigmas_f, axis=0)

        # Reconstruct the predicted covariance and add process noise
        P_pred = np.zeros((1, 1))
        for i in range(len(Wm)):
            d = (sigmas_f[i] - self.x).reshape(-1, 1)
            P_pred += Wc[i] * (d @ d.T)
        self.P = P_pred + self.Q

        # Apply defensive floors to keep the variance state and covariance positive
        self.x[0] = max(float(self.x[0]), 1e-6)
        self.P = np.maximum(self.P, 1e-10 * np.eye(1))

        # Store intermediate quantities for the next update step
        self._sigmas_f = sigmas_f
        self._Wm = Wm
        self._Wc = Wc

    def update(self, z: float) -> None:
        """
        Run the UKF update step with the Heston-specific cross-covariance correction

        This method projects the predicted sigma points into observation space,
        computes the predicted observation and innovation variance, builds the
        state-observation cross covariance, adds the missing Heston correction term,
        and updates the state estimate and covariance.

        Parameters:
        z : float
            Observed measurement at the current time step

        Returns:
        None
        """
        sigmas_f = self._sigmas_f
        Wm, Wc = self._Wm, self._Wc

        # Project predicted sigma points into observation space
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])

        # Compute the predicted observation and the innovation variance
        zp = float(np.sum(Wm * sigmas_h.flatten()))
        S_val = float(np.sum(Wc * (sigmas_h.flatten() - zp) ** 2)) + float(self.R[0, 0])

        # Apply a floor to avoid division by zero in the Kalman gain
        S_val = max(S_val, 1e-6)

        # Compute the standard UKF state-observation cross covariance
        Pxz = 0.0
        for i in range(len(Wm)):
            dx = float(sigmas_f[i][0]) - float(self.x[0])
            dz = float(sigmas_h[i][0]) - zp
            Pxz += Wc[i] * dx * dz

        # Add the Heston-specific correction term
        # This term captures the correlation between state noise and observation noise
        Pxz += self._rho_xi_vt_dt

        # Compute the scalar Kalman gain
        # Clip the value defensively to avoid unstable updates in near-degenerate cases
        K = np.clip(Pxz / S_val, -20.0, 20.0)

        # Update the state estimate and its covariance
        innovation = float(z) - zp
        self.x[0] = max(float(self.x[0]) + K * innovation, 1e-6)
        self.P[0, 0] = max(self.P[0, 0] - K * S_val * K, 1e-10)

        # Save diagnostics for likelihood evaluation and post-analysis
        self.S = np.array([[S_val]])
        self.K = np.array([[K]])
        self.zp = zp
        self._innovation = innovation


def build_ukf_core(params: HestonParams, dt: float, v0: float) -> HestonUKFCore:
    """
    Build and initialize a HestonUKFCore instance

    This helper creates the state transition function and the measurement
    function associated with the supplied Heston parameter set, then
    initializes the Unscented Kalman Filter core with consistent initial noise levels.

    Parameters:
    params : HestonParams
        Model parameters used to define the state transition and measurement rules
    dt : float
        Time step expressed in years
    v0 : float
        Initial variance level used to initialize the filter

    Returns:
    HestonUKFCore
        Initialized UKF core object ready to be used in filtering
    """

    def fx(v: np.ndarray, dt: float) -> np.ndarray:
        """
        Deterministic part of the variance transition equation

        The stochastic part of the variance dynamics is handled through the
        process noise covariance rather than directly inside this function.

        Parameters:
        v : np.ndarray
            Current variance state.
        dt : float
            Time step expressed in years.

        Returns:
        np.ndarray
            Predicted next variance state.
        """
        v_val = max(float(v[0]), 1e-6)
        v_next = v_val + params.kappa * (params.theta - v_val) * dt
        return np.array([max(v_next, 1e-6)])

    def hx(v: np.ndarray) -> np.ndarray:
        """
        Expected log return conditional on the current variance

        This function maps the latent variance state to the expected observation

        Parameters:
        v : np.ndarray
            Current variance state

        Returns:
        np.ndarray
            Expected observed return conditional on the supplied state
        """
        v_val = max(float(v[0]), 1e-6)
        return np.array([(params.mu - 0.5 * v_val) * dt])

    # Initialize process noise and measurement noise from the starting variance level
    ukf = HestonUKFCore(
        fx=fx,
        hx=hx,
        x0=v0,
        P0=v0,
        Q0=params.xi**2 * v0 * dt,
        R0=v0 * dt,
        dt=dt,
    )

    # Initialize the Heston-specific correction term
    # This value is updated later before each filtering step
    ukf._rho_xi_vt_dt = params.rho * params.xi * v0 * dt

    return ukf