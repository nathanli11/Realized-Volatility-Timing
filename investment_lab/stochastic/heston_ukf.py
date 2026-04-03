"""
heston_ukf.py

Unscented Kalman Filter implementation for a Heston-type stochastic variance model.

This module combines two main tasks:
1. Rolling maximum likelihood calibration of Heston model parameters
2. Filtering of latent variance using an Unscented Kalman Filter

The hidden state is instantaneous variance.
The observation is the daily log return of the underlying asset.

The model is discretized with a daily time step.
Process noise and measurement noise both depend on the current variance level and
are updated at each filtering step.

Typical workflow:
    fit()
    filter()
    implied_realized_spread()
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

try:
    from tqdm.auto import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_lab.stochastic.heston import HestonParams
from investment_lab.stochastic.kalman import HestonUKFCore, build_ukf_core


class HestonUKF:
    """
    Rolling Heston calibration and latent variance filtering with an Unscented Kalman Filter

    This class estimates a time series of latent variance from observed daily log returns

    The workflow has three main stages:
        1. fit() estimates model parameters on rolling windows using maximum likelihood
        2. filter() or filter_with_diagnostics() estimates the latent variance series
        3. implied_realized_spread() compares implied volatility with the filtered volatility estimate

    The class also stores rolling parameter estimates, calibration diagnostics, and filtering diagnostics for later inspection
    """

    def __init__(
        self,
        initial_params: Optional[HestonParams] = None,
        dt: float = 1.0 / 252.0,
        cache_dir: Optional[str | Path] = ".cache/heston_ukf",
        optimizer_maxiter: int = 300,
    ) -> None:
        """
        Initialize the HestonUKF object

        Parameters:
        initial_params : HestonParams, optional
            Initial parameter guess used as the optimizer starting point
            If not provided, default Heston parameters are used
        dt : float, default 1.0 / 252.0
            Time step expressed in years
        cache_dir : str or Path, optional
            Directory used to save and reload rolling calibration results
            If set to None, caching is disabled
        optimizer_maxiter : int, default 300
            Maximum number of iterations allowed for the optimizer

        Returns:
        None
        """
        self.initial_params = initial_params or HestonParams()
        self.dt = dt
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.optimizer_maxiter = optimizer_maxiter

        self._params: Optional[HestonParams] = None
        self._rolling_params: Optional[pd.DataFrame] = None
        self._fit_diagnostics: Optional[pd.DataFrame] = None
        self._v_filtered: Optional[pd.Series] = None
        self._filter_diagnostics: Optional[pd.DataFrame] = None

    @staticmethod
    def _params_to_record(params: HestonParams) -> dict[str, float]:
        """
        Convert a parameter object into a flat dictionary

        This helper is used when storing rolling calibration results in a DataFrame

        Parameters:
        params : HestonParams
            Parameter object to serialize

        Returns:
        dict[str, float]
            Dictionary containing one entry per model parameter
        """
        return {
            "kappa": params.kappa,
            "theta": params.theta,
            "xi": params.xi,
            "rho": params.rho,
            "mu": params.mu,
        }

    @staticmethod
    def _record_to_params(record: pd.Series) -> HestonParams:
        """
        Reconstruct a parameter object from a stored DataFrame row

        Parameters:
        record : pd.Series
            One row containing the model parameters

        Returns:
        HestonParams
            Reconstructed parameter object
        """
        return HestonParams(
            kappa=float(record["kappa"]),
            theta=float(record["theta"]),
            xi=float(record["xi"]),
            rho=float(record["rho"]),
            mu=float(record["mu"]),
        )

    def _update_ukf_functions(self, ukf: HestonUKFCore, params: HestonParams) -> None:
        """
        Update the transition and measurement functions of the UKF core

        This method is used in rolling mode because the model parameters may change from one date to the next. 
        The UKF core object is reused, but its transition and measurement functions must be refreshed with the current parameter set.

        Parameters:
        ukf : HestonUKFCore
            UKF core object
        params : HestonParams
            Current model parameters

        Returns:
        None
        """

        def fx(v: np.ndarray, dt: float) -> np.ndarray:
            """
            Transition function for latent variance

            Parameters:
            v : np.ndarray
                Current latent state
            dt : float
                Time step

            Returns:
            np.ndarray
                Predicted next latent state
            """
            v_val = max(float(v[0]), 1e-8)
            v_next = v_val + params.kappa * (params.theta - v_val) * dt
            return np.array([max(v_next, 1e-8)])

        def hx(v: np.ndarray) -> np.ndarray:
            """
            Measurement function for expected return

            Parameters:
            v : np.ndarray
                Current latent state

            Returns:
            np.ndarray
                Expected observed return implied by the state
            """
            v_val = max(float(v[0]), 1e-8)
            return np.array([(params.mu - 0.5 * v_val) * self.dt])

        ukf.fx = fx
        ukf.hx = hx

    def _build_cache_path(
        self,
        returns: pd.Series,
        window: int,
        ticker: str = "",
    ) -> Optional[Path]:
        """
        Build the cache file path used to store rolling calibration outputs

        The file name includes the ticker, rolling window length, and sample
        date range so that cached results remain easy to identify

        Parameters:
        returns : pd.Series
            Return series used to define the sample date range
        window : int
            Rolling window length
        ticker : str, default ""
            Optional identifier added to the cache file name

        Returns:
        Path or None
            Full path to the cache file, or None if caching is disabled
        """
        if self.cache_dir is None:
            return None

        start = returns.index[0].strftime("%Y%m%d")
        end = returns.index[-1].strftime("%Y%m%d")
        prefix = f"{ticker}_" if ticker else ""

        return self.cache_dir / f"rolling_{prefix}w{window}_{start}_{end}.parquet"

    @staticmethod
    def _cache_columns() -> list[str]:
        """
        Return the parameter columns expected in cached calibration files

        Parameters:
        None

        Returns:
        list[str]
            Ordered list of parameter column names
        """
        return ["kappa", "theta", "xi", "rho", "mu"]

    def _ukf_step(
        self,
        ukf: HestonUKFCore,
        params: HestonParams,
        r: float,
    ) -> dict[str, float]:
        """
        Run one UKF predict-update cycle and return step diagnostics

        This method updates process noise, measurement noise, and the
        correlation correction term from the current variance estimate
        before calling the UKF predict and update steps.

        Parameters:
        ukf : HestonUKFCore
            UKF core object
        params : HestonParams
            Model parameters used at the current step
        r : float
            Observed return at the current step

        Returns:
        dict[str, float]
            Dictionary containing filtered variance, filtered volatility,
            expected return, innovation, innovation variance, standardized
            innovation, Kalman gain, and one-step log likelihood contribution.
        """
        v_pred = max(float(ukf.x[0]), 1e-6)

        # Rebuild state-dependent noise terms from the current variance estimate
        ukf.Q = np.array([[params.xi**2 * v_pred * self.dt]])
        ukf.R = np.array([[v_pred * self.dt]])

        # Inject the correlation correction required by the Heston specification
        ukf._rho_xi_vt_dt = params.rho * params.xi * v_pred * self.dt

        # Run the prediction step before incorporating the new observation
        ukf.predict()

        v_after_predict = max(float(ukf.x[0]), 1e-8)

        # Compute the expected return implied by the predicted variance
        expected_r = (params.mu - 0.5 * v_after_predict) * self.dt

        # Update the filter with the observed return
        ukf.update(np.array([r]))

        innovation = float(ukf._innovation)
        innovation_var = float(ukf.S[0, 0]) if ukf.S is not None else v_pred * self.dt
        innovation_var = max(innovation_var, 1e-6)

        std_innovation = innovation / np.sqrt(innovation_var)
        kalman_gain = float(ukf.K[0, 0]) if ukf.K is not None else np.nan

        # Compute the one-step Gaussian innovation likelihood contribution
        innov_ll = -0.5 * (
            np.log(2.0 * np.pi * innovation_var)
            + innovation**2 / innovation_var
        )

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

    def _log_likelihood(
        self,
        params: HestonParams,
        log_returns: np.ndarray,
    ) -> float:
        """
        Compute the sample log likelihood from one-step UKF innovations

        The likelihood is obtained by summing the Gaussian innovation log likelihood over the input return sample

        Parameters:
        params : HestonParams
            Model parameters used to initialize and run the filter
        log_returns : np.ndarray
            Input array of log returns

        Returns:
        float
            Total sample log likelihood. Returns: negative infinity if the
            calculation fails or if the sample is too short
        """
        if len(log_returns) < 5:
            return -np.inf

        # Initialize the filter from the long-run variance level
        v0 = max(params.theta, 1e-6)
        ukf = build_ukf_core(params, self.dt, v0)
        ll = 0.0

        try:
            for r in log_returns:
                step_diag = self._ukf_step(ukf, params, float(r))
                ll += float(step_diag["loglik"])
        except Exception:
            return -np.inf

        return ll if np.isfinite(ll) else -np.inf

    def fit(
        self,
        log_returns: pd.Series,
        window: int = 252,
        use_cache: bool = True,
        save_every: int = 10,
        refit_every: int = 1,
        ticker: str = "",
    ) -> "HestonUKF":
        """
        Calibrate Heston parameters on rolling windows by maximum likelihood

        For each rolling window of daily log returns, this method estimates one
        parameter set using a bounded optimizer. The objective is the negative
        UKF log likelihood, optionally augmented by a soft penalty when the
        Feller condition is violated

        If the input series is shorter than the requested rolling window, the
        window is reduced so that at least one out-of-sample observation remains
        available for later filtering

        Parameters:
        log_returns : pd.Series
            Daily log return series indexed by date
        window : int, default 252
            Rolling window length used for parameter estimation
        use_cache : bool, default True
            Whether to load and save rolling calibration results from disk
        save_every : int, default 10
            Frequency of intermediate cache checkpoints
        refit_every : int, default 1
            Recalibration frequency in business days
            A value of 1 means daily recalibration
            A value greater than 1 reuses the previous parameter set between refits
        ticker : str, default ""
            Optional ticker name used to make cache file names easier to read

        Returns:
        HestonUKF
            The fitted instance
        """
        logging.info(
            "Fitting HestonUKF: n=%d observations, window=%d.",
            len(log_returns),
            window,
        )

        returns = log_returns.dropna()
        n = len(returns)

        if n < 6:
            raise ValueError(
                "At least 6 observations are required for calibration."
            )

        # Ensure that one observation remains outside the calibration window
        # This avoids a fully in-sample setup when the user later runs filtering
        if n <= window:
            new_window = n - 1
            logging.warning(
                "Input series is too short for the requested window. "
                "Window reduced from %d to %d.",
                window,
                new_window,
            )
            window = new_window

        # Build the cache path and create the parent directory if needed
        cache_path = self._build_cache_path(returns, window, ticker)
        if use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        rolling_df: Optional[pd.DataFrame] = None

        # Try to restore previously computed rolling results from disk
        if use_cache and cache_path is not None and cache_path.exists():
            try:
                loaded = pd.read_parquet(cache_path)
                if isinstance(loaded, pd.DataFrame) and all(
                    col in loaded.columns for col in self._cache_columns()
                ):
                    rolling_df = loaded.sort_index()
                    logging.info(
                        "Loaded rolling cache from %s with %d rows.",
                        cache_path,
                        len(rolling_df),
                    )
            except Exception as exc:
                logging.warning(
                    "Could not load rolling cache from %s. Reason: %s",
                    cache_path,
                    exc,
                )

        # x0 stores the optimizer starting point
        # It is updated over time so that each calibration starts from the last solution
        x0 = self.initial_params.to_array()

        # rolling_records stores the fitted parameters for each calibration date
        # diagnostics_records stores optimizer diagnostics and likelihood information
        rolling_records: list[dict[str, float | pd.Timestamp]] = []
        diagnostics_records: list[dict[str, object]] = []
        start_end = window

        # If a compatible cache exists, resume calibration from the last cached date
        if rolling_df is not None and len(rolling_df) > 0:
            valid_index = rolling_df.index.intersection(returns.index[window:])
            if len(valid_index) > 0:
                rolling_df = rolling_df.loc[valid_index]

                rolling_records = [
                    {
                        "date": idx,
                        **self._params_to_record(
                            self._record_to_params(rolling_df.loc[idx])
                        ),
                    }
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

                available_diag_cols = [
                    col for col in diagnostic_cols if col in rolling_df.columns
                ]

                if available_diag_cols:
                    diagnostics_records = [
                        {
                            "date": idx,
                            **rolling_df.loc[idx, available_diag_cols].to_dict(),
                        }
                        for idx in rolling_df.index
                    ]

                last_cached_date = rolling_df.index[-1]
                last_cached_pos = returns.index.get_loc(last_cached_date)
                start_end = int(last_cached_pos) + 1
                x0 = rolling_df.iloc[-1][self._cache_columns()].to_numpy(
                    dtype=np.float64
                )

                logging.info(
                    "Resuming rolling calibration after cached date %s.",
                    last_cached_date,
                )

        # If the cache already covers the whole sample, restore the internal state and exit early
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
            available_diag_cols = [
                col for col in diagnostic_cols if col in rolling_df.columns
            ]

            self._fit_diagnostics = (
                rolling_df[available_diag_cols].copy()
                if available_diag_cols
                else None
            )
            self._params = self._record_to_params(self._rolling_params.iloc[-1])

            logging.info("Rolling calibration fully restored from cache.")
            return self

        total_steps = n - start_end
        already_done = start_end - window
        iterator = range(start_end, n)

        # Wrap the loop in a progress bar when tqdm is available
        if _HAS_TQDM and total_steps > 0:
            iterator = _tqdm(
                iterator,
                total=total_steps,
                initial=0,
                desc=f"Heston rolling MLE w={window}",
                unit="day",
                postfix={"done": already_done, "total": n - window},
            )

        for end in iterator:
            fit_date = returns.index[end]

            # Build the rolling sample ending just before the fit date
            sample_series = returns.iloc[end - window : end]
            sample = sample_series.values
            sample_start_date = sample_series.index[0]
            sample_end_date = sample_series.index[-1]

            # Refit only at the requested frequency
            # Between refits, the previous parameter set is reused
            steps_since_start = end - start_end
            should_refit = (
                refit_every <= 1
                or steps_since_start % refit_every == 0
            )

            x_start = x0.copy()
            start_params = HestonParams.from_array(x_start)

            # Evaluate the objective at the starting point.
            # This is useful to decide whether the optimizer meaningfully improved the fit
            start_loglik = self._log_likelihood(start_params, sample)
            start_feller_violation = max(
                0.0,
                start_params.xi**2 - 2.0 * start_params.kappa * start_params.theta,
            )
            start_obj = -start_loglik + 1e4 * start_feller_violation

            if should_refit:

                # Define the penalized objective function used by the optimizer
                # The penalty discourages parameter sets that violate the Feller condition
                def neg_ll(x: np.ndarray) -> float:
                    p = HestonParams.from_array(x)
                    feller_violation = max(
                        0.0,
                        p.xi**2 - 2.0 * p.kappa * p.theta,
                    )
                    penalty = 1e4 * feller_violation
                    return -self._log_likelihood(p, sample) + penalty

                # Suppress numerical warnings emitted during optimization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        neg_ll,
                        x_start,
                        method="L-BFGS-B",
                        bounds=HestonParams.bounds(),
                        options={
                            "maxiter": self.optimizer_maxiter,
                            "ftol": 1e-10,
                            "gtol": 1e-7,
                        },
                    )

                # Accept the optimizer output either when it converged successfully
                # or when it improved the objective relative to the starting point
                if result.success or (result.fun < start_obj):
                    fitted = HestonParams.from_array(result.x)
                    x0 = result.x
                else:
                    logging.warning(
                        "MLE did not converge on %s. Previous parameters kept. "
                        "Optimizer message: %s",
                        fit_date,
                        result.message,
                    )
                    fitted = HestonParams.from_array(x_start)

            else:
                # Skip recalibration and simply reuse the previous parameter set
                fitted = HestonParams.from_array(x_start)
                result = type(
                    "_FakeResult",
                    (),
                    {
                        "success": True,
                        "fun": -start_loglik,
                        "status": 0,
                        "message": "skipped because of refit frequency",
                        "nfev": 0,
                        "nit": 0,
                    },
                )()

            # Recompute the final likelihood and diagnostics for the accepted parameter set
            final_loglik = self._log_likelihood(fitted, sample)
            feller_violation = max(
                0.0,
                fitted.xi**2 - 2.0 * fitted.kappa * fitted.theta,
            )

            row = {"date": fit_date}
            row.update(self._params_to_record(fitted))
            rolling_records.append(row)

            diagnostics_records.append(
                {
                    "date": fit_date,
                    "window_start": sample_start_date,
                    "window_end": sample_end_date,
                    "window_size": int(len(sample)),
                    "start_loglik": float(start_loglik),
                    "final_loglik": float(final_loglik),
                    "loglik_improvement": float(final_loglik - start_loglik),
                    "feller_violation": float(feller_violation),
                    "objective_value": float(
                        result.fun if np.isfinite(result.fun) else np.nan
                    ),
                    "optimizer_success": bool(result.success),
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "nfev": int(getattr(result, "nfev", -1)),
                    "nit": int(getattr(result, "nit", -1)),
                }
            )

            # Save partial results regularly so a long calibration can be resumed later
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
                rolling_params_df.join(diagnostics_df, how="left").to_parquet(
                    cache_path
                )

        # Store final calibration outputs in memory
        self._rolling_params = pd.DataFrame(rolling_records).set_index("date")
        self._fit_diagnostics = pd.DataFrame(diagnostics_records).set_index("date")

        # Save final results to disk if caching is enabled
        if use_cache and cache_path is not None:
            self._rolling_params.join(
                self._fit_diagnostics,
                how="left",
            ).to_parquet(cache_path)

        self._params = self._record_to_params(self._rolling_params.iloc[-1])

        logging.info(
            "Rolling MLE completed with %d calibrated parameter sets. "
            "Last set: kappa=%.3f theta=%.4f xi=%.3f rho=%.3f mu=%.4f",
            len(self._rolling_params),
            self._params.kappa,
            self._params.theta,
            self._params.xi,
            self._params.rho,
            self._params.mu,
        )

        return self

    def filter_with_diagnostics(self, log_returns: pd.Series) -> pd.DataFrame:
        """
        Filter the latent variance series and return full step diagnostics

        This method runs the UKF through the input return series and returns
        a DataFrame containing filtered variance, filtered volatility,
        innovation terms, Kalman gains, and likelihood contributions.

        If rolling parameters are available, the parameter set is updated date by date. 
        Otherwise, the last fitted parameter set is used for the whole sample.

        Parameters:
        log_returns : pd.Series
            Daily log return series indexed by date

        Returns:
        pd.DataFrame
            DataFrame indexed by date containing one row per filtering step
        """
        if self._params is None:
            raise RuntimeError(
                "fit() must be called before filter_with_diagnostics()."
            )

        returns = log_returns.dropna()

        # If no rolling parameters are available, use the last fitted parameter set over the entire return series
        if self._rolling_params is None:
            params = self._params
            v0 = max(params.theta, 1e-6)
            ukf = build_ukf_core(params, self.dt, v0)
            iter_dates = returns.index
        else:
            # In rolling mode, only keep dates present in both the parameter history and the return series
            iter_dates = self._rolling_params.index.intersection(returns.index)
            if len(iter_dates) == 0:
                raise RuntimeError(
                    "No common dates were found between rolling parameters and returns."
                )

            first_params = self._record_to_params(self._rolling_params.loc[iter_dates[0]])
            v0 = max(first_params.theta, 1e-6)
            ukf = build_ukf_core(first_params, self.dt, v0)

        rows = []

        # Run the filter sequentially and store all diagnostics at each step
        for date in iter_dates:
            if self._rolling_params is not None:
                params = self._record_to_params(self._rolling_params.loc[date])
                self._update_ukf_functions(ukf, params)

            r = returns.loc[date]
            out = self._ukf_step(ukf, params, float(r))
            out["date"] = date
            rows.append(out)

        df_diag = pd.DataFrame(rows).set_index("date")
        df_diag.index.name = iter_dates.name or "date"

        self._v_filtered = df_diag["v_hat"].rename("v_hat")
        self._filter_diagnostics = df_diag

        return df_diag

    def filter(self, log_returns: pd.Series) -> pd.Series:
        """
        Filter the latent variance series and return only the variance estimate

        This is a convenience wrapper around filter_with_diagnostics()

        Parameters:
        log_returns : pd.Series
            Daily log return series indexed by date

        Returns:
        pd.Series
            Filtered variance series indexed by date
        """
        df_diag = self.filter_with_diagnostics(log_returns)
        return df_diag["v_hat"].rename("v_hat")

    @property
    def sigma_hat(self) -> pd.Series:
        """
        Return the filtered volatility series

        Parameters:
        None

        Returns:
        pd.Series
            Filtered volatility series derived from the filtered variance estimate
        """
        if self._v_filtered is None:
            raise RuntimeError("filter() must be called before sigma_hat.")
        return np.sqrt(self._v_filtered).rename("sigma_hat")

    def implied_realized_spread(self, sigma_iv: pd.Series) -> pd.Series:
        """
        Compute the spread between implied volatility and filtered volatility

        The implied volatility series is aligned to the filtered volatility index

        Parameters:
        sigma_iv : pd.Series
            Implied volatility series

        Returns:
        pd.Series
            Difference between implied volatility and filtered volatility
        """
        spread = sigma_iv.reindex(self.sigma_hat.index) - self.sigma_hat
        spread.name = "iv_rv_spread"
        return spread

    @property
    def params(self) -> Optional[HestonParams]:
        """
        Return the last fitted parameter set

        Parameters:
        None

        Returns:
        HestonParams or None
            Last fitted parameter set, or None if fit() has not been called yet
        """
        return self._params

    @property
    def rolling_params(self) -> pd.DataFrame:
        """
        Return the history of rolling calibrated parameters.

        Parameters:
        None

        Returns:
        pd.DataFrame
            DataFrame containing one calibrated parameter set per date
        """
        if self._rolling_params is None:
            raise RuntimeError("fit() must be called before rolling_params.")
        return self._rolling_params.copy()

    @property
    def fit_diagnostics(self) -> pd.DataFrame:
        """
        Return calibration diagnostics generated during fit()

        Parameters:
        None

        Returns:
        pd.DataFrame
            DataFrame containing rolling calibration diagnostics
        """
        if self._fit_diagnostics is None:
            raise RuntimeError(
                "No calibration diagnostics are available. Call fit() first."
            )
        return self._fit_diagnostics.copy()

    @property
    def filter_diagnostics(self) -> pd.DataFrame:
        """
        Return filtering diagnostics generated during filter_with_diagnostics()

        Parameters:
        None

        Returns:
        pd.DataFrame
            DataFrame containing step-by-step filtering diagnostics
        """
        if self._filter_diagnostics is None:
            raise RuntimeError(
                "filter_with_diagnostics() must be called before filter_diagnostics."
            )
        return self._filter_diagnostics.copy()

    def forecast_variance(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """
        Forecast conditional variance at a future horizon

        This method uses the analytical mean-reversion formula of the Heston
        variance process to forecast variance at a fixed horizon

        Parameters:
        v_t : float
            Current variance level
        horizon : int, default 21
            Forecast horizon expressed in trading days
        params : HestonParams, optional
            Parameter set to use for the forecast. If not provided, the last fitted parameter set is used

        Returns:
        float
            Forecast variance at the requested horizon
        """
        p = params or self._params
        if p is None:
            raise RuntimeError("No calibrated parameters are available.")

        v_t = max(float(v_t), 1e-6)
        tau = horizon * self.dt
        v_forecast = p.theta + (v_t - p.theta) * np.exp(-p.kappa * tau)

        return max(float(v_forecast), 1e-6)

    def forecast_average_variance(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """
        Forecast average variance over a future horizon

        This method returns the expected average variance over the full horizon,
        not only the terminal variance at the end of the horizon

        Parameters:
        v_t : float
            Current variance level
        horizon : int, default 21
            Forecast horizon expressed in trading days
        params : HestonParams, optional
            Parameter set to use for the forecast. If not provided, the last
            fitted parameter set is used

        Returns:
        float
            Expected average variance over the forecast horizon
        """
        p = params or self._params
        if p is None:
            raise RuntimeError("No calibrated parameters are available.")

        v_t = max(float(v_t), 1e-6)
        tau = horizon * self.dt

        # When mean reversion is effectively zero, fall back to a flat forecast
        if p.kappa < 1e-10:
            v_avg = v_t
        else:
            decay_term = (1.0 - np.exp(-p.kappa * tau)) / (p.kappa * tau)
            v_avg = p.theta + (v_t - p.theta) * decay_term

        return max(float(v_avg), 1e-6)

    def forecast_average_volatility(
        self,
        v_t: float,
        horizon: int = 21,
        params: Optional[HestonParams] = None,
    ) -> float:
        """
        Forecast average volatility over a future horizon

        This method converts the average variance forecast into volatility

        Parameters:
        v_t : float
            Current variance level
        horizon : int, default 21
            Forecast horizon expressed in trading days
        params : HestonParams, optional
            Parameter set to use for the forecast. If not provided, the last fitted parameter set is used

        Returns:
        float
            Expected average volatility over the forecast horizon
        """
        return np.sqrt(
            self.forecast_average_variance(
                v_t,
                horizon=horizon,
                params=params,
            )
        )