import logging
from typing import Any, Optional, Self

import numpy as np
import pandas as pd
from tqdm import tqdm

from investment_lab.data.option_db import OptionLoader
from investment_lab.util import check_is_true, ffill_options_data


class StrategyBacktester:
    _BACKTEST_COLS = ["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]
    _PNL_COLS = [
        "pnl",
        "delta_pnl",
        "gamma_pnl",
        "theta_pnl",
        "vega_pnl",
        "residual_pnl",
        "leverage",
        "cashflow",
    ]

    def __init__(self, df_positions: pd.DataFrame) -> None:
        missing_cols = set(self._BACKTEST_COLS).difference(df_positions.columns)
        check_is_true(
            len(missing_cols) == 0,
            f"Positions data is missing required columns: {missing_cols}",
        )
        check_is_true(
            len(df_positions) > 2,
            "Positions data is empty or too small to run backtest.",
        )

        self._df_positions = df_positions[self._BACKTEST_COLS]
        self._is_backtested = False
        self._df_pnl = pd.DataFrame()
        self._df_nav = pd.DataFrame()
        self._df_metainfo = pd.DataFrame()
        self._df_drifted_positions = pd.DataFrame()

    def compute_backtest(self, tcost_args: Optional[dict[str, Any]] = None) -> Self:
        df_positions_raw = self._preprocess_positions(self._df_positions[self._BACKTEST_COLS])

        tcost_args = tcost_args or {}
        df_positions = self.apply_tcost(df_positions_raw, **tcost_args).sort_values(["option_id", "date"])

        logging.info("Computing period to period difference, for P&L calculations.")
        df_positions["dv"] = df_positions.groupby(["option_id"])["mid"].diff().fillna(0)
        df_positions["dsigma"] = df_positions.groupby(["option_id"])["implied_volatility"].diff().fillna(0)
        df_positions["dS"] = df_positions.groupby(["option_id"])["spot"].diff().fillna(0)
        df_positions["dt"] = 1

        logging.info("Append previous period greeks for P&L calculations.")
        # FIX: use .bfill() instead of deprecated fillna(method="bfill")
        df_positions["prev_theta"] = df_positions.groupby("option_id")["theta"].shift(1).bfill()
        df_positions["prev_gamma"] = df_positions.groupby("option_id")["gamma"].shift(1).bfill()
        df_positions["prev_delta"] = df_positions.groupby("option_id")["delta"].shift(1).bfill()
        df_positions["prev_vega"] = df_positions.groupby("option_id")["vega"].shift(1).bfill()

        # FIX: align on business days so obs_date matches NAV index correctly
        df_positions["obs_date"] = df_positions["entry_date"].apply(
            lambda x: x - pd.offsets.BDay(1)
        )

        df_pnl = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 0, 0]],
            columns=self._PNL_COLS,
            index=[df_positions["date"].min() - pd.Timedelta(days=1)],
        )
        df_nav = pd.DataFrame(
            [[1.0]],
            columns=["NAV"],
            index=[df_positions["date"].min() - pd.Timedelta(days=1)],
        )
        logging.info(
            "Starting backtest computation over %s unique dates.",
            len(df_positions["date"].unique()),
        )

        drifted_positions = []
        for d in tqdm(df_positions["date"].sort_values().unique()):
            df_day = df_positions[df_positions["date"] == d].copy()
            df_day = df_day.merge(df_nav, left_on="obs_date", right_index=True, how="left")
            df_day["scaled_weight"] = (df_day["weight"] * df_day["NAV"]).fillna(df_day["weight"])

            df_day["pnl"] = df_day["scaled_weight"] * df_day["dv"]
            df_day["gamma_pnl"] = 0.5 * df_day["scaled_weight"] * df_day["dS"] ** 2 * df_day["prev_gamma"]
            df_day["delta_pnl"] = df_day["scaled_weight"] * df_day["dS"] * df_day["prev_delta"]
            df_day["theta_pnl"] = df_day["scaled_weight"] * df_day["dt"] * df_day["prev_theta"]
            df_day["vega_pnl"] = df_day["scaled_weight"] * df_day["dsigma"] * df_day["prev_vega"]
            df_day["residual_pnl"] = (
                df_day["pnl"]
                - df_day["delta_pnl"]
                - df_day["gamma_pnl"]
                - df_day["theta_pnl"]
                - df_day["vega_pnl"]
            )
            df_day["leverage"] = df_day["scaled_weight"] * df_day["spot"]
            df_day["cashflow"] = 0.0

            # Entry cashflow: pay premium (negative for long, positive for short)
            df_day.loc[df_day["entry_date"] == df_day["date"], "cashflow"] = (
                -df_day["scaled_weight"] * df_day["mid"]
            )

            # FIX: use intrinsic payoff at expiration instead of potentially stale mid
            expiry_mask = df_day["expiration"] == df_day["date"]
            if expiry_mask.any():
                call_mask = df_day["call_put"] == "C"
                put_mask = df_day["call_put"] == "P"
                payoff = np.where(
                    call_mask,
                    np.maximum(df_day["spot"] - df_day["strike"], 0.0),
                    np.where(
                        put_mask,
                        np.maximum(df_day["strike"] - df_day["spot"], 0.0),
                        df_day["mid"],  # spot / delta-hedge rows: use mid
                    ),
                )
                df_day.loc[expiry_mask, "cashflow"] = (
                    df_day.loc[expiry_mask, "scaled_weight"] * pd.Series(payoff, index=df_day.index)[expiry_mask]
                )

            df_pnl = pd.concat([df_pnl, df_day.groupby("date")[self._PNL_COLS].sum()])

            # FIX: extract scalar NAV value to avoid Series ambiguity
            if d not in df_nav.index:
                latest_nav_val = float(df_nav["NAV"].iloc[-1])
            else:
                latest_nav_val = float(df_nav.loc[d, "NAV"])
            df_nav.loc[d, "NAV"] = latest_nav_val + float(df_pnl.loc[d, "pnl"])

            drifted_positions.append(df_day)

        logging.info("Backtest computation completed.")
        self._is_backtested = True
        self._df_pnl = df_pnl.drop(columns=["leverage", "cashflow"]).copy()
        self._df_nav = df_nav.copy()
        self._df_metainfo = df_pnl[["leverage", "cashflow"]].copy()
        self._df_drifted_positions = pd.concat(drifted_positions).reset_index(drop=True)
        return self

    @staticmethod
    def _preprocess_positions(df_positions: pd.DataFrame):
        """Extend the position dataframe with option info."""
        logging.info("Loading option data for the backtest period.")
        df_positions_cp = df_positions.copy()
        start, end = df_positions_cp["date"].min(), df_positions_cp["date"].max()
        tickers = df_positions_cp["ticker"].unique().tolist()
        df_options = OptionLoader.load_data(start, end, process_kwargs={"ticker": tickers})

        # Synthetic spot row so delta-hedge legs are priced correctly
        df_spot = (
            df_options.groupby(["date", "ticker"])
            .apply(
                lambda x: pd.Series(
                    {
                        "option_id": x["ticker"].iloc[0],
                        "spot": x["spot"].iloc[0],
                        "bid": x["spot"].iloc[0],
                        "ask": x["spot"].iloc[0],
                        "mid": x["spot"].iloc[0],
                        "delta": 1,
                    }
                )
            )
            .reset_index()
        )
        df_options_spot = pd.concat([df_options, df_spot])
        df_positions_extended = df_positions_cp.merge(
            df_options_spot, how="left", on=["ticker", "option_id", "date"]
        )
        # Remove rows past expiration (except spot / delta-hedge rows with NaT expiration)
        df_positions_extended = df_positions_extended[
            (df_positions_extended["date"] <= df_positions_extended["expiration"])
            | df_positions_extended["expiration"].isna()
        ]
        df_positions_extended = ffill_options_data(df_positions_extended)
        return df_positions_extended

    @classmethod
    def apply_tcost(cls, df_positions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logging.info("No transaction cost applied.")
        return df_positions

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def pnl(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Call 'compute_backtest' first.")
        return self._df_pnl

    @property
    def nav(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Call 'compute_backtest' first.")
        return self._df_nav

    @property
    def metainfo(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Call 'compute_backtest' first.")
        return self._df_metainfo

    @property
    def drifted_positions(self) -> pd.DataFrame:
        check_is_true(self._is_backtested, "Call 'compute_backtest' first.")
        return self._df_drifted_positions

    def __del__(self):
        logging.info("Deleting StrategyBacktest instance.")
        self._df_positions = pd.DataFrame()
        self._df_pnl = pd.DataFrame()
        self._df_nav = pd.DataFrame()
        self._df_metainfo = pd.DataFrame()
        self._df_drifted_positions = pd.DataFrame()


# --------------------------------------------------------------------------- #
#  Transaction cost subclasses                                                 #
# --------------------------------------------------------------------------- #

class BacktesterBidAskFromData(StrategyBacktester):
    """Use the bid/ask spread recorded in market data on trade dates.

    - Long legs : buy at ask on entry, sell at bid on exit.
    - Short legs: sell at bid on entry, buy at ask on exit.
    """

    def __init__(self, df_positions: pd.DataFrame) -> None:
        super().__init__(df_positions)

    @classmethod
    def apply_tcost(cls, df_positions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logging.info("Applying bid-ask spread from data on transaction dates.")
        df_cp = df_positions.copy()

        entry = df_cp["entry_date"] == df_cp["date"]
        exit_ = df_cp["expiration"] == df_cp["date"]
        short = df_cp["weight"] < 0
        long_ = ~short

        df_cp["mid"] = np.where(entry & short, df_cp["bid"], df_cp["mid"])
        df_cp["mid"] = np.where(exit_ & short, df_cp["ask"], df_cp["mid"])
        df_cp["mid"] = np.where(entry & long_, df_cp["ask"], df_cp["mid"])
        df_cp["mid"] = np.where(exit_ & long_, df_cp["bid"], df_cp["mid"])
        return df_cp


class BacktesterFixedRelativeBidAsk(StrategyBacktester):
    """Apply a fixed relative half-spread on trade dates.

    Parameters passed via ``tcost_args``:
        relative_half_spread (float): Half-spread as a fraction of mid.
            Default 0.03 (i.e. 3 %).

    Example
    -------
    >>> BacktesterFixedRelativeBidAsk(df_positions).compute_backtest(
    ...     tcost_args={"relative_half_spread": 0.03}
    ... )
    """

    def __init__(self, df_positions: pd.DataFrame) -> None:
        super().__init__(df_positions)

    @classmethod
    def apply_tcost(
        cls,
        df_positions: pd.DataFrame,
        relative_half_spread: float = 0.03,
        **kwargs,
    ) -> pd.DataFrame:
        logging.info(
            "Applying fixed relative half-spread of %.2f%% on transaction dates.",
            relative_half_spread * 100,
        )
        df_cp = df_positions.copy()

        entry = df_cp["entry_date"] == df_cp["date"]
        exit_ = df_cp["expiration"] == df_cp["date"]
        short = df_cp["weight"] < 0
        long_ = ~short

        spread = relative_half_spread * df_cp["mid"]

        df_cp["mid"] = np.where(entry & short, df_cp["mid"] - spread, df_cp["mid"])
        df_cp["mid"] = np.where(exit_ & short, df_cp["mid"] + spread, df_cp["mid"])
        df_cp["mid"] = np.where(entry & long_, df_cp["mid"] + spread, df_cp["mid"])
        df_cp["mid"] = np.where(exit_ & long_, df_cp["mid"] - spread, df_cp["mid"])
        return df_cp
