import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from investment_lab.data.option_db import OptionLoader
from investment_lab.data.rates_db import USRatesLoader
from investment_lab.dataclass import OptionLegSpec, VarianceSwapLegSpec
from investment_lab.option_selection import select_options, select_closest_maturity
from investment_lab.rates import compute_forward
from investment_lab.util import check_is_true, ffill_options_data


class OptionTradeABC(ABC):
    _REQUIRED_COLUMNS = [
        "date",
        "option_id",
        "expiration",
        "delta",
        "strike",
        "moneyness",
        "call_put",
        "spot",
        "ticker",
    ]

    @classmethod
    def generate_trades(
        cls,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | str,
        legs: list[OptionLegSpec],
        cost_neutral: bool = False,
        hedging_args: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Generate the trade dataframe containing the information for each trade with daily positions.

        Args:
            start_date (datetime): Start date.
            end_date (datetime): End date.
            tickers (list[str] | str): Tickers of the underliers.
            legs (list[OptionLegSpec]): List of leg definitions.
            cost_neutral (bool): Whether to neutralize the cost between legs.
            hedging_args (Optional[dict]): Arguments forwarded to ``_hedge_trades``.

        Returns:
            pd.DataFrame with columns:
            ['date', 'option_id', 'entry_date', 'leg_name', 'weight', 'ticker']
        """
        df_trades_daily = cls._generate_trades(
            start_date, end_date, tickers=tickers, legs=legs, cost_neutral=cost_neutral
        )
        hedging_args = hedging_args or {}
        return cls._hedge_trades(df_trades_daily, **hedging_args)[
            ["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]
        ]

    @classmethod
    def _generate_trades(
        cls,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | str,
        legs: list[OptionLegSpec],
        cost_neutral: bool = False,
    ) -> pd.DataFrame:
        """Generate trades for the main option legs (no hedging overlay)."""
        df_options = cls._load_option_data(
            start_date, end_date, process_kwargs={"ticker": tickers}
        )
        df_trades = cls._select_options(df_options, legs, cost_neutral=cost_neutral)
        df_trades_daily = cls._convert_trades_to_timeseries(df_trades)
        df_trades_daily = df_trades_daily.merge(
            df_options, on=["date", "option_id", "ticker"], how="left"
        )
        df_trades_daily = df_trades_daily[
            df_trades_daily["date"].between(start_date, end_date)
        ]
        df_trades_daily = df_trades_daily.drop_duplicates(
            subset=["date", "leg_name", "option_id"]
        )
        df_trades_daily = ffill_options_data(df_trades_daily)
        if "risk_free_rate" not in df_trades_daily.columns:
            start, end = df_trades_daily["date"].min(), df_trades_daily["date"].max()
            df_rates = USRatesLoader.load_data(start, end)
            df_trades_daily = compute_forward(
                df_options=df_trades_daily, df_rates=df_rates
            )
        return df_trades_daily

    @classmethod
    def _load_option_data(
        cls, start_date: datetime, end_date: datetime, **kwargs
    ) -> pd.DataFrame:
        logging.info("Loading option data from %s to %s", start_date, end_date)
        option_df = cls.load_data(start_date, end_date, **kwargs)
        missing_cols = set(cls._REQUIRED_COLUMNS).difference(option_df.columns)
        check_is_true(
            len(missing_cols) == 0,
            f"Option data is missing required columns: {missing_cols}",
        )
        return cls._preprocess_option_data(option_df)

    @classmethod
    @abstractmethod
    def load_data(cls, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _preprocess_option_data(cls, df_option: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing option data.")
        return df_option

    @classmethod
    def _select_options(
        cls,
        df_options: pd.DataFrame,
        legs: list[OptionLegSpec],
        cost_neutral: bool = False,
    ) -> pd.DataFrame:
        df_list = []
        for leg in deepcopy(legs):
            leg_name = leg.pop("leg_name", "")
            weight = leg.pop("weight", np.nan)
            rebal_week_day = leg.pop("rebal_week_day", 1)
            check_is_true(
                np.all([0 <= rebal <= 4 for rebal in rebal_week_day]),
                "Error, provide a rebalance week day among {0,1,2,3,4}",
            )
            logging.info(
                "Selecting options for leg: %s using the rules:\n%s", leg_name, leg
            )
            selected_option_df = select_options(df_options, **leg)
            selected_option_df["leg_name"] = leg_name
            selected_option_df["weight"] = (
                weight
                / selected_option_df["spot"].where(
                    selected_option_df["spot"] != 0, np.nan
                )
            ).ffill()
            selected_option_df = selected_option_df[
                selected_option_df["date"].dt.day_of_week.isin(rebal_week_day)
            ]
            df_list.append(selected_option_df.rename(columns={"date": "entry_date"}))

        df = pd.concat(df_list)
        if cost_neutral:
            df = cls._neutralize_cost(df)

        return df[
            [
                "entry_date",
                "option_id",
                "expiration",
                "leg_name",
                "weight",
                "ticker",
            ]
        ].drop_duplicates(subset=["entry_date", "leg_name", "ticker"])

    @classmethod
    def _neutralize_cost(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        logging.info("Adjusting weights to make the strategy cost neutral.")
        df_cp = df_trades.copy()
        df_cp["premium"] = df_cp["weight"] * df_cp["mid"]
        df_cp["L/S"] = np.where(df_cp["weight"] > 0, "Long", "Short")
        df_trade_pivot = df_cp.pivot_table(
            index=["entry_date", "ticker"],
            columns="L/S",
            values="premium",
            aggfunc="sum",
        )
        df_trade_pivot["missing_premium"] = (
            -df_trade_pivot["Long"] - df_trade_pivot["Short"]
        )
        df_trade_pivot["scaling_factor"] = np.where(
            df_trade_pivot["missing_premium"] < 0,
            (df_trade_pivot["Short"] + df_trade_pivot["missing_premium"])
            / df_trade_pivot["Short"],
            (df_trade_pivot["Long"] + df_trade_pivot["missing_premium"])
            / df_trade_pivot["Long"],
        )
        df_cp = df_cp.merge(
            df_trade_pivot.reset_index()[
                ["entry_date", "ticker", "scaling_factor", "missing_premium"]
            ],
            on=["entry_date", "ticker"],
            how="left",
        )
        df_cp["weight"] = np.where(
            ((df_cp["missing_premium"] < 0) & (df_cp["weight"] < 0))
            | ((df_cp["missing_premium"] > 0) & (df_cp["weight"] > 0)),
            df_cp["weight"] * df_cp["scaling_factor"],
            df_cp["weight"],
        )
        return df_cp

    @classmethod
    def _convert_trades_to_timeseries(cls, df_trades: pd.DataFrame) -> pd.DataFrame:
        logging.info("Converting %s df_trades to daily time series", len(df_trades))
        df_cp = df_trades.copy()
        df_cp["date"] = df_cp.apply(
            lambda r: pd.date_range(start=r["entry_date"], end=r["expiration"], freq="B"),
            axis=1,
        )
        df_cp = df_cp.explode("date").reset_index(drop=True)
        return df_cp[["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]]

    @classmethod
    def _hedge_trades(cls, df_trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """No hedging by default — identity passthrough."""
        return df_trades


class OptionTrade(OptionLoader, OptionTradeABC):
    pass


class DeltaHedgedOptionTrade(OptionTrade):
    """Delta-hedge the option portfolio daily with the underlying spot.

    For each (date, ticker, entry_date) group the net delta × weight is computed
    and an offsetting spot position is added.

    FIX vs original
    ---------------
    The original code set ``expiration = date + 1 BDay``, which caused the hedge
    row to disappear from the backtest after a single day (the backtester drops
    rows where ``date > expiration``).

    Correct behaviour: the spot hedge should live for the same duration as the
    option leg it is hedging.  We therefore set ``expiration`` to the *maximum*
    expiration of the option legs in the same group so the hedge survives until
    the last leg expires.
    """

    @classmethod
    def _hedge_trades(cls, df_trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logging.info("Applying delta hedging to df_trades.")

        # Maximum expiration per (date, ticker, entry_date) group — keeps the
        # delta-hedge row alive for the full life of the trade.
        max_expiry = (
            df_trades.groupby(["date", "ticker", "entry_date"])["expiration"]
            .max()
            .reset_index()
            .rename(columns={"expiration": "max_expiration"})
        )

        df_hedge = (
            df_trades.groupby(["date", "ticker", "entry_date"])
            .apply(
                lambda x: pd.Series(
                    {
                        "option_id": x["ticker"].iloc[0],
                        "leg_name": "DELTA_HEDGING",
                        # Net delta exposure to neutralise
                        "weight": -(x["delta"] * x["weight"]).sum(),
                    }
                )
            )
            .reset_index()
        )

        # Attach the correct expiration
        df_hedge = df_hedge.merge(max_expiry, on=["date", "ticker", "entry_date"], how="left")
        df_hedge = df_hedge.rename(columns={"max_expiration": "expiration"})

        return (
            pd.concat([df_trades, df_hedge], ignore_index=True)
            .sort_values(by=["date", "option_id"])
            .reset_index(drop=True)
        )


class DeltaGammaHedgedOptionTrade(DeltaHedgedOptionTrade):
    """Add a gamma-hedge overlay on top of the delta hedge.

    Parameters passed via ``hedging_args``:
        hedging_leg (OptionLegSpec): Specification of the gamma-hedge option leg.

    Example
    -------
    >>> DeltaGammaHedgedOptionTrade.generate_trades(
    ...     ...,
    ...     hedging_args={
    ...         "hedging_leg": {
    ...             "day_to_expiry_target": 5,
    ...             "strike_target": -0.1,
    ...             "strike_col": "delta",
    ...             "call_or_put": "P",
    ...             "weight": 0.2,
    ...             "leg_name": "Long 10D Put 5d",
    ...             "rebal_week_day": [2],
    ...         }
    ...     },
    ... )
    """

    @classmethod
    def _hedge_trades(
        cls, df_trades: pd.DataFrame, *, hedging_leg: OptionLegSpec, **kwargs
    ) -> pd.DataFrame:
        logging.info("Applying gamma-delta hedging to df_trades.")
        start = df_trades["date"].min()
        end = df_trades["date"].max()
        tickers = df_trades["ticker"].unique().tolist()

        df_gamma_hedge = cls._generate_trades(
            cost_neutral=False,
            end_date=end,
            start_date=start,
            tickers=tickers,
            legs=[hedging_leg],
        )
        df_combined = pd.concat([df_trades, df_gamma_hedge], ignore_index=True)
        # Delegate to DeltaHedgedOptionTrade for the net delta neutralisation
        return super()._hedge_trades(df_combined)


class VarianceSwap(OptionTrade):
    """Replicate a variance swap via a static strip of options (Demeterfi et al.)."""

    @classmethod
    def _select_options(
        cls,
        df_options: pd.DataFrame,
        legs: list[VarianceSwapLegSpec],
        **kwargs,
    ) -> pd.DataFrame:
        df_list = []
        check_is_true(len(legs) == 1, "VarianceSwap can only handle 1 leg.")
        for leg in deepcopy(legs):
            leg_name = "VARIANCE SWAP"
            weight = leg.pop("weight", np.nan)
            leg.pop("strike_spacing", np.nan)  # kept for API compatibility
            rebal_week_day = leg.pop("rebal_week_day", 1)
            check_is_true(
                np.all([0 <= rebal <= 4 for rebal in rebal_week_day]),
                "Error, provide a rebalance week day among {0,1,2,3,4}",
            )
            logging.info(
                "Selecting options for leg: %s using the rules:\n%s", leg_name, leg
            )
            selected_option_df = select_closest_maturity(df_options, **leg)
            selected_option_df = selected_option_df.loc[
                (
                    (selected_option_df["call_put"] == "P")
                    & (selected_option_df["moneyness"] <= 1.0)
                )
                | (
                    (selected_option_df["call_put"] == "C")
                    & (selected_option_df["moneyness"] >= 1.0)
                )
            ]
            selected_option_df["leg_name"] = leg_name

            selected_option_df_pvt = (
                selected_option_df.pivot_table(
                    index=["date", "expiration", "strike", "option_id"],
                    columns="call_put",
                    values="mid",
                )
                .reset_index()
                .sort_values("strike")
            )

            selected_option_df_pvt["dK"] = selected_option_df_pvt["strike"].diff()
            selected_option_df_pvt["weight_C"] = (
                (selected_option_df_pvt["dK"] / 2) * selected_option_df_pvt["C"]
            ) / selected_option_df_pvt["strike"] ** 2
            selected_option_df_pvt["weight_P"] = (
                (selected_option_df_pvt["dK"] / 2) * selected_option_df_pvt["P"]
            ) / selected_option_df_pvt["strike"] ** 2
            selected_option_df_pvt["weight"] = selected_option_df_pvt[
                "weight_P"
            ].fillna(selected_option_df_pvt["weight_C"])
            selected_option_df_pvt = selected_option_df_pvt[
                ["date", "option_id", "weight"]
            ]

            selected_option_df = selected_option_df.merge(
                selected_option_df_pvt, on=["date", "option_id"]
            )
            selected_option_df = selected_option_df.loc[
                selected_option_df["date"].dt.day_of_week.isin(rebal_week_day)
            ]

            selected_option_df_norm = (
                selected_option_df.groupby("date")
                .apply(
                    lambda df_group: cls._normalize_strike_weights(
                        df_group, target_weight=weight
                    )
                )
                .reset_index(drop=True)
            )
            df_list.append(
                selected_option_df_norm.rename(columns={"date": "entry_date"})
            )

        df = pd.concat(df_list)
        return df[["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]]

    @classmethod
    def _normalize_strike_weights(
        cls, df_group: pd.DataFrame, target_weight: float
    ) -> pd.DataFrame:
        spot = df_group["spot"].iloc[0]
        strike_weight_sum = df_group["weight"].sum()
        target_size = target_weight / spot
        if strike_weight_sum > 0:
            normalization_factor = target_size / strike_weight_sum
            df_group["weight"] = df_group["weight"] * normalization_factor
        else:
            df_group["weight"] = 0
        return df_group
