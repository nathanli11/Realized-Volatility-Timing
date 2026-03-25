import pandas as pd
import numpy as np


def levels_to_returns(levels: pd.Series, method: str = "log") -> pd.Series:
    """Convert a series of price levels to returns.
    Process wide format only.

    Parameters:
        levels: Series of price levels.

    Returns:
        Series of returns.
    """
    if method == "log":
        return np.log1p(levels.pct_change())
    elif method == "simple":
        return levels.pct_change()
    else:
        raise ValueError("Method must be 'log' or 'simple'")


def returns_to_levels(returns: pd.Series, method: str = "log", base: float = 1.0) -> pd.Series:
    """Convert a series of returns to price levels.
    Process wide format only.

    Parameters:
        returns: Series of returns.

    Returns:
        Series of price levels.
    """
    if method == "log":
        return base * np.expm1(returns).cumsum()
    elif method == "simple":
        return base * (1 + returns).cumprod()
    else:
        raise ValueError("Method must be 'log' or 'simple'")
