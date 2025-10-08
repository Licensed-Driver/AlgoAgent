import datetime
import numpy as np
import pandas as pd

def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    delta = (pd.Timestamp(index[1]) - pd.Timestamp(index[0])).total_seconds()
    # crude heuristics
    if delta <= 120:  # ~1-2 minutes
        return 252.0 * 390.0  # trading minutes/year
    if delta <= 900:  # <= 15 minutes
        return 252.0 * (390.0 / 15.0)
    if delta <= 7200:  # <= 2 hours
        return 252.0 * (390.0 / 60.0)
    if delta <= 24 * 3600 + 3600:  # daily-ish
        return 252.0
    # weekly/monthly fallback
    return 52.0

def equity_to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()

def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    return float(dd.min()) if len(dd) > 0 else 0.0

def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    rets = equity_to_returns(equity)
    if len(rets) == 0:
        return 0.0
    ppy = _periods_per_year(equity.index)
    mean = rets.mean() - (rf / ppy)
    std = rets.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((mean / std) * np.sqrt(ppy))

def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    print(pd.Timestamp(equity.index[-1]))
    days = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days
    if days <= 0:
        return float(total_return)
    years = days / 365.25
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)

def basic_stats(equity: pd.Series) -> dict:
    return {
        "final_equity": float(equity.iloc[-1]) if len(equity) else np.nan,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else np.nan,
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(equity),
        "n_points": int(len(equity)),
    }

