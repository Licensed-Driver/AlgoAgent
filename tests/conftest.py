import numpy as np
import pandas as pd
import pytest

NO_FEES = {
    "model": "fixed",
    "per_share": 0.0,
    "min_per_order": 0.0,
    "sec_fee_per_dollar": 0.0,
    "taf_fee_per_share": 0.0,
}


def make_prices(closes, opens=None, freq="D"):
    """Daily bars, one bar per day."""
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq=freq, tz="EST")
    return pd.DataFrame({"Close": closes, "Open": opens}, index=idx)


def make_feats(prices, cols=1):
    return pd.DataFrame(
        {f"x{i}": np.zeros(len(prices)) for i in range(cols)}, index=prices.index
    )


def make_ohlcv(n, seed=0, freq="1min"):
    """Random OHLCV with Low <= Open, Close <= High."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 09:30", periods=n, freq=freq)
    open_ = 100 + np.cumsum(rng.normal(0, 0.02, n))
    high = open_ * (1 + rng.uniform(0.0005, 0.004, n))
    low = open_ * (1 - rng.uniform(0.0005, 0.004, n))
    close = rng.uniform(low, high)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.uniform(1e3, 1e4, n),
        },
        index=idx,
    )


@pytest.fixture
def flat_market():
    """Flat prices so any change in equity comes from costs."""
    prices = make_prices(np.full(60, 100.0))
    return prices, make_feats(prices)
