import numpy as np
import pandas as pd
import pytest

from rl_trader.features import (
    PARALLEL_LOOKBACK,
    WARMUP_BARS,
    add_indicators,
    build_feature_matrix,
    build_features_parallel,
)
from tests.conftest import make_ohlcv


def test_warmup_trim_covers_every_window():
    """Warmup trim should remove every NaN on its own."""
    df = make_ohlcv(1500)
    trimmed = add_indicators(df).replace([np.inf, -np.inf], np.nan)
    assert not trimmed.isna().any().any(), "NaNs left after the warmup trim"


def test_warmup_is_the_longest_chain():
    # 252-bar percentile rank on top of a 60-bar realized vol
    assert WARMUP_BARS == 60 + 252 - 1


@pytest.mark.parametrize(
    "n_rows,chunk_size,workers", [(3000, 1000, 4), (5000, 900, 4), (4000, 2500, 2)]
)
def test_parallel_matches_sequential(n_rows, chunk_size, workers):
    """Parallel build should match sequential exactly."""
    df = make_ohlcv(n_rows)
    sequential = build_feature_matrix(df.copy())
    parallel = build_features_parallel(df.copy(), num_workers=workers, chunk_size=chunk_size)
    pd.testing.assert_frame_equal(sequential, parallel)


def test_parallel_lookback_clears_the_ewm_recursions():
    """Lookback needs extra room for the EWM indicators to converge."""
    assert PARALLEL_LOOKBACK > WARMUP_BARS
    # Leftover ATR(14) error after the extra bars
    assert (13 / 14) ** (PARALLEL_LOOKBACK - WARMUP_BARS) < 1e-15


def test_features_do_not_use_future_bars():
    """Cutting off the end shouldn't change earlier features."""
    df = make_ohlcv(1200)
    cut = 900

    full = build_feature_matrix(df.copy())
    truncated = build_feature_matrix(df.iloc[:cut].copy())

    shared = full.index.intersection(truncated.index)
    assert len(shared) > 0
    pd.testing.assert_frame_equal(full.loc[shared], truncated.loc[shared])


def test_price_levels_are_dropped():
    """Raw price columns shouldn't be in the features."""
    df = make_ohlcv(1000)
    X = build_feature_matrix(df)
    for col in ("Open", "Close", "High", "Low", "Volume", "VWAP", "dollar_vol"):
        assert col not in X.columns


def test_output_is_finite():
    X = build_feature_matrix(make_ohlcv(1500))
    assert np.isfinite(X.to_numpy()).all()
