import math
import numpy as np
import pandas as pd
import ta

# ============================== helpers ==============================

_EPS = 1e-9
PI2 = 2 * math.pi

def _get_time_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Return a full DatetimeIndex.
    - If idx is a MultiIndex with level 0 = date and level 1 = time, combine them.
    - If idx is a DatetimeIndex, return it directly.
    """
    if isinstance(idx, pd.MultiIndex):
        if idx.nlevels != 2:
            raise ValueError("Expected a 2-level MultiIndex: (date, time).")
        dates = idx.get_level_values(0)
        times = idx.get_level_values(1)
        return pd.to_datetime(dates.astype(str) + " " + times.astype(str))
    elif isinstance(idx, pd.DatetimeIndex):
        return idx
    else:
        raise ValueError("Index must be DatetimeIndex or MultiIndex(date, time).")

def _pct_rank(s: pd.Series, window: int) -> pd.Series:
    # Percentile rank of the last value inside a rolling window (no peeking)
    return s.rolling(window, min_periods=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def _rolling_autocorr(s: pd.Series, window: int, lag: int = 1) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda x: (pd.Series(x).autocorr(lag) if pd.Series(x).std(ddof=0) > 0 else 0.0),
        raw=False,
    )

def _safe_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).std()

# ============================== main ==============================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build end-of-bar features with no lookahead.
    Requires columns: Open, High, Low, Close, Volume (Volume used by some features).
    Index must be DatetimeIndex OR MultiIndex(date, time).
    """
    out = df.copy()

    # ---- basic returns (log) ----
    out["logret_1"] = np.log(out["Close"]).diff()

    # ---- ATR & EMA ----
    atr14 = ta.volatility.AverageTrueRange(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, fillna=False
    ).average_true_range()
    ema20 = ta.trend.EMAIndicator(close=out["Close"], window=20, fillna=False).ema_indicator()

    # ---- multi-horizon returns / realized vol (close-to-close) ----
    out["ret_3"]  = out["logret_1"].rolling(3,  min_periods=3 ).sum()
    out["ret_15"] = out["logret_1"].rolling(15, min_periods=15).sum()
    out["ret_60"] = out["logret_1"].rolling(60, min_periods=60).sum()
    out["rv_15"]  = _safe_std(out["logret_1"], 15)
    out["rv_60"]  = _safe_std(out["logret_1"], 60)

    # ---- ATR-normalized structure ----
    out["range_atr"]    = (out["High"] - out["Low"]) / (atr14 + _EPS)
    out["oc_atr"]       = (out["Close"] - out["Open"]).abs() / (atr14 + _EPS)
    out["close_z_atr"]  = (out["Close"] - ema20) / (atr14 + _EPS)

    # ---- candle shape (wicks) + CLV ----
    body_max = out[["Open", "Close"]].max(axis=1)
    body_min = out[["Open", "Close"]].min(axis=1)
    out["upper_wick_atr"] = (out["High"] - body_max) / (atr14 + _EPS)
    out["lower_wick_atr"] = (body_min - out["Low"]) / (atr14 + _EPS)
    out["clv"] = ((out["Close"] - out["Low"]) /
                  ((out["High"] - out["Low"]).replace(0, np.nan) + _EPS)) * 2 - 1

    # ---- Bollinger distance + percentile position ----
    bb = ta.volatility.BollingerBands(close=out["Close"], window=20, fillna=False)
    bb_mid = bb.bollinger_mavg()
    bb_w = (bb.bollinger_hband() - bb.bollinger_lband()).abs() + _EPS
    out["dist_bb"] = (out["Close"] - bb_mid) / bb_w
    out["bb_pct"]  = bb.bollinger_pband()

    # ---- MACD histogram ----
    macd = ta.trend.MACD(close=out["Close"], fillna=False)
    out["macd_hist"] = macd.macd_diff()

    # ---- volume surprise (rolling z-score) ----
    vol_ma_100 = out["Volume"].rolling(100, min_periods=100).mean()
    vol_sd_100 = out["Volume"].rolling(100, min_periods=100).std()
    out["vol_z"] = (out["Volume"] - vol_ma_100) / (vol_sd_100 + _EPS)

    # ---- vol regime + gap ----
    out["rv_300"]    = _safe_std(out["logret_1"], 300)
    out["vol_of_vol"] = out["rv_60"] / (out["rv_300"] + _EPS)
    out["gap_atr"]    = (out["Open"] - out["Close"].shift(1)) / (atr14 + _EPS)

    # ---- advanced realized volatility estimators ----
    # Parkinson
    hl = np.log((out["High"] / out["Low"]).replace(0, np.nan)) ** 2
    out["rv_parkinson_20"] = (hl.rolling(20, min_periods=20).sum() / (4 * math.log(2))).pow(0.5)

    # Rogers–Satchell
    co = np.log(out["Close"] / out["Open"]).replace([np.inf, -np.inf], np.nan)
    uh = np.log(out["High"]  / out["Open"]).replace([np.inf, -np.inf], np.nan)
    dl = np.log(out["Low"]   / out["Open"]).replace([np.inf, -np.inf], np.nan)
    rs = (uh * (uh - co) + dl * (dl - co)).clip(lower=0)
    out["rv_rs_20"] = rs.rolling(20, min_periods=20).sum().pow(0.5)

    # Yang–Zhang
    prev_close = out["Close"].shift(1)
    o_rets = np.log(out["Open"]  / prev_close).replace([np.inf, -np.inf], np.nan)
    c_rets = np.log(out["Close"] / out["Open"]).replace([np.inf, -np.inf], np.nan)
    u = np.log(out["High"] / out["Open"]).replace([np.inf, -np.inf], np.nan)
    d = np.log(out["Low"]  / out["Open"]).replace([np.inf, -np.inf], np.nan)
    n = 20
    k = 0.34 / (1 + n) + 0.00094
    sigma_o2  = o_rets.rolling(n, min_periods=n).var()
    sigma_c2  = c_rets.rolling(n, min_periods=n).var()
    sigma_rs2 = (u * (u - c_rets) + d * (d - c_rets)).rolling(n, min_periods=n).mean()
    out["rv_yz_20"] = (sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2).pow(0.5)

    # ---- percentile regime flags ----
    out["atr_pct"]  = _pct_rank(atr14, 252)
    out["rv60_pct"] = _pct_rank(out["rv_60"], 252)

    # ---- directional persistence & efficiency ----
    ret = out["logret_1"].fillna(0.0)
    sign_change  = (np.sign(ret) != np.sign(ret.shift(1))).astype(int).cumsum()
    streak_count = ret.groupby(sign_change).cumcount() + 1
    out["streak"] = streak_count * np.sign(ret).replace(0, 1)

    change10  = out["Close"].diff(10).abs()
    volsum10  = out["Close"].diff().abs().rolling(10, min_periods=10).sum()
    out["eff_ratio_10"] = (change10 / (volsum10 + _EPS)).fillna(0.0)

    out["ret_ac1_20"] = _rolling_autocorr(out["logret_1"], 20, lag=1)
    pos = ret.clip(lower=0)
    neg = (-ret).clip(lower=0)
    out["semivol_up_20"] = pos.rolling(20, min_periods=20).std()
    out["semivol_dn_20"] = neg.rolling(20, min_periods=20).std()
    out["skew_60"] = ret.rolling(60, min_periods=60).skew()
    out["kurt_60"] = ret.rolling(60, min_periods=60).kurt()

    # ---- breakout structure (Donchian) ----
    hh20 = out["High"].rolling(20, min_periods=20).max()
    ll20 = out["Low"].rolling(20, min_periods=20).min()
    rng20 = (hh20 - ll20).replace(0, np.nan)
    out["donch_pct_20"] = (out["Close"] - ll20) / (rng20 + _EPS) - 0.5
    out["donch_w_20"]   = rng20 / (atr14 + _EPS)

    # ---- volume & liquidity ----
    out["dollar_vol"] = (out["Close"] * out["Volume"]).astype("float64")
    dvol_ma = out["dollar_vol"].rolling(100, min_periods=100).mean()
    dvol_sd = out["dollar_vol"].rolling(100, min_periods=100).std()
    out["dvol_z_100"] = (out["dollar_vol"] - dvol_ma) / (dvol_sd + _EPS)

    # OBV change (end-of-bar update only)
    obv = (np.sign(out["Close"].diff().fillna(0.0)) * out["Volume"]).cumsum()
    out["obv_diff"] = obv.diff().fillna(0.0)

    # ---- intraday VWAP context (session = date level 0) ----
    try:
        session_key = out.index.get_level_values(0) if isinstance(out.index, pd.MultiIndex) else _get_time_index(out.index).normalize()
        typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
        cum_pv = (typical_price * out["Volume"]).groupby(session_key).cumsum()
        cum_v  = out["Volume"].groupby(session_key).cumsum().replace(0, np.nan)
        vwap   = (cum_pv / (cum_v + _EPS)).astype(float)

        d_high = out["High"].groupby(session_key).cummax()
        d_low  = out["Low"].groupby(session_key).cummin()
        d_rng  = (d_high - d_low).replace(0, np.nan)

        out["dist_vwap_atr"] = (out["Close"] - vwap) / (atr14 + _EPS)
        out["intraday_pos"]  = (out["Close"] - d_low) / (d_rng + _EPS) - 0.5
    except Exception:
        out["dist_vwap_atr"] = np.nan
        out["intraday_pos"]  = np.nan

    # ---- robust oscillators (minimal to avoid redundancy) ----
    out["rsi_14"] = ta.momentum.RSIIndicator(close=out["Close"], window=14, fillna=False).rsi()
    out["adx_14"] = ta.trend.ADXIndicator(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, fillna=False
    ).adx()
    out["stoch_k"] = ta.momentum.StochasticOscillator(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, smooth_window=3
    ).stoch()

    # ---- cyclical time features (using combined date+time) ----
    t = _get_time_index(out.index)
    minutes = t.hour * 60 + t.minute
    out["tod_sin"] = np.sin(PI2 * minutes / (60 * 24))
    out["tod_cos"] = np.cos(PI2 * minutes / (60 * 24))
    dow = t.dayofweek
    out["dow_sin"] = np.sin(PI2 * dow / 7)
    out["dow_cos"] = np.cos(PI2 * dow / 7)

    # ---- clean-up: kill infs, drop warmup rows, then drop remaining NaNs ----
    out = out.replace([np.inf, -np.inf], np.nan)

    # Largest window used ≈ 300 (rv_300). Trim initial warmup globally.
    min_ready = 300
    out = out.iloc[min_ready:]

    # Drop remaining NaNs created by min_periods across features
    out = out.dropna(how="any")

    # Winsorize obvious z/ratio features to control tails (±6)
    clip_cols = [
        "vol_z", "dvol_z_100", "dist_bb", "bb_pct", "close_z_atr", "range_atr",
        "oc_atr", "upper_wick_atr", "lower_wick_atr", "donch_pct_20", "donch_w_20",
        "dist_vwap_atr", "eff_ratio_10", "ret_ac1_20", "semivol_up_20", "semivol_dn_20",
        "rv_parkinson_20", "rv_rs_20", "rv_yz_20", "atr_pct", "rv60_pct", "intraday_pos"
    ]
    for c in clip_cols:
        if c in out.columns:
            out[c] = out[c].clip(lower=-6.0, upper=6.0)

    return out.astype("float32")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_indicators(df)

    # Toggle raw columns here. Keep Close/Open for sanity checks if you want.
    DROP_COLS = {"High", "Low", "Volume", "dollar_vol"}  # remove raw structure-only anchors
    keep = [c for c in feats.columns if c not in DROP_COLS]

    # Final sanitation
    X = feats[keep].replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return X
