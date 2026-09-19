import math
import numpy as np
import pandas as pd
import ta

# ============================== helpers ==============================

_EPS = 1e-9
PI2 = 2 * math.pi

# rv60_pct needs 60 bars for rv_60 and then 252 for the rank
WARMUP_BARS = 311

# Extra history per chunk so ATR/EMA/MACD converge to the same values as a full build
PARALLEL_LOOKBACK = WARMUP_BARS + 1024

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
    # Min-max position in the window, way faster than a rolling rank
    roll_min = s.rolling(window, min_periods=window).min()
    roll_max = s.rolling(window, min_periods=window).max()
    return (s - roll_min) / (roll_max - roll_min + _EPS)

def _rolling_autocorr(s: pd.Series, window: int, lag: int = 1) -> pd.Series:
    return s.rolling(window, min_periods=window).corr(s.shift(lag))

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
    inv_atr = 1.0 / (atr14 + _EPS)
    out["range_atr"]    = (out["High"] - out["Low"]) * inv_atr
    out["oc_atr"]       = (out["Close"] - out["Open"]).abs() * inv_atr
    out["close_z_atr"]  = (out["Close"] - ema20) * inv_atr

    # ---- candle shape (wicks) + CLV ----
    body_max = np.maximum(out["Open"], out["Close"])
    body_min = np.minimum(out["Open"], out["Close"])
    out["upper_wick_atr"] = (out["High"] - body_max) * inv_atr
    out["lower_wick_atr"] = (body_min - out["Low"]) * inv_atr
    hl_rng = (out["High"] - out["Low"])
    out["clv"] = ((out["Close"] - out["Low"]) / (hl_rng + _EPS)) * 2 - 1

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
    log_h = np.log(out["High"].replace(0, np.nan))
    log_l = np.log(out["Low"].replace(0, np.nan))
    log_c = np.log(out["Close"].replace(0, np.nan))
    log_o = np.log(out["Open"].replace(0, np.nan))

    hl_sq = (log_h - log_l) ** 2
    out["rv_parkinson_20"] = (hl_sq.rolling(20, min_periods=20).sum() / (4 * math.log(2))).pow(0.5)

    # Rogers–Satchell
    co = log_c - log_o
    uh = log_h - log_o
    dl = log_l - log_o
    rs = (uh * (uh - co) + dl * (dl - co)).clip(lower=0)
    out["rv_rs_20"] = rs.rolling(20, min_periods=20).sum().pow(0.5)

    # Yang–Zhang
    prev_close = out["Close"].shift(1)
    o_rets = np.log(out["Open"] / prev_close)
    c_rets = co

    sigma_o2 = o_rets.rolling(20, min_periods=20).var()
    sigma_c2 = c_rets.rolling(20, min_periods=20).var()
    sigma_rs2 = rs.rolling(20, min_periods=20).mean()

    k = 0.34 / (1 + 20) + 0.00094
    out["rv_yz_20"] = (sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2).pow(0.5)

    # ---- percentile regime flags ----
    out["atr_pct"]  = _pct_rank(atr14, 252)
    out["rv60_pct"] = _pct_rank(out["rv_60"], 252)

    # ---- directional persistence & efficiency ----
    ret = out["logret_1"].fillna(0.0)
    sign_ret = np.sign(ret)
    sign_change = (sign_ret != sign_ret.shift(1)).cumsum()
    out["streak"] = ret.groupby(sign_change).cumcount() + 1
    out["streak"] = out["streak"] * sign_ret.replace(0, 1)

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
        d_rng  = (d_high - d_low)

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
    time_factor = PI2 / (60 * 24)
    out["tod_sin"] = np.sin(minutes * time_factor)
    out["tod_cos"] = np.cos(minutes * time_factor)
    dow = t.dayofweek
    day_factor = PI2 / 7
    out["dow_sin"] = np.sin(dow * day_factor)
    out["dow_cos"] = np.cos(dow * day_factor)

    # ---- clean-up: kill infs, drop warmup rows, then drop remaining NaNs ----
    out = out.replace([np.inf, -np.inf], np.nan)

    # Drop warmup rows
    out = out.iloc[WARMUP_BARS:]
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
    # Also drop Open/Close/VWAP since raw prices are non-stationary
    DROP_COLS = {"High", "Low", "Volume", "dollar_vol", "Open", "Close", "VWAP"}
    keep = [c for c in feats.columns if c not in DROP_COLS]

    # Final sanitation
    X = feats[keep].replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return X


def build_features_parallel(df: pd.DataFrame, num_workers: int = 4, chunk_size: int = 50000) -> pd.DataFrame:
    """
    Threaded version of build_feature_matrix. Splits on day boundaries, gives
    each chunk PARALLEL_LOOKBACK bars of extra history, then keeps only each
    chunk's own rows so the result matches a sequential build.
    """
    import concurrent.futures

    n_rows = len(df)
    lookback = PARALLEL_LOOKBACK

    if n_rows <= lookback:
        return build_feature_matrix(df)

    session_keys = df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index.normalize()
    session_keys_np = session_keys.to_numpy()
    is_new_day = np.concatenate(([True], session_keys_np[1:] != session_keys_np[:-1]))
    day_start_indices = np.where(is_new_day)[0]

    # (history_start, own_start, end) per chunk
    chunk_specs = []
    current_start = 0

    while current_start < n_rows:
        target_end = current_start + chunk_size
        if target_end >= n_rows:
            end_idx = n_rows
        else:
            valid_starts = day_start_indices[day_start_indices <= target_end]
            if len(valid_starts) > 0 and valid_starts[-1] > current_start:
                end_idx = valid_starts[-1]
            else:
                future_starts = day_start_indices[day_start_indices > current_start]
                if len(future_starts) > 0:
                    end_idx = future_starts[0]
                else:
                    end_idx = n_rows

        overlap_start = max(0, current_start - lookback)
        chunk_specs.append((overlap_start, current_start, end_idx))

        current_start = end_idx

    processed_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(build_feature_matrix, df.iloc[hist:end].copy())
            for hist, _, end in chunk_specs
        ]
        # Gather in submission order so the index stays sorted
        for (_, own_start, end), future in zip(chunk_specs, futures):
            feats = future.result()
            own_rows = df.index[own_start:end]
            processed_chunks.append(feats[feats.index.isin(own_rows)])

    return pd.concat(processed_chunks, axis=0)
