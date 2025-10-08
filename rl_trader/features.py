import datetime
import math
import pandas as pd
import numpy as np
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Basic returns
    out["ret_1"] = out["Close"].pct_change().fillna(0.0)
    out["logret_1"] = np.log(out["Close"]).diff().fillna(0.0)

    # Trend / Momentum
    rsi14 = ta.momentum.RSIIndicator(out["Close"], window=14)
    stoch = ta.momentum.StochasticOscillator(out["High"], out["Low"], out["Close"])
    willr14 = ta.momentum.WilliamsRIndicator(out["High"], out["Low"], out["Close"]) 
    roc12 = ta.momentum.ROCIndicator(out["Close"], window=12)
    out["rsi_14"] = rsi14.rsi()
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["willr_14"] = willr14.williams_r()
    out["roc_12"] = roc12.roc()

    # Volatility
    atr14 = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"]) 
    bb = ta.volatility.BollingerBands(out["Close"]) 
    out["atr_14"] = atr14.average_true_range()
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()

    # Trend
    ema12 = ta.trend.EMAIndicator(out["Close"], window=12)
    ema26 = ta.trend.EMAIndicator(out["Close"], window=26)
    macd = ta.trend.MACD(out["Close"]) 
    adx14 = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"]) 
    out["ema_12"] = ema12.ema_indicator()
    out["ema_26"] = ema26.ema_indicator()
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["adx_14"] = adx14.adx()

    # Volume-based
    out["mfi_14"] = ta.volume.MFIIndicator(out["High"], out["Low"], out["Close"], out["Volume"]).money_flow_index()
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(out["Close"], out["Volume"]).on_balance_volume()

    # Price levels
    out["hl_range"] = out["High"] - out["Low"]
    out["oc_range"] = (out["Close"] - out["Open"]).abs()

    # Cyclical Time
    def index_to_time(input_timestamp: tuple[pd.Timestamp, pd.Timestamp]):
        timestamp = input_timestamp[1]
        return -math.cos((float(timestamp.hour) * float(60.0)) + float(timestamp.minute))
    out["time"] = out.index.map(mapper=index_to_time, na_action=None)

    # Replace infs, forward-fill missing values, then fill any residuals with 0
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_indicators(df)
    # Drop raw OHLCV if preferred; keep VWAP as anchor
    keep = [c for c in feats.columns if c not in ("Open","High","Low","Volume")]
    return feats[keep]
