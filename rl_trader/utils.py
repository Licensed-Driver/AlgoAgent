import numpy as np
import pandas as pd
from typing import Tuple

def set_seed(seed: int = 42):
    import random, torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    stats = {"mean": df.mean(), "std": df.std().replace(0, 1.0)}
    z = (df - stats["mean"]) / stats["std"]
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z, stats

def apply_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    z = (df - stats["mean"]) / stats["std"]
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z

def rolling_windows(df: pd.DataFrame, train_days: int, valid_days: int, test_days: int, stride_days: int):
    idx = df.index.sort_values()
    start = idx.min()
    end = idx.max()
    cur = start
    one_day = pd.Timedelta(days=1)
    while True:
        train_start = cur
        train_end = cur + pd.Timedelta(days=train_days) - one_day
        valid_end = train_end + pd.Timedelta(days=valid_days)
        test_end = valid_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        yield (df.loc[train_start:train_end], df.loc[train_end+one_day:valid_end], df.loc[valid_end+one_day:test_end])
        cur = cur + pd.Timedelta(days=stride_days)
