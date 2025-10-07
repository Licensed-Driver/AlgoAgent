import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def _to_timeframe(tf: str):
    tf = tf.lower()
    if tf in ("1min", "1m", "minute"):
        return TimeFrame.Minute
    if tf in ("5min","5m"):
        return TimeFrame(5, "Minute")
    if tf in ("15min","15m"):
        return TimeFrame(15, "Minute")
    if tf in ("1d","day","daily"):
        return TimeFrame.Day
    raise ValueError(f"Unsupported timeframe: {tf}")

def main():
    fetch_alpaca_bars("NVDA", "2020-01-01", "2020-01-31", "1Min")

def fetch_alpaca_bars(symbol: str, start: str, end: str, timeframe: str = "1Min") -> pd.DataFrame:
    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if key is None or secret is None:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET in .env")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=pd.Timestamp(start, tz="EST"),
        end=pd.Timestamp(end, tz="EST"),
        timeframe=_to_timeframe(timeframe),
        feed='sip',  # best available
        limit=None,
        adjustment='raw',
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError(f"No bars returned for {symbol} between {start} and {end}")
    # Alpaca returns MultiIndex [symbol, time]
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=0, drop=True)
    bars = bars.rename(columns={
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","trade_count":"Trades","vwap":"VWAP"
    })
    bars = bars[["Open","High","Low","Close","Volume","VWAP"]].sort_index()
    # Expand indices to have all the timesteps
    filled_index = pd.date_range(start=bars.index.min().tz_convert("EST"), end=pd.Timestamp(end, tz="EST")+pd.DateOffset(days=1)-pd.DateOffset(seconds=1), freq="1min")
    # Create expanded df for it
    expanded_bars = bars.reindex(filled_index)
    # Find which rows were added
    new_rows = ~expanded_bars.index.isin(bars.index)
    # Forward fill all the close value that are empty since no movement means no bar info
    prev_close = expanded_bars['Close'].ffill()
    filled_bars = expanded_bars.copy()
    # Set all price values to close of the last active minute since it wouldn't have changed
    filled_bars.loc[new_rows, ["Open","High","Low","Close"]] = prev_close[new_rows].values.reshape(-1,1)
    # Set volume to 0
    filled_bars.loc[new_rows,"Volume"] = 0
    # Back fill any missing bars before the first trade of the month and filling any vwap values left over (since VWAP doesn't change if price doesn't)
    filled_bars = filled_bars.bfill().ffill()
    return filled_bars

def _month_bounds(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = ts.normalize().replace(day=1)
    # next month start
    if ts.month == 12:
        next_start = start.replace(year=ts.year + 1, month=1)
    else:
        next_start = start.replace(month=ts.month + 1)
    end = next_start - pd.Timedelta(seconds=1)
    return start.tz_convert("EST"), end.tz_convert("EST")

def _iterate_months(start: str, end: str):
    s = pd.Timestamp(start, tz="EST")
    e = pd.Timestamp(end, tz="EST")
    m_start, m_end = _month_bounds(s)
    while m_end <= e:
        yield m_start, m_end
        m_start, m_end = m_start + pd.DateOffset(months=1), m_start + pd.DateOffset(months=2) - pd.DateOffset(seconds=1)

def load_or_fetch_monthly(symbol: str, start: str, end: str, timeframe: str = "1Min", cache_dir: str | Path = "data_cache") -> pd.DataFrame:
    """
    Load historical bars by month from a local cache. For each month in [start, end],
    if a monthly parquet exists, load it; otherwise fetch from Alpaca and cache it.
    Returns a concatenated DataFrame for the full period without duplicates.
    """
    cache_dir = Path(cache_dir) / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)

    monthly_dfs = []
    for m_start, m_end in _iterate_months(start, end):
        key = f"{symbol}_{m_start.strftime('%Y-%m')}_{timeframe}.parquet"
        fpath = cache_dir / key
        if fpath.exists():
            df = pd.read_parquet(fpath)
        else:
            df = fetch_alpaca_bars(symbol, str(m_start), str(m_end), timeframe)
            df.to_parquet(fpath)
        monthly_dfs.append(df)

    if not monthly_dfs:
        raise RuntimeError("No data loaded for the requested period")
    out = pd.concat(monthly_dfs).sort_index()
    # De-duplicate overlapping boundaries
    out = out[~out.index.duplicated(keep="last")]
    # Clip to exact [start, end]
    s = pd.Timestamp(start, tz="EST")
    e = pd.Timestamp(end, tz="EST")
    out = out.loc[(out.index >= s) & (out.index <= e)]
    return out

def validate_data_cache(symbol: str, start:str, end:str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    curr = s
    while(curr.month != e.month):
        validating = pd.read_parquet(f"data_cache/{symbol.capitalize()}/{symbol.capitalize()}_{curr.year}-{curr.month:02}_1Min.parquet")
        cross_reference = fetch_alpaca_bars(symbol, f"{curr.year}-{curr.month}-01", f"{curr.year}-{curr.month}-{curr.days_in_month}")

if __name__=="__main__":
    main()