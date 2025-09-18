import argparse
import pandas as pd
from rl_trader.data import load_or_fetch_monthly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--timeframe", default="1Min")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache_dir", default=None, help="Directory to store monthly cached parquet files")
    args = ap.parse_args()

    # Load default cache_dir from config if not provided
    from rl_trader.config import DataConfig
    data_cfg = DataConfig(symbol=args.symbol, start=args.start, end=args.end)
    if args.cache_dir is None:
        args.cache_dir = getattr(data_cfg, "cache_dir", "data_cache")
    df = load_or_fetch_monthly(args.symbol, args.start, args.end, args.timeframe, cache_dir=args.cache_dir)
    out = args.out or f"data_{args.symbol}_{args.start}_{args.end}_{args.timeframe}.parquet"
    df.to_parquet(out)
    print(f"Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
