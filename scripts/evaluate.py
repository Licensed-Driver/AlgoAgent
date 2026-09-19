import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from rl_trader.features import build_feature_matrix
from rl_trader.utils import scale_features, apply_stats
from rl_trader.backtest import run_backtest
from rl_trader.config import DataConfig, EnvConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to .zip model file")
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--timeframe", default="1Min")
    ap.add_argument("--cache_dir", default="data_cache")
    ap.add_argument("--algo", default="recurrent", choices=["ppo", "recurrent"])
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic action selection (default is deterministic)")
    ap.add_argument("--stats_path", default="logs/saves/feature_stats.json", help="Path to feature stats json")
    ap.add_argument("--vecnorm_path", default="logs/saves/vecnormalize.pkl", help="Path to vecnormalize pickle")
    ap.add_argument("--live", action="store_true", help="Show live equity curve")

    args = ap.parse_args()

    # Load data
    print(f"Loading data for {args.symbol}...")
    from rl_trader.data import load_or_fetch_monthly
    df = load_or_fetch_monthly(args.symbol, args.start, args.end, args.timeframe, cache_dir=args.cache_dir)
    print(f"Data loaded. Shape: {df.shape}")

    # Features
    print("Building features...")
    X = build_feature_matrix(df)
    prices = df[['Close', 'Open']].loc[X.index]

    # Scale with saved stats if we have them
    if os.path.exists(args.stats_path):
        print(f"Loading feature stats from {args.stats_path}...")
        with open(args.stats_path, "r") as f:
            stats = json.load(f)
        stats_series = {
            "mean": pd.Series(stats["mean"]),
            "std": pd.Series(stats["std"])
        }
        X_s = apply_stats(X, stats_series)
    else:
        print("WARNING: No feature stats found. Scaling on full dataset (leaky but ok for quick check).")
        X_s, _ = scale_features(X)

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.algo == "recurrent":
        model = RecurrentPPO.load(args.model_path, device="auto")
    else:
        model = PPO.load(args.model_path, device="auto")

    # Backtest
    print("Running backtest...")
    vecnorm = args.vecnorm_path if os.path.exists(args.vecnorm_path) else None
    if vecnorm:
        print(f"Using VecNormalize stats from {vecnorm}")

    # Test on the last 20%
    n = len(prices)
    split_idx = int(n * 0.8)
    prices_test = prices.iloc[split_idx:]
    X_test = X_s.iloc[split_idx:]

    eq = run_backtest(model, prices_test, X_test,
                      initial_equity=10000.0,
                      max_position_pct=0.5,
                      vecnorm_path=vecnorm,
                      live_plot=args.live,
                      deterministic=not args.stochastic,
                      step_size=EnvConfig.step_size)

    # Plot
    print("Plotting...")
    plt.figure(figsize=(12, 6))
    eq.plot()
    plt.title(f"Evaluation: {args.symbol} (Last 20% of data)")
    plt.ylabel("Equity")
    plt.savefig("artifacts/manual_eval.png")
    print("Saved plot to artifacts/manual_eval.png")

    eq.to_csv("artifacts/manual_eval.csv")
    print("Saved CSV to artifacts/manual_eval.csv")

if __name__ == "__main__":
    main()
