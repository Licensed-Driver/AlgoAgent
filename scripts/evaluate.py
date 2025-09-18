import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rl_trader.features import build_feature_matrix
from rl_trader.utils import apply_stats, scale_features
from rl_trader.backtest import run_backtest
from stable_baselines3 import PPO
from rl_trader.metrics import basic_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet from fetch_data (eval period)")
    ap.add_argument("--model", required=True, help="SB3 .zip model file")
    ap.add_argument("--spread_bps", type=float, default=None)
    ap.add_argument("--slippage_bps", type=float, default=None)
    ap.add_argument("--initial_equity", type=float, default=None)
    ap.add_argument("--reward_mode", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if df.index.tz is not None:
        df = df.tz_convert("UTC")
    prices = df["Close"]
    X = build_feature_matrix(df)

    # Here we assume no saved stats; for a real pipeline, persist and reuse stats per split.
    Xs, stats = scale_features(X)

    # Load config defaults for env parameters
    from rl_trader.config import EnvConfig
    env_cfg = EnvConfig()
    if args.initial_equity is None: args.initial_equity = env_cfg.initial_equity
    if args.spread_bps is None: args.spread_bps = env_cfg.spread_bps
    if args.slippage_bps is None: args.slippage_bps = env_cfg.slippage_bps
    if args.reward_mode is None: args.reward_mode = env_cfg.reward_mode

    model = PPO.load(args.model)
    eq = run_backtest(model, prices, Xs,
                      initial_equity=args.initial_equity,
                      spread_bps=args.spread_bps,
                      slippage_bps=args.slippage_bps,
                      max_position_pct=env_cfg.max_position_pct,
                      reward_mode=args.reward_mode)
    stats = basic_stats(eq)
    print("Stats:", stats)
    # Plot
    plt.figure()
    eq.plot(title="Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    out = Path("equity_curve.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")
    # Save equity series for debugging/inspection
    eq.to_csv(Path("equity_curve.csv"))

if __name__ == "__main__":
    main()
