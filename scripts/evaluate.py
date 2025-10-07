import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rl_trader.features import build_feature_matrix
from rl_trader.utils import apply_stats  # use saved stats
from rl_trader.backtest import run_backtest  # ensure this handles truncated + vecnorm
from stable_baselines3 import PPO
from rl_trader.metrics import basic_stats
from rl_trader.config import EnvConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)           # SB3 .zip
    ap.add_argument("--feature_stats", required=False)  # JSON/PKL from training
    ap.add_argument("--vecnorm", required=False)        # vecnorm.pkl from training
    ap.add_argument("--spread_bps", type=float)
    ap.add_argument("--slippage_bps", type=float)
    ap.add_argument("--initial_equity", type=float)
    ap.add_argument("--reward_mode")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).sort_index()
    if df.index.tz is not None:
        df = df.tz_convert("UTC")

    prices = df["Close"]
    X = build_feature_matrix(df)

    # Apply saved feature scaling stats
    if args.feature_stats:
        feat_stats = pd.read_json(args.feature_stats)
        Xs = apply_stats(X, feat_stats).astype("float32")
    else:
        # fallback (not recommended for real eval): drop NaNs and cast
        Xs = X.astype("float32")

    # Align and drop NaNs jointly
    aligned = pd.concat([prices, Xs], axis=1).dropna()
    prices = aligned.iloc[:, 0]
    Xs     = aligned.iloc[:, 1:]
    assert prices.index.equals(Xs.index)

    # Env defaults
    env_cfg = EnvConfig()
    initial_equity = args.initial_equity if args.initial_equity is not None else env_cfg.initial_equity
    spread_bps     = args.spread_bps     if args.spread_bps     is not None else env_cfg.spread_bps
    slippage_bps   = args.slippage_bps   if args.slippage_bps   is not None else env_cfg.slippage_bps
    reward_mode    = args.reward_mode    if args.reward_mode    is not None else env_cfg.reward_mode

    # Load model
    model = PPO.load(args.model, device="cpu")  # device here only affects additional training; predict works on CPU

    # Backtest (pass vecnorm path if used in training)
    eq = run_backtest(
        model,
        prices,
        Xs,
        initial_equity=initial_equity,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        max_position_pct=env_cfg.max_position_pct,
        reward_mode=reward_mode,
        vecnorm_path=args.vecnorm if args.vecnorm else None,
        deterministic=True,
    )

    stats = basic_stats(eq)
    print("Stats:", stats)

    # Plot + save
    out_png = Path("equity_curve.png")
    out_csv = Path("equity_curve.csv")
    plt.figure(figsize=(10,5))
    eq.plot(title="Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved {out_png}")
    eq.to_csv(out_csv)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
