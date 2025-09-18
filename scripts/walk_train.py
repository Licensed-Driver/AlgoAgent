import argparse
import pandas as pd
from pathlib import Path
from rl_trader.features import build_feature_matrix
from rl_trader.utils import scale_features
from rl_trader.walkforward import walk_forward
from rl_trader.metrics import basic_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Parquet file (full period)")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--train_days", type=int, default=None)
    ap.add_argument("--valid_days", type=int, default=None)
    ap.add_argument("--test_days", type=int, default=None)
    ap.add_argument("--stride_days", type=int, default=None)
    ap.add_argument("--max_splits", type=int, default=None)
    ap.add_argument("--total_timesteps", type=int, default=150_000)
    ap.add_argument("--spread_bps", type=float, default=2.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--initial_equity", type=float, default=None)
    ap.add_argument("--reward_mode", default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_freq", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gae_lambda", type=float, default=None)
    ap.add_argument("--clip_range", type=float, default=None)
    ap.add_argument("--ent_coef", type=float, default=None)
    ap.add_argument("--vf_coef", type=float, default=None)
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_epochs", type=int, default=None)
    # Fee model options
    ap.add_argument("--fee_model", choices=["fixed","tiered"], default="fixed")
    ap.add_argument("--per_share", type=float, default=0.005)
    ap.add_argument("--min_per_order", type=float, default=1.0)
    ap.add_argument("--tiered_per_share", type=float, default=0.0035)
    ap.add_argument("--tiered_min_order", type=float, default=0.35)
    ap.add_argument("--sec_fee_per_dollar", type=float, default=0.0)
    ap.add_argument("--taf_fee_per_share", type=float, default=0.0)
    args = ap.parse_args()

    # Load defaults from config.py and let CLI override
    from rl_trader.config import EnvConfig, FeeConfig, PPOConfig, DataConfig, WalkConfig
    env_cfg = EnvConfig(); fee_cfg = FeeConfig(); ppo_cfg = PPOConfig(); data_cfg = DataConfig(symbol="", start="", end=""); walk_cfg = WalkConfig()
    # Data
    if args.symbol is None: args.symbol = data_cfg.symbol
    if args.start is None: args.start = data_cfg.start
    if args.end is None: args.end = data_cfg.end
    if args.timeframe is None: args.timeframe = data_cfg.timeframe
    if args.cache_dir is None: args.cache_dir = getattr(data_cfg, "cache_dir", "data_cache")
    # Walk
    if args.train_days is None: args.train_days = walk_cfg.train_days
    if args.valid_days is None: args.valid_days = walk_cfg.valid_days
    if args.test_days is None: args.test_days = walk_cfg.test_days
    if args.stride_days is None: args.stride_days = walk_cfg.stride_days
    if args.max_splits is None: args.max_splits = walk_cfg.max_splits

    if args.data:
        df = pd.read_parquet(args.data)
    else:
        from rl_trader.data import load_or_fetch_monthly
        assert args.symbol and args.start and args.end, "Provide --data or (symbol, start, end)"
        df = load_or_fetch_monthly(args.symbol, args.start, args.end, args.timeframe, cache_dir=args.cache_dir)

    if df.index.tz is not None:
        df = df.tz_convert("UTC")
    prices = df["Close"]
    X = build_feature_matrix(df)

    fee_kwargs = dict(
        model=args.fee_model,
        per_share=args.per_share,
        min_per_order=args.min_per_order,
        tiered_per_share=args.tiered_per_share,
        tiered_min_order=args.tiered_min_order,
        sec_fee_per_dollar=args.sec_fee_per_dollar,
        taf_fee_per_share=args.taf_fee_per_share,
    )
    env_kwargs = dict(initial_equity=args.initial_equity, spread_bps=args.spread_bps,
                      slippage_bps=args.slippage_bps, max_position_pct=env_cfg.max_position_pct, reward_mode=args.reward_mode,
                      reward_scale=args.reward_scale, fee_kwargs=fee_kwargs)

    # PPO defaults
    if args.total_timesteps is None: args.total_timesteps = ppo_cfg.total_timesteps
    if args.eval_freq is None: args.eval_freq = ppo_cfg.eval_freq
    if args.learning_rate is None: args.learning_rate = ppo_cfg.learning_rate
    if args.device is None: args.device = ppo_cfg.device
    if args.gamma is None: args.gamma = ppo_cfg.gamma
    if args.gae_lambda is None: args.gae_lambda = ppo_cfg.gae_lambda
    if args.clip_range is None: args.clip_range = ppo_cfg.clip_range
    if args.ent_coef is None: args.ent_coef = ppo_cfg.ent_coef
    if args.vf_coef is None: args.vf_coef = ppo_cfg.vf_coef
    if args.n_steps is None: args.n_steps = ppo_cfg.n_steps
    if args.batch_size is None: args.batch_size = ppo_cfg.batch_size
    if args.n_epochs is None: args.n_epochs = ppo_cfg.n_epochs

    eq_list = walk_forward(prices, X, total_timesteps=args.total_timesteps,
                           train_days=args.train_days, valid_days=args.valid_days,
                           test_days=args.test_days, stride_days=args.stride_days,
                           env_kwargs=env_kwargs, ppo_kwargs={
                               "seed": args.seed,
                               "device": (args.device or "cpu"),
                               "eval_freq": args.eval_freq,
                               "learning_rate": args.learning_rate,
                               "gamma": args.gamma,
                               "gae_lambda": args.gae_lambda,
                               "clip_range": args.clip_range,
                               "ent_coef": args.ent_coef,
                               "vf_coef": args.vf_coef,
                               "n_steps": args.n_steps,
                               "batch_size": args.batch_size,
                               "n_epochs": args.n_epochs,
                           },
                           max_splits=args.max_splits)

    # Save summary
    out_dir = Path("walk_forward")
    out_dir.mkdir(exist_ok=True)
    metrics_rows = []
    for i, eq in enumerate(eq_list):
        eq.to_csv(out_dir / f"equity_curve_split_{i}.csv")
        row = {"split": i}
        row.update(basic_stats(eq))
        metrics_rows.append(row)
    import pandas as pd
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"Wrote {len(eq_list)} equity curves and metrics.csv to {out_dir}/")

if __name__ == "__main__":
    main()
