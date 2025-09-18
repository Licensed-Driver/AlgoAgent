import argparse
import pandas as pd
from pathlib import Path
from rl_trader.features import build_feature_matrix
from rl_trader.utils import scale_features
from rl_trader.env import SingleTickerEnv
from rl_trader.agent import train_ppo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Parquet file from scripts.fetch_data")
    ap.add_argument("--symbol", help="If --data not provided, fetch params", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--total_timesteps", type=int, default=None)
    ap.add_argument("--spread_bps", type=float, default=None)
    ap.add_argument("--slippage_bps", type=float, default=None)
    ap.add_argument("--initial_equity", type=float, default=None)
    ap.add_argument("--reward_mode", default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_freq", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_position_pct", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gae_lambda", type=float, default=None)
    ap.add_argument("--clip_range", type=float, default=None)
    ap.add_argument("--ent_coef", type=float, default=None)
    ap.add_argument("--vf_coef", type=float, default=None)
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_epochs", type=int, default=None)
    # Fee model options
    ap.add_argument("--fee_model", choices=["fixed","tiered"], default=None)
    ap.add_argument("--per_share", type=float, default=None)
    ap.add_argument("--min_per_order", type=float, default=None)
    ap.add_argument("--tiered_per_share", type=float, default=None)
    ap.add_argument("--tiered_min_order", type=float, default=None)
    ap.add_argument("--sec_fee_per_dollar", type=float, default=None)
    ap.add_argument("--taf_fee_per_share", type=float, default=None)
    args = ap.parse_args()

    # Always load defaults from config.py, then let CLI override if provided
    from rl_trader.config import EnvConfig, FeeConfig, PPOConfig, DataConfig
    env_cfg = EnvConfig(); fee_cfg = FeeConfig(); ppo_cfg = PPOConfig(); data_cfg = DataConfig(symbol="", start="", end="")
    # Data defaults
    if args.symbol is None: args.symbol = data_cfg.symbol
    if args.start is None: args.start = data_cfg.start
    if args.end is None: args.end = data_cfg.end
    if args.timeframe is None: args.timeframe = data_cfg.timeframe
    if args.cache_dir is None: args.cache_dir = getattr(data_cfg, "cache_dir", "data_cache")
    # Env defaults
    if args.initial_equity is None: args.initial_equity = env_cfg.initial_equity
    if args.spread_bps is None: args.spread_bps = env_cfg.spread_bps
    if args.slippage_bps is None: args.slippage_bps = env_cfg.slippage_bps
    if args.reward_mode is None: args.reward_mode = env_cfg.reward_mode
    if args.reward_scale is None:
        rs = env_cfg.reward_scale
        if isinstance(rs, str):
            if rs.lower() == "initial_equity":
                args.reward_scale = args.initial_equity
            elif rs.lower() in ("none", "null"):
                args.reward_scale = None
            else:
                try:
                    args.reward_scale = float(rs)
                except Exception:
                    args.reward_scale = None
        else:
            args.reward_scale = rs
    # Fee defaults from config if CLI not provided
    if args.fee_model is None: args.fee_model = fee_cfg.model
    if args.per_share is None: args.per_share = fee_cfg.per_share
    if args.min_per_order is None: args.min_per_order = fee_cfg.min_per_order
    if args.tiered_per_share is None: args.tiered_per_share = fee_cfg.tiered_per_share
    if args.tiered_min_order is None: args.tiered_min_order = fee_cfg.tiered_min_order
    if args.sec_fee_per_dollar is None: args.sec_fee_per_dollar = fee_cfg.sec_fee_per_dollar
    if args.taf_fee_per_share is None: args.taf_fee_per_share = fee_cfg.taf_fee_per_share

    # PPO defaults from config if CLI not provided
    if args.total_timesteps is None: args.total_timesteps = ppo_cfg.total_timesteps
    if args.eval_freq is None: args.eval_freq = ppo_cfg.eval_freq
    if args.learning_rate is None: args.learning_rate = ppo_cfg.learning_rate
    if args.device is None: args.device = ppo_cfg.device

    if args.data:
        df = pd.read_parquet(args.data)
    else:
        from rl_trader.data import load_or_fetch_monthly
        assert args.symbol and args.start and args.end, "Provide --data or (symbol, start, end)"
        df = load_or_fetch_monthly(args.symbol, args.start, args.end, args.timeframe, cache_dir=args.cache_dir)

    df = df.tz_convert("UTC") if df.index.tz is not None else df
    X = build_feature_matrix(df)
    prices = df["Close"]
    Xs, stats = scale_features(X)

    fee_kwargs = dict(
        model=args.fee_model,
        per_share=args.per_share,
        min_per_order=args.min_per_order,
        tiered_per_share=args.tiered_per_share,
        tiered_min_order=args.tiered_min_order,
        sec_fee_per_dollar=args.sec_fee_per_dollar,
        taf_fee_per_share=args.taf_fee_per_share,
    )

    def make_env():
        return SingleTickerEnv(prices, Xs,
                               initial_equity=args.initial_equity,
                               spread_bps=args.spread_bps,
                               slippage_bps=args.slippage_bps,
                               max_position_pct=(args.max_position_pct if args.max_position_pct is not None else env_cfg.max_position_pct),
                               reward_mode=args.reward_mode,
                               reward_scale=args.reward_scale,
                               fee_kwargs=fee_kwargs)

    # PPO defaults from config if CLI not provided
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

    model = train_ppo(
        make_env,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        log_dir="./logs",
        seed=args.seed,
        device=args.device,
        eval_freq=args.eval_freq,
    )
    Path("models").mkdir(exist_ok=True)
    out = Path("models") / "ppo_single_ticker.zip"
    model.save(str(out))
    print(f"Saved model to {out}")

if __name__ == "__main__":
    main()
