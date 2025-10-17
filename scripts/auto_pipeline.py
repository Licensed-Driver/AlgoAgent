import argparse
import glob
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

from rl_trader.data import fetch_alpaca_bars
from rl_trader.features import build_feature_matrix
from rl_trader.utils import scale_features, apply_stats
from rl_trader.env import SingleTickerEnv
from rl_trader.agent import train_ppo
from rl_trader.backtest import run_backtest
from rl_trader.walkforward import walk_forward
from rl_trader.metrics import basic_stats
from collections.abc import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

def main():
    ap = argparse.ArgumentParser(description="End-to-end RL trading pipeline (fetch → train → evaluate → walk-forward)")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--cache_dir", default=None)

    # Training (CLI overrides config defaults)
    ap.add_argument("--total_timesteps", type=int, default=None)
    ap.add_argument("--initial_equity", type=float, default=None)
    ap.add_argument("--spread_bps", type=float, default=None)
    ap.add_argument("--slippage_bps", type=float, default=None)
    ap.add_argument("--reward_mode", default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_freq", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--device", default=None)
    # Fee model options
    ap.add_argument("--fee_model", choices=["fixed","tiered"], default=None)
    ap.add_argument("--per_share", type=float, default=None)
    ap.add_argument("--min_per_order", type=float, default=None)
    ap.add_argument("--tiered_per_share", type=float, default=None)
    ap.add_argument("--tiered_min_order", type=float, default=None)
    ap.add_argument("--sec_fee_per_dollar", type=float, default=None)
    ap.add_argument("--taf_fee_per_share", type=float, default=None)
    # PPO hyperparameters (optional CLI overrides)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gae_lambda", type=float, default=None)
    ap.add_argument("--clip_range", type=float, default=None)
    ap.add_argument("--ent_coef", type=float, default=None)
    ap.add_argument("--vf_coef", type=float, default=None)
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_epochs", type=int, default=None)
    ap.add_argument("--sub_procs", type=int, default=None)
    ap.add_argument("--vec_norm_reward", type=bool, default=None)
    ap.add_argument("--vec_norm_obs", type=bool, default=None)
    # Env (optional CLI overrides)
    ap.add_argument("--max_position_pct", type=float, default=None)

    # Walk-forward
    ap.add_argument("--train_days", type=int, default=30)
    ap.add_argument("--valid_days", type=int, default=7)
    ap.add_argument("--test_days", type=int, default=7)
    ap.add_argument("--stride_days", type=int, default=7)
    ap.add_argument("--skip_walk_forward", action="store_true", help="Skip walk-forward stage")
    ap.add_argument("--max_splits", type=int, default=None, help="Limit number of walk-forward splits")
    ap.add_argument("--min_episode_len", type=int, default=None)
    ap.add_argument("--max_episode_len", type=int, default=None)
    ap.add_argument("--spread_std_bps", type=float, default=None)
    ap.add_argument("--slippage_std_bps", type=float, default=None)
    ap.add_argument("--price_jitter_bps", type=float, default=None)
    ap.add_argument("--do_nothing_penalty", type=float, default=None)
    ap.add_argument("--double_action_penalty", type=float, default=None)
    args = ap.parse_args()

    # Load defaults from config.py, then let CLI override
    from rl_trader.config import EnvConfig, FeeConfig, PPOConfig, DataConfig, WalkConfig
    env_cfg = EnvConfig(); fee_cfg = FeeConfig(); ppo_cfg = PPOConfig(); data_cfg = DataConfig(symbol="", start="", end=""); walk_cfg = WalkConfig()
    # Data defaults
    if args.symbol is None: args.symbol = data_cfg.symbol
    if args.start is None: args.start = data_cfg.start
    if args.end is None: args.end = data_cfg.end
    if args.timeframe is None: args.timeframe = data_cfg.timeframe
    if args.cache_dir is None: args.cache_dir = getattr(data_cfg, "cache_dir", "data_cache")
    # Env
    if args.initial_equity is None: args.initial_equity = env_cfg.initial_equity
    if args.spread_bps is None: args.spread_bps = env_cfg.spread_bps
    if args.slippage_bps is None: args.slippage_bps = env_cfg.slippage_bps
    if args.reward_mode is None: args.reward_mode = env_cfg.reward_mode
    if args.max_position_pct is None: args.max_position_pct = env_cfg.max_position_pct
    if args.min_episode_len is None: args.min_episode_len = env_cfg.min_episode_len
    if args.max_episode_len is None: args.max_episode_len = env_cfg.max_episode_len
    if args.spread_std_bps is None: args.spread_std_bps = env_cfg.spread_std_bps
    if args.slippage_std_bps is None: args.slippage_std_bps = env_cfg.slippage_std_bps
    if args.price_jitter_bps is None: args.price_jitter_bps = env_cfg.price_jitter_bps
    if args.do_nothing_penalty is None: args.do_nothing_penalty = env_cfg.do_nothing_penalty
    if args.double_action_penalty is None: args.double_action_penalty = env_cfg.double_action_penalty
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
    # Fee
    if args.fee_model is None: args.fee_model = fee_cfg.model
    if args.per_share is None: args.per_share = fee_cfg.per_share
    if args.min_per_order is None: args.min_per_order = fee_cfg.min_per_order
    if args.tiered_per_share is None: args.tiered_per_share = fee_cfg.tiered_per_share
    if args.tiered_min_order is None: args.tiered_min_order = fee_cfg.tiered_min_order
    if args.sec_fee_per_dollar is None: args.sec_fee_per_dollar = fee_cfg.sec_fee_per_dollar
    if args.taf_fee_per_share is None: args.taf_fee_per_share = fee_cfg.taf_fee_per_share
    # PPO
    if args.total_timesteps is None: args.total_timesteps = ppo_cfg.total_timesteps
    if args.eval_freq is None: args.eval_freq = ppo_cfg.eval_freq
    if args.learning_rate is None: args.learning_rate = ppo_cfg.learning_rate
    if args.device is None: args.device = ppo_cfg.device
    if args.sub_procs is None: args.sub_procs = ppo_cfg.n_envs
    if args.vec_norm_reward is None: args.vec_norm_reward = ppo_cfg.vec_norm_reward
    if args.vec_norm_obs is None: args.vec_norm_obs = ppo_cfg.vec_norm_obs
    # Extra PPO hparams (use if present in config)
    args.gamma = getattr(args, "gamma", None)
    args.gae_lambda = getattr(args, "gae_lambda", None)
    args.clip_range = getattr(args, "clip_range", None)
    args.ent_coef = getattr(args, "ent_coef", None)
    args.vf_coef = getattr(args, "vf_coef", None)
    args.n_steps = getattr(args, "n_steps", None)
    args.batch_size = getattr(args, "batch_size", None)
    args.n_epochs = getattr(args, "n_epochs", None)
    if args.gamma is None: args.gamma = ppo_cfg.gamma
    if args.gae_lambda is None: args.gae_lambda = ppo_cfg.gae_lambda
    if args.clip_range is None: args.clip_range = ppo_cfg.clip_range
    if args.ent_coef is None: args.ent_coef = ppo_cfg.ent_coef
    if args.vf_coef is None: args.vf_coef = ppo_cfg.vf_coef
    if args.n_steps is None: args.n_steps = ppo_cfg.n_steps
    if args.batch_size is None: args.batch_size = ppo_cfg.minibatch_size
    if args.n_epochs is None: args.n_epochs = ppo_cfg.n_epochs
    # Walk-forward defaults
    if args.train_days is None: args.train_days = walk_cfg.train_days
    if args.valid_days is None: args.valid_days = walk_cfg.valid_days
    if args.test_days is None: args.test_days = walk_cfg.test_days
    if args.stride_days is None: args.stride_days = walk_cfg.stride_days
    if args.max_splits is None: args.max_splits = walk_cfg.max_splits
    if not args.skip_walk_forward:
        args.skip_walk_forward = walk_cfg.skip_walk_forward

    outdir = Path("artifacts")
    outdir.mkdir(exist_ok=True)
    wf_dir = outdir / "walk_forward"

    if not args.symbol or not args.start or not args.end:
        raise RuntimeError("Missing symbol/start/end (set in CLI or DataConfig)")
    print("=== 1) Fetching data (monthly cached) ===")
    try:
        from rl_trader.data import load_or_fetch_monthly
        df = load_or_fetch_monthly(args.symbol, args.start, args.end, args.timeframe, cache_dir=args.cache_dir)
    except Exception:
        raise(Exception("Data Fetch Exception."))
        # Fallback to direct fetch if loader import fails for any reason
        from rl_trader.data import fetch_alpaca_bars
        df = fetch_alpaca_bars(args.symbol, args.start, args.end, args.timeframe)
    raw_path = outdir / f"data_{args.symbol}_{args.start}_{args.end}_{args.timeframe}.parquet"
    df.to_parquet(raw_path)
    print(f"Wrote {raw_path} with {len(df)} rows.")

    print("=== 2) Building features ===")
    X = build_feature_matrix(df)
    prices = df[['Close', 'Open']]

    print("=== 3) Train on first 90% (avoid leakage) — raw profit scaled ===")
    date_index = prices.index.get_level_values("date")
    unique_dates = date_index.unique()
    n = len(unique_dates)
    split = unique_dates[min(max(int(n * 0.9), 1), n - 1)]
    prices_tr = prices.loc[:split].copy()
    X_tr = X.loc[:split].copy()
    X_tr_s, stats = scale_features(X_tr)
    prices_eval = prices.loc[split:].copy()
    X_eval = X.loc[split:].copy()
    X_eval_s = apply_stats(X_eval, stats)

    # Save scaled feature set for backtesting
    os.makedirs("logs/saves/", exist_ok=True)
    with open("logs/saves/feature_stats.json", "w") as f:
        stats_json = {
            "mean": stats["mean"].to_dict(),
            "std": stats["std"].to_dict()
        }
        json.dump(stats_json, f, indent=2)

    fee_kwargs = dict(
        model=args.fee_model,
        per_share=args.per_share,
        min_per_order=args.min_per_order,
        tiered_per_share=args.tiered_per_share,
        tiered_min_order=args.tiered_min_order,
        sec_fee_per_dollar=args.sec_fee_per_dollar,
        taf_fee_per_share=args.taf_fee_per_share,
    )

    def make_env(rank, seed=42):
        def _thunk():
            env = SingleTickerEnv(prices_tr, X_tr_s,
                               initial_equity=args.initial_equity,
                               spread_bps=args.spread_bps,
                               slippage_bps=args.slippage_bps,
                               max_position_pct=args.max_position_pct,
                               reward_mode=args.reward_mode,
                               reward_scale=args.reward_mode,
                               fee_kwargs=fee_kwargs,
                               min_episode_len=args.min_episode_len,
                               max_episode_len=args.max_episode_len,
                               spread_std_bps=args.spread_std_bps,
                               slippage_std_bps=args.slippage_std_bps,
                               price_jitter_bps=args.price_jitter_bps,
                               do_nothing_penalty=args.do_nothing_penalty,
                               double_action_penalty=args.double_action_penalty)
            env.reset(seed + rank)
            log_dir = "logs/monitor/"
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))
            return env
        return _thunk
    def make_eval_env():
        env = SingleTickerEnv(prices_eval, X_eval_s,
                              initial_equity=args.initial_equity,
                              spread_bps=args.spread_bps,
                              slippage_bps=args.slippage_bps,
                              max_position_pct=args.max_position_pct,
                              reward_mode=args.reward_mode,
                              reward_scale=args.reward_mode,
                              fee_kwargs=fee_kwargs,
                              min_episode_len=args.min_episode_len,
                              max_episode_len=args.max_episode_len,
                              spread_std_bps=args.spread_std_bps,
                              slippage_std_bps=args.slippage_std_bps,
                              price_jitter_bps=args.price_jitter_bps,
                              do_nothing_penalty=args.do_nothing_penalty,
                              double_action_penalty=args.double_action_penalty)
        env.reset(args.seed if args.seed is not None else 0)
        return env
    model = train_ppo(make_env,
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
                      log_dir="./logs/auto",
                      seed=args.seed,
                      device=args.device,
                      eval_freq=args.eval_freq,
                      sub_procs=args.sub_procs,
                      vec_norm_obs=args.vec_norm_obs,
                      vec_norm_reward=args.vec_norm_reward,
                      eval_env_fn=make_eval_env)
    
    # Plot episode rewards
    files = glob.glob("logs/monitor/*.monitor.csv")
    if files:
        monitor_dfs = [pd.read_csv(f, skiprows=1) for f in files]  # skip comment header
        monitor_df = pd.concat(monitor_dfs).sort_values("t")
        print(monitor_df.tail())
        plt.figure(figsize=(10, 5))
        plt.plot(monitor_df["t"], monitor_df["r"].rolling(20).mean(), label="20-episode rolling mean")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("Episode Rewards Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("artifacts/episode_reward.png", dpi=200, bbox_inches="tight")
        plt.close()

    model_dir = Path("models"); model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ppo_auto_single_ticker.zip"
    model.save(str(model_path))
    print(f"Saved model → {model_path}")

    print("=== 4) Evaluating on last 10% of the window ===")
    vecnorm_path = Path("logs/saves/vecnormalize.pkl")
    eq = run_backtest(model, prices_eval, X_eval_s,
                      initial_equity=args.initial_equity,
                      spread_bps=args.spread_bps,
                      slippage_bps=args.slippage_bps,
                      max_position_pct=args.max_position_pct,
                      reward_mode=args.reward_mode,
                      vecnorm_path=str(vecnorm_path) if vecnorm_path.exists() else None)

    plt.figure(figsize=(10, 5))
    #plt.scatter(eq.index, eq.values, label=f"Equity Curve (eval) — {args.symbol}", s=5)
    eq.plot()
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.tight_layout()
    eq_path = outdir / "equity_curve_eval.png"
    plt.savefig(eq_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved equity curve → {eq_path}")


    # Also save raw equity series for inspection
    eq_csv_path = outdir / "equity_curve_eval.csv"
    eq.to_csv(eq_csv_path)
    print(f"Saved equity series → {eq_csv_path}")
    stats = basic_stats(eq)
    with open(outdir / "eval_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved eval stats →", outdir / "eval_stats.json")

    if not args.skip_walk_forward:
        print("=== 5) Walk-forward training & test evaluation ===")
        env_kwargs = dict(initial_equity=args.initial_equity,
                          spread_bps=args.spread_bps,
                          slippage_bps=args.slippage_bps,
                          max_position_pct=args.max_position_pct,
                          reward_mode=args.reward_mode,
                          reward_scale=(args.initial_equity if args.reward_scale is None else args.reward_scale),
                          fee_kwargs=fee_kwargs)
        eq_list = walk_forward(prices, X, total_timesteps=max(150_000, args.total_timesteps//2),
                               train_days=args.train_days, valid_days=args.valid_days,
                               test_days=args.test_days, stride_days=args.stride_days,
                               env_kwargs=env_kwargs, ppo_kwargs={
                                   "seed": args.seed, "device": args.device, "eval_freq": args.eval_freq,
                                   "learning_rate": args.learning_rate, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                                   "clip_range": args.clip_range, "ent_coef": args.ent_coef, "vf_coef": args.vf_coef,
                                   "n_steps": args.n_steps, "batch_size": args.batch_size, "n_epochs": args.n_epochs
                               },
                               max_splits=args.max_splits)
        wf_dir.mkdir(exist_ok=True)
        metrics_rows = []
        for i, series in enumerate(eq_list):
            series.to_csv(wf_dir / f"equity_curve_split_{i}.csv")
            row = {"split": i}
            row.update(basic_stats(series))
            metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(wf_dir / "metrics.csv", index=False)
        print(f"Saved {len(eq_list)} walk-forward equity curves and metrics.csv to {wf_dir}")
    else:
        print("=== 5) Walk-forward skipped (per flag) ===")

    print("""
=== Done ===
Artifacts:
- Raw data: {raw}
- Model: {model}
- Eval curve: {eq}
""".format(raw=raw_path, model=model_path, eq=eq_path))
    if not args.skip_walk_forward:
        print(f"- Walk-forward curves: {wf_dir}/equity_curve_split_*.csv")
    else:
        print("- Walk-forward: skipped")

if __name__ == "__main__":
    main()
