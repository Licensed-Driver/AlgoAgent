from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    symbol: str
    start: str
    end: str
    timeframe: str = "1Min"
    limit: Optional[int] = None
    cache_dir: str = "data_cache"

@dataclass
class EnvConfig:
    window: int = 1
    spread_bps: float = 2.0  # bid/ask spread in basis points (0.01% = 1 bps)
    slippage_bps: float = 0.0
    initial_equity: float = 5000.0
    max_position_pct: float = 0.5  # long-only, cannot exceed equity
    reward_mode: str = "pnl"  # 'pnl', 'pnl_raw', 'logpnl', 'sharpe_step'
    # Can be a float-like string (e.g., "10000"), "initial_equity", or "none"/"null"
    reward_scale: str = "initial_equity"

@dataclass
class FeeConfig:
    model: str = "fixed"  # 'fixed' or 'tiered'
    per_share: float = 0.005
    min_per_order: float = 1.0
    tiered_per_share: float = 0.0035
    tiered_min_order: float = 0.35
    sec_fee_per_dollar: float = 0.0
    taf_fee_per_share: float = 0.0

@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    learning_rate: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_steps: int = 4096
    batch_size: int = 512
    n_epochs: int = 20
    eval_freq: int = 10000
    device: str = "cpu"  # 'cpu' or 'cuda'

@dataclass
class WalkConfig:
    train_days: int = 30
    valid_days: int = 7
    test_days: int = 7
    stride_days: int = 7
    skip_walk_forward: bool = False
    max_splits: Optional[int] = None
