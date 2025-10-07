import numpy as np
import pandas as pd
from .env import SingleTickerEnv
from stable_baselines3.common.vec_env import VecNormalize

def run_backtest(
    model,
    prices: pd.Series,
    features: pd.DataFrame,
    initial_equity: float = 10_000.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 0.0,
    max_position_pct: float = 1.0,
    reward_mode: str = "pnl",
    vecnorm_path: str | None = None,   # path to saved VecNormalize, if used in training
    deterministic: bool = True,
):
    # Build a plain, single env (no randomness/windowing for backtest)
    env = SingleTickerEnv(
        prices=prices,
        features=features,
        initial_equity=initial_equity,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        max_position_pct=max_position_pct,
        reward_mode=reward_mode,
    )

    # If you trained with VecNormalize, load stats and wrap for eval
    if vecnorm_path:
        # === VecEnv path (4-return API) ===
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

        obs = venv.reset()
        eq_hist, ts_index = [], []

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)  # shape (1,)
            obs, rewards, dones, infos = venv.step(action)               # <-- 4 items
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            if dones[0]:   # VecEnv has a single "done" flag; truncation info may be in infos
                break
            eq_hist.append(info0.get("equity", np.nan))
            ts_index.append(prices.index[env._i - 1])

        return pd.Series(eq_hist, index=pd.Index(ts_index, name=prices.index.name))

    else:
        # === Raw env path (5-return Gymnasium API) ===
        obs, _ = env.reset()
        eq_hist, ts_index = [], []

        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)  # scalar action ok
            obs, reward, done, truncated, info = env.step(action)       # <-- 5 items
            if(done or truncated):
                break
            eq_hist.append(info.get("equity", np.nan))
            ts_index.append(prices.index[env._i - 1])

        return pd.Series(eq_hist, index=pd.Index(ts_index, name=prices.index.name))
