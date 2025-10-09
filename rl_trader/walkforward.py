import pandas as pd
import numpy as np
from .utils import rolling_windows, scale_features, apply_stats
from .env import SingleTickerEnv
from .agent import train_ppo

def walk_forward(
    prices: pd.Series,
    X: pd.DataFrame,
    total_timesteps: int = 150_000,
    train_days=30,
    valid_days=7,
    test_days=7,
    stride_days=7,
    env_kwargs=None,
    ppo_kwargs=None,
    max_splits: int | None = None,
):
    env_kwargs = env_kwargs or {}
    ppo_kwargs = ppo_kwargs or {}

    results = []
    for i, (df_tr, df_va, df_te) in enumerate(
        rolling_windows(pd.concat([prices, X], axis=1), train_days, valid_days, test_days, stride_days)
    ):
        if max_splits is not None and i >= max_splits:
            break
        p_tr, x_tr = df_tr.iloc[:,0], df_tr.iloc[:,1:]
        p_va, x_va = df_va.iloc[:,0], df_va.iloc[:,1:]
        p_te, x_te = df_te.iloc[:,0], df_te.iloc[:,1:]

        x_tr_s, stats = scale_features(x_tr)
        x_va_s = apply_stats(x_va, stats)
        x_te_s = apply_stats(x_te, stats)

        def make_env_tr(rank, seed=42):
            def _thunk():
                env = SingleTickerEnv(p_tr, x_tr_s, **env_kwargs)
                env.reset(rank + seed)
                return env
            return _thunk
        def make_env_va(rank, seed=42):
            def _thunk():
                env = SingleTickerEnv(p_va, x_va_s, **env_kwargs)
                env.reset(rank + seed)
                return env
            return _thunk

        model = train_ppo(
            make_env_tr,
            total_timesteps=total_timesteps,
            log_dir=f"./logs/wf_{i}",
            eval_env_fn=make_env_va,
            **ppo_kwargs,
        )

        # Evaluate on test
        env_te = SingleTickerEnv(p_te, x_te_s, **env_kwargs)
        obs, _ = env_te.reset()
        done = False
        eq_hist = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_te.step(action)
            eq_hist.append(info.get("equity", np.nan))
        eq = pd.Series(eq_hist, index=x_te_s.index[1:len(eq_hist)+1])
        results.append(eq)
    return results
