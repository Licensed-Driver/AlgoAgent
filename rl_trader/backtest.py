import numpy as np
import pandas as pd
from .env import SingleTickerEnv
from stable_baselines3.common.vec_env import VecNormalize

def run_backtest(
    model,
    prices: pd.Series,
    features: pd.DataFrame,
    initial_equity: float = 5_000.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 0.0,
    max_position_pct: float = 1.0,
    reward_mode: str = "pnl",
    vecnorm_path: str | None = None,   # path to saved VecNormalize, if used in training
    deterministic: bool = True,
    live_plot: bool = False,
    step_size: int = 1,
):
    # Only import matplotlib if we're plotting
    push_frame = None
    if live_plot:
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            print("Warning: could not select the TkAgg backend, plotting may fail.")

        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        line_eq, = ax.plot([], [], label="Agent Equity", color="blue")
        line_bh, = ax.plot([], [], label="Buy & Hold", color="gray", linestyle="--")
        ax.legend()
        ax.set_title("Live Evaluation")
        plt.show(block=False)

        update_freq = 10  # redraw every 10 steps
        bh_units = initial_equity / prices.iloc[0]["Open"]
        x_data, y_eq, y_bh = [], [], []

        def push_frame(equity):
            curr_price = env.prices.iloc[env._day_offsets[env._start] + env._time]["Open"]
            x_data.append(len(x_data))
            y_eq.append(equity)
            y_bh.append(bh_units * curr_price)
            if len(x_data) % update_freq:
                return
            line_eq.set_data(x_data, y_eq)
            line_bh.set_data(x_data, y_bh)
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(min(min(y_eq), min(y_bh)) * 0.95, max(max(y_eq), max(y_bh)) * 1.05)
            ax.set_xlim(0, len(x_data) * 1.05)
            plt.draw()
            plt.pause(0.001)

    # Build a plain, single env (no randomness/windowing for backtest)
    env = SingleTickerEnv(
        prices=prices,
        features=features,
        initial_equity=initial_equity,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        max_position_pct=max_position_pct,
        reward_mode=reward_mode,
        step_size=step_size,
    )

    # If you trained with VecNormalize, load stats and wrap for eval
    if vecnorm_path:
        # === VecEnv path (4-return API) ===
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

        # Set on the env directly since VecEnv won't pass it through
        env.force_full_reset = True
        obs = venv.reset()
        lstm_states = None
        # LSTM needs episode starts
        episode_starts = np.ones((venv.num_envs,), dtype=bool)
        eq_hist, ts_index = [], []

        while True:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
            obs, rewards, dones, infos = venv.step(action)               # <-- 4 items
            episode_starts = dones
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            if dones[0]:   # VecEnv has a single "done" flag; truncation info may be in infos
                break
            eq_hist.append(info0.get("equity", np.nan))
            ts_index.append(env._days[env._start].strftime("%d-%m-%Y") + " " + env._time_levels[env._time].strftime("%H:%M:%S"))

            if push_frame:
                push_frame(eq_hist[-1])

        return pd.Series(eq_hist, index=pd.Index(ts_index, name=prices.index.name))

    else:
        # === Raw env path (5-return Gymnasium API) ===
        env.force_full_reset = True
        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        eq_hist, ts_index = [], []

        done = False
        truncated = False
        while not (done or truncated):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)  # scalar action ok
            obs, reward, done, truncated, info = env.step(action)       # <-- 5 items
            episode_starts = np.array([done or truncated])
            if(done or truncated):
                break
            eq_hist.append(info.get("equity", np.nan))
            ts_index.append(env._days[env._start].strftime("%d-%m-%Y") + " " + env._time_levels[env._time].strftime("%H:%M:%S"))

            if push_frame:
                push_frame(eq_hist[-1])

        return pd.Series(eq_hist, index=pd.Index(ts_index, name=prices.index.name))
