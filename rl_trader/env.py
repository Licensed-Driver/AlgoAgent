import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .fees import IBKRFeeModel
from .reward import step_reward, DifferentialSharpe
import math

class SingleTickerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, initial_equity: float = 10_000.0,
                 spread_bps: float = 2.0, slippage_bps: float = 0.0, max_position_pct: float = 1.0,
                 reward_mode: str = "pnl_raw", reward_scale: float | None = None,
                 fee_kwargs: dict | None = None, min_episode_len: int=512, max_episode_len: int=2048,
                 spread_std_bps: float=0.5, slippage_std_bps: float=0.3, price_jitter_bps:float=0, step_size: int=1):
        super().__init__()
        if prices.index.nlevels == 1:
            dt_idx = prices.index
            multi_idx = pd.MultiIndex.from_arrays(
                [dt_idx.date, dt_idx.time],
                names=["date", "time"],
            )
            prices = prices.copy()
            prices.index = multi_idx
            features = features.copy()
            features.index = multi_idx
        assert prices.index.equals(features.index), "Prices and features must be aligned index"

        self.prices = prices.astype(np.float32)

        # Shift features a bar since we trade at the open and only know the last close
        self.features = features.shift(1).fillna(0.0).astype(np.float32)

        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.initial_equity = float(initial_equity)
        self.max_position_pct = float(max_position_pct)
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.fees = IBKRFeeModel(**(fee_kwargs or {}))

        self.obs_columns = list(self.features.columns)
        # Features + gap + position pct + time of day
        self._obs_dim = len(self.obs_columns) + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        # Target long exposure as a fraction of max_position_pct. Long only since
        # borrow costs aren't modelled
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.min_episode_len = int(min_episode_len)
        self.max_episode_len = int(max_episode_len)
        self.spread_std_bps = float(spread_std_bps)
        self.slippage_std_bps = float(slippage_std_bps)
        self.price_jitter_bps = float(price_jitter_bps)
        self.step_size = int(step_size)

        # Differential Sharpe state
        eta = 0.01 if self.reward_mode == "differential_sharpe" else 1.0 / max_episode_len
        self.dsr = DifferentialSharpe(eta=eta)

        self.np_random = None

        self._prices_np = self.prices.to_numpy(dtype=np.float32)
        self._features_np = self.features.to_numpy(dtype=np.float32)
        day_counts = self.prices.groupby(level="date").size().to_numpy(dtype=np.int32)
        self._day_lengths = day_counts
        self._day_offsets = np.zeros_like(day_counts)
        if len(day_counts) > 1:
            np.cumsum(day_counts[:-1], out=self._day_offsets[1:])
        self._days = self.prices.index.get_level_values("date").unique()
        self._time_levels = self.prices.index.levels[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        total_len = len(self._days)

        full_reset = (options and options.get("full_reset")) or getattr(self, "force_full_reset", False)

        if full_reset:
            # deterministically run the whole dataset
            ep_len = total_len
            self._start = 0
            self._time = 0
        else:
            ep_len = int(self.np_random.integers(self.min_episode_len, self.max_episode_len + 1))
            if ep_len >= total_len: ep_len = total_len - 1
            ep_len = max(ep_len, 1)

            # Start at 1 because index 0 has no "prev close" and features are shifted
            max_start = max(1, total_len - ep_len)
            self._start = int(self.np_random.integers(1, max_start + 1))

        self._episode_start = self._start
        self._end = min(self._start + ep_len - 1, total_len - 1)

        if not full_reset:
             self._time = self.np_random.integers(0, self.step_size)
        else:
             self._time = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        # Reward is open to open
        self._prev_equity_at_open = self.initial_equity
        if self.reward_mode == "differential_sharpe":
             self.dsr.reset()

        obs = self._obs()
        info = {
            "seed": seed,
            "start": self._days[self._episode_start],
            "end": self._days[self._end],
        }
        return obs, info

    def _get_close_now(self) -> float:
        mid = float(self._prices_np[self._day_offsets[self._start] + self._time][0])
        if self.price_jitter_bps > 0:
            mid *= 1.0 + 1e-4 * self.price_jitter_bps * self.np_random.normal()
        return mid

    def _get_open_now(self) -> float:
        mid = float(self._prices_np[self._day_offsets[self._start] + self._time][1])
        if self.price_jitter_bps > 0:
            mid *= 1.0 + 1e-4 * self.price_jitter_bps * self.np_random.normal()
        return mid

    def _get_prev_close(self) -> float:
        # Close of the last step, might be on the previous day
        prev_time = self._time - self.step_size
        if prev_time >= 0:
            idx = self._day_offsets[self._start] + prev_time
        elif self._start > 0:
            prev_day = self._start - 1
            idx = self._day_offsets[prev_day] + self._day_lengths[prev_day] - 1
        else:
            # No previous close on the very first bar, so no gap
            return self._get_open_now()
        return float(self._prices_np[idx][0])

    def _max_affordable(self, ask: float) -> int:
        # Most shares we can buy with the cash we have after commission
        lo, hi = 0, int(self.cash // ask) if ask > 0 else 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if mid * ask + self.fees.commission(mid, ask) <= self.cash:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _reward_from(self, prev_equity: float, new_equity: float) -> float:
        if self.reward_mode == "differential_sharpe":
            ret = (new_equity - prev_equity) / max(prev_equity, 1e-9)
            reward = self.dsr.step(ret)
            return reward * float(self.reward_scale) if self.reward_scale else reward
        return step_reward(prev_equity, new_equity, self.reward_mode, self.reward_scale)

    def _best_bid_ask(self, mid: float):
        spread_bps = max(0.0, self.spread_bps + self.spread_std_bps * self.np_random.normal())
        slip_bps = max(0.0, self.slippage_bps + self.slippage_std_bps * self.np_random.normal())

        # bps to decimal
        half_spread = mid * (spread_bps * 0.5 / 10000.0)
        slip = mid * (slip_bps / 10000.0)

        bid = mid - half_spread - slip
        ask = mid + half_spread + slip
        return bid, ask

    def _obs(self):
        idx = self._day_offsets[self._start] + self._time

        # Already lagged a bar in __init__
        x = self._features_np[idx]

        open_now = self._get_open_now()
        prev_close = self._get_prev_close()

        # Gap from the last close, since the features are a bar behind
        gap = (open_now / prev_close) - 1.0

        total_equity = self.cash + self.shares * open_now
        current_pos_pct = (self.shares * open_now) / total_equity if total_equity > 1e-9 else 0.0
        time_of_day = self._time / self._day_lengths[self._start]

        # Raw equity left out since it's non-stationary
        obs = np.concatenate((x, np.array([gap, current_pos_pct, time_of_day], dtype=np.float32)))
        return obs

    def step(self, action):
        done = False

        # We trade at the Open
        price_now = self._get_open_now()
        bid, ask = self._best_bid_ask(price_now)

        equity_before_action = self.cash + self.shares * price_now

        # Reward for the position held from the last open to this one, including its costs
        reward = self._reward_from(self._prev_equity_at_open, equity_before_action)

        if isinstance(action, (list, tuple, np.ndarray)):
            action_scalar = float(action[0])
        else:
            action_scalar = float(action)

        target_pct = float(np.clip(action_scalar, 0.0, 1.0))
        target_equity_in_stock = target_pct * equity_before_action * self.max_position_pct
        target_shares = math.floor(max(0.0, target_equity_in_stock) / price_now)

        delta_shares = target_shares - self.shares

        if delta_shares != 0:
            if delta_shares > 0: # BUY
                shares_to_buy = min(delta_shares, self._max_affordable(ask))
                if shares_to_buy > 0:
                    self.cash -= shares_to_buy * ask + self.fees.commission(shares_to_buy, ask)
                    self.shares += shares_to_buy
            else: # SELL
                shares_to_sell = abs(delta_shares)
                proceeds = shares_to_sell * bid
                comm = self.fees.commission(shares_to_sell, bid)
                self.cash += (proceeds - comm)
                self.shares -= shares_to_sell

        equity_after_action = self.cash + self.shares * price_now

        # Use pre-trade equity so this step's costs show up in the next reward
        self._prev_equity_at_open = equity_before_action

        current_day_len = self._day_lengths[self._start]
        self._time += self.step_size

        # End of day, move to the next one at a random offset
        if self._time >= current_day_len:
            self._time = self.np_random.integers(0, self.step_size)
            self._start += 1
            if self._start > self._end:
                done = True
                self._start = self._end  # clamp so the final _obs() stays in range

        if done:
            # Last step, so charge its costs now
            reward += self._reward_from(equity_before_action, equity_after_action)

        obs = self._obs()
        return obs, float(reward), done, False, {"equity": equity_after_action}
