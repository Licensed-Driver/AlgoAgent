import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .fees import IBKRFeeModel
from .reward import step_reward
import math


class SingleTickerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.Series, features: pd.DataFrame, initial_equity: float = 10_000.0,
                 spread_bps: float = 2.0, slippage_bps: float = 0.0, max_position_pct: float = 1.0,
                 reward_mode: str = "pnl_raw", reward_scale: float | None = None,
                 fee_kwargs: dict | None = None, min_episode_len: int=512, max_episode_len: int=2048,
                 spread_std_bps: float=0.5, slippage_std_bps: float=0.3, price_jitter_bps:float=0,
                 do_nothing_penalty: float=0.0, double_action_penalty:float=0.0):
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
        self.prices = prices.astype(np.float32)  # Close as midprice
        self.features = features.astype(np.float32)
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.initial_equity = float(initial_equity)
        self.max_position_pct = float(max_position_pct)
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.fees = IBKRFeeModel(**(fee_kwargs or {}))
        # Observation: feature vector + has a position
        self.obs_columns = list(self.features.columns)
        self._obs_dim = len(self.obs_columns) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        # Action: Buy, hold, or sell one hot vector [_, _, _]
        self.action_space = spaces.Discrete(3)
        self.do_nothing_penalty = do_nothing_penalty
        self.double_action_penalty = double_action_penalty

        # Adding randomness for episode splitting
        self.min_episode_len = int(min_episode_len)
        self.max_episode_len = int(max_episode_len)
        self.spread_std_bps = float(spread_std_bps)
        self.slippage_std_bps = float(slippage_std_bps)
        self.price_jitter_bps = float(price_jitter_bps)

        # State
        self._time = 0
        self._start = 0
        self._episode_start = 0
        self._end = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity
        self._days = self.prices.index.get_level_values("date").unique()

        # Pre-compute day offsets for fast lookup
        day_counts = self.prices.groupby(level="date").size().to_numpy(dtype=np.int32)
        self._day_lengths = day_counts
        self._day_offsets = np.zeros_like(day_counts)
        if len(day_counts) > 1:
            np.cumsum(day_counts[:-1], out=self._day_offsets[1:])
        self._prices_np = self.prices.to_numpy(dtype=np.float32, copy=True)
        self._features_np = self.features.to_numpy(dtype=np.float32, copy=True)
        self._time_levels = self.prices.index.levels[1]

        # RNG holder
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # make per-env RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # choose random amount of days for episode
        total_len = len(self._days)
        ep_len = int(self.np_random.integers(self.min_episode_len, self.max_episode_len + 1))
        if ep_len >= total_len:
            ep_len = total_len - 1
        ep_len = max(ep_len, 1)
        max_start = max(0, total_len - ep_len)
        self._start = int(self.np_random.integers(0, max_start + 1))
        self._episode_start = self._start
        self._end = min(self._start + ep_len - 1, total_len - 1)

        self._time = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity

        obs = self._obs()
        info = {
            "seed": seed,
            "start": self._days[self._episode_start],
            "end": self._days[self._end],
        }
        return obs, info
    
    def _get_day_idx(self, idx: int):
        return self._days[idx]

    def _get_time_idx(self, idx: int):
        return self._time_levels[idx]
    
    def _get_price(self, i, j):
        base = self._day_offsets[i] + j
        return self._prices_np[base]
        
    def _price_at(self) -> float:
        mid = float(self._prices_np[self._day_offsets[self._start] + self._time])
        if self.price_jitter_bps > 0.0:
            mid *= 1.0 + 1e-4 * self.price_jitter_bps * self.np_random.normal()
        return mid

    def _best_bid_ask(self, mid: float):
        # randomize around mean spread/slippage
        spread_bps = self.spread_bps + self.spread_std_bps * self.np_random.normal()
        slip_bps = self.slippage_bps + self.slippage_std_bps * self.np_random.normal()
        spread_bps = max(0.0, spread_bps)
        slip_bps = max(0.0, slip_bps)

        spread = mid * (spread_bps / 10_000.0)
        bid = mid - 0.5 * spread
        ask = mid + 0.5 * spread
        slip = mid * (slip_bps / 10_000.0)
        return bid - slip, ask + slip

    def _obs(self):
        idx = self._day_offsets[self._start] + self._time
        x = self._features_np[idx]
        has_position = 1.0 if self.shares > 1e-8 else 0.0
        obs = np.concatenate((x, np.array([has_position], dtype=np.float32)))
        return obs.astype(np.float32, copy=False)

    def step(self, action):
        if isinstance(action, (list, tuple)):
            action = np.asarray(action)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        # Current state
        idx = self._day_offsets[self._start] + self._time
        mid = float(self._prices_np[idx])
        bid, ask = self._best_bid_ask(mid)
        reward = 0.0
        target_shares = self.shares

        # Get action: 0=buy, 1=hold, 2=sell
        if action == 0:
            if self.shares == 0:
                target_shares = math.floor(self.cash / mid)
            else:
                reward = self.do_nothing_penalty
        elif action == 1:
            equity_after = self.cash + self.shares * self._price_at()
            reward = step_reward(self.prev_equity, equity_after, self.reward_mode, self.reward_scale) + self.do_nothing_penalty
        elif action == 2:
            if self.shares != 0:
                target_shares = 0.0
            else:
                reward = self.do_nothing_penalty
        else:
            raise ValueError(f"Invalid action: {action}")

        target_dollar = target_shares * mid
        current_dollar = self.shares * mid
        delta_dollar = target_dollar - current_dollar
        equity_before = self.cash + self.shares * mid

        # Execute trade to move towards target
        if abs(delta_dollar) > 1e-8:
            if delta_dollar > 0:
                # Buy
                shares_to_buy = delta_dollar / ask
                # never exceed max_position_pct
                max_pos_value = self.max_position_pct * equity_before
                desired_value = min(target_dollar, max_pos_value)
                shares_to_buy = max(0.0, (desired_value - current_dollar) / ask)
                cost = shares_to_buy * ask
                fee = self.fees.commission(shares_to_buy, ask)
                if cost + fee > self.cash:
                    # scale down to available cash
                    shares_to_buy = max(0.0, (self.cash - fee) / ask)
                    cost = shares_to_buy * ask
                    fee = self.fees.commission(shares_to_buy, ask)
                self.cash -= (cost + fee)
                self.shares += shares_to_buy
            else:
                # Sell
                shares_to_sell = (-delta_dollar) / bid
                shares_to_sell = min(shares_to_sell, self.shares)
                proceeds = shares_to_sell * bid
                fee = self.fees.commission(shares_to_sell, bid)
                self.cash += (proceeds - fee)
                self.shares -= shares_to_sell

            mid_now = self._price_at()
            equity_after = self.cash + self.shares * mid_now
            
            reward = step_reward(self.prev_equity, equity_after, self.reward_mode, self.reward_scale)
        else:
            equity_after = equity_before

        done = False
        info = {"equity":equity_after, "action": action}

        # If the day is over, reset _time and advance the day
        current_day_len = self._day_lengths[self._start]
        if self._time < current_day_len - 1:
            self._time += 1
        elif self._start >= self._end:
            done = True
        else:
            self._time = 0
            self._start += 1

        self.prev_equity = equity_after
        obs = self._obs()
        return obs, float(reward), done, False, info
