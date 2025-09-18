import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .fees import IBKRFeeModel
from .reward import step_reward

class SingleTickerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.Series, features: pd.DataFrame, initial_equity: float = 10_000.0,
                 spread_bps: float = 2.0, slippage_bps: float = 0.0, max_position_pct: float = 1.0,
                 reward_mode: str = "pnl_raw", reward_scale: float | None = None,
                 fee_kwargs: dict | None = None):
        super().__init__()
        assert prices.index.equals(features.index), "Prices and features must be aligned index"
        self.prices = prices.astype(float)  # Close as midprice
        self.features = features.astype(float)
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.initial_equity = float(initial_equity)
        self.max_position_pct = float(max_position_pct)
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self.fees = IBKRFeeModel(**(fee_kwargs or {}))
        # Observation: feature vector + current position pct and cash pct
        self.obs_columns = list(self.features.columns)
        self._obs_dim = len(self.obs_columns) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        # Action: target allocation [0..1] (long-only). Could extend to shorting by allowing [-1,1].
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._i = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity
        return self._obs(), {}

    def _best_bid_ask(self, mid: float):
        spread = mid * (self.spread_bps / 10_000.0)
        bid = mid - 0.5 * spread
        ask = mid + 0.5 * spread
        # simple slippage: widen prices by slippage_bps toward worse fill
        slip = mid * (self.slippage_bps / 10_000.0)
        return bid - slip, ask + slip

    def _obs(self):
        x = self.features.iloc[self._i].to_numpy(dtype=np.float32)
        price = float(self.prices.iloc[self._i])
        equity = self.cash + self.shares * price
        pos_pct = 0.0 if equity <= 0 else (self.shares * price) / equity
        cash_pct = 0.0 if equity <= 0 else self.cash / equity
        return np.concatenate([x, [pos_pct, cash_pct]]).astype(np.float32)

    def step(self, action):
        # Clip action to [0, 1]
        target_pct = float(np.clip(action[0], 0.0, 1.0))
        # Current state
        mid = float(self.prices.iloc[self._i])
        bid, ask = self._best_bid_ask(mid)

        equity_before = self.cash + self.shares * mid
        target_dollar = target_pct * equity_before
        current_dollar = self.shares * mid
        delta_dollar = target_dollar - current_dollar

        # Execute trade to move towards target
        done = False
        info = {}
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

        # Advance time
        self._i += 1
        if self._i >= len(self.prices) - 1:
            done = True

        mid_next = float(self.prices.iloc[self._i])
        equity_after = self.cash + self.shares * mid_next
        reward = step_reward(self.prev_equity, equity_after, self.reward_mode, self.reward_scale)
        self.prev_equity = equity_after
        obs = self._obs()
        return obs, float(reward), done, False, {"equity": equity_after, **info}
