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
                 spread_std_bps: float=0.5, slippage_std_bps: float=0.3, price_jitter_bps:float=0, do_nothing_penalty: float=0.0, double_action_penalty:float=0.0):
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
        self._i = 0
        self._start = 0
        self._end = 0
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity

        # RNG holder
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # make per-env RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # choose random episode window
        total_len = len(self.prices)
        ep_len = int(self.np_random.integers(self.min_episode_len, self.max_episode_len + 1))
        if ep_len >= total_len:
            ep_len = total_len - 1
        self._start = int(self.np_random.integers(0, total_len - ep_len))
        self._end = self._start + ep_len

        self._i = self._start
        self.cash = self.initial_equity
        self.shares = 0.0
        self.prev_equity = self.initial_equity

        obs = self._obs()
        info = {"seed": seed, "start": self._start, "end": self._end}
        return obs, info
        
    def _price_at(self, idx: int) -> float:
        mid = float(self.prices.iloc[idx])
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
        x = self.features.iloc[self._i].to_numpy(dtype=np.float32)
        has_position = True if self.shares > 0 else False
        return np.concatenate([x, [has_position]]).astype(np.float32)

    def step(self, action):
        # Current state
        mid = float(self.prices.iloc[self._i])
        bid, ask = self._best_bid_ask(mid)
        reward=0

        # If the day ended then we sell
        if((self.features.iloc[self._i].iloc[-1] < 0.507538) and (self.features.iloc[self._i].iloc[-1] > 0.496217)):
            action = 2

        # Get action: 0=buy, 1=hold, 2=sell
        match action:
            case 0:
                if(self.shares == 0):
                    target_shares = math.floor(self.cash/mid)
                else:
                    target_shares = self.shares
                    reward = self.do_nothing_penalty
            case 1:
                target_shares = self.shares
                mid_next = float(self.prices.iloc[self._i])
                equity_after = self.cash + self.shares * self._price_at(self._i)
            
                reward = step_reward(self.prev_equity, equity_after, self.reward_mode, self.reward_scale) + self.do_nothing_penalty
            case 2:
                if(self.shares != 0):
                    target_shares = 0
                else:
                    target_shares = self.shares
                    reward = self.do_nothing_penalty

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

            mid_now = self._price_at(self._i)
            equity_after = self.cash + self.shares * mid_now
            
            reward = step_reward(self.prev_equity, equity_after, self.reward_mode, self.reward_scale)
        else: equity_after = equity_before

        done = False
        info = {"equity":equity_after, "action": action}

        # Advance time
        self._i += 1
        if self._i >= self._end:
            done = True

        self.prev_equity = equity_after
        obs = self._obs()
        return obs, float(reward), done, False, info
