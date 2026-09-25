import math

import numpy as np
import pytest

from rl_trader.env import SingleTickerEnv
from tests.conftest import NO_FEES, make_feats, make_prices


def build_env(prices, feats=None, **kwargs):
    """Env with no randomness that runs the whole dataset."""
    params = dict(
        initial_equity=10_000.0,
        spread_bps=0.0,
        spread_std_bps=0.0,
        slippage_std_bps=0.0,
        price_jitter_bps=0.0,
        reward_mode="pnl_raw",
        min_episode_len=len(prices),
        max_episode_len=len(prices),
    )
    params.update(kwargs)
    env = SingleTickerEnv(prices, make_feats(prices) if feats is None else feats, **params)
    env.force_full_reset = True
    return env


def roll(env, actions, seed=0):
    """Step through actions until they run out or the episode ends.

    Rewards only add up to PnL once the episode finishes, since the last
    step's costs get charged at the end.
    """
    obs, _ = env.reset(seed=seed)
    observations, rewards, info = [obs], [], {}
    for action in actions:
        obs, reward, done, truncated, info = env.step([action])
        observations.append(obs)
        rewards.append(reward)
        if done or truncated:
            break
    return observations, rewards, info


def test_env_accounting_basic():
    prices = make_prices(np.linspace(100, 110, 100))
    env = build_env(prices, initial_equity=1000.0, max_position_pct=1.0)
    _, _, info = roll(env, [1.0] * 12)
    assert env.shares > 0 and env.cash >= 0
    assert info["equity"] > 1000.0  # price rose while fully invested


def test_position_sizing_respects_max_position_pct():
    prices = make_prices(np.full(10, 100.0))
    env = build_env(prices, max_position_pct=0.5, fee_kwargs=NO_FEES)
    roll(env, [1.0, 1.0])
    assert env.shares == math.floor((0.5 * 10_000.0) / 100.0)
    assert 4900 < env.cash < 5100


@pytest.mark.parametrize("max_pct", [0.25, 0.5, 1.0])
def test_exposure_never_exceeds_limit(max_pct):
    prices = make_prices(100 + 10 * np.sin(np.linspace(0, 6, 80)))
    env = build_env(prices, max_position_pct=max_pct, fee_kwargs=NO_FEES)
    rng = np.random.default_rng(0)

    obs, _ = env.reset(seed=0)
    for _ in range(70):
        obs, reward, done, truncated, info = env.step([rng.uniform(0, 1)])
        equity = env.cash + env.shares * env._get_open_now()
        exposure = (env.shares * env._get_open_now()) / equity
        # Price can drift a bit past the target between bars
        assert exposure <= max_pct + 0.05
        assert env.cash >= -1e-6
        if done or truncated:
            break


def test_cash_is_never_driven_negative_by_fees():
    # Min commission can push an order just over the cash available
    prices = make_prices(np.full(40, 97.3))
    env = build_env(
        prices,
        initial_equity=1000.0,
        max_position_pct=1.0,
        fee_kwargs={"model": "fixed", "per_share": 0.01, "min_per_order": 5.0},
    )
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, reward, done, truncated, info = env.step([1.0])
        assert env.cash >= -1e-9, f"cash went negative: {env.cash}"
        if done or truncated:
            break


def test_churn_on_flat_prices_is_punished(flat_market):
    """Flipping in and out on flat prices has to cost reward."""
    prices, feats = flat_market
    env = build_env(prices, feats, max_position_pct=1.0, spread_bps=2.0)
    # Run to the end so the last step's costs get charged
    actions = [1.0 if i % 2 == 0 else 0.0 for i in range(len(prices) + 5)]
    _, rewards, info = roll(env, actions)

    realized = info["equity"] - 10_000.0
    assert realized < 0, "trading flat prices should lose money"
    assert sum(rewards) < 0, "churning should give a negative reward"
    assert sum(rewards) == pytest.approx(realized, abs=1e-6)


def test_reward_stream_equals_realized_pnl():
    """Rewards should add up to the change in equity."""
    prices = make_prices(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, 70)))
    env = build_env(prices, max_position_pct=1.0, spread_bps=3.0)
    rng = np.random.default_rng(2)
    actions = [rng.uniform(0, 1) for _ in range(len(prices) + 5)]
    _, rewards, info = roll(env, actions)

    assert sum(rewards) == pytest.approx(info["equity"] - 10_000.0, abs=1e-6)


def test_holding_cash_earns_nothing():
    prices = make_prices(np.linspace(100, 130, 40))
    env = build_env(prices, max_position_pct=1.0)
    _, rewards, info = roll(env, [0.0] * 35)

    assert env.shares == 0
    assert info["equity"] == pytest.approx(10_000.0)
    assert sum(rewards) == pytest.approx(0.0, abs=1e-9)


def test_observations_ignore_future_prices():
    """Changing future prices shouldn't change any earlier observation."""
    base = np.linspace(100, 120, 60)
    split = 30

    poisoned = base.copy()
    poisoned[split:] = 10_000.0

    actions = [0.6] * (split - 2)
    obs_a, _, _ = roll(build_env(make_prices(base), max_position_pct=1.0), actions)
    obs_b, _, _ = roll(build_env(make_prices(poisoned), max_position_pct=1.0), actions)

    for t, (a, b) in enumerate(zip(obs_a, obs_b)):
        np.testing.assert_allclose(a, b, err_msg=f"obs at t={t} changed with future prices")


def test_features_are_lagged_by_one_bar():
    """Agent at t should see the feature row from t-1."""
    prices = make_prices(np.full(20, 100.0))
    feats = make_feats(prices)
    feats["x0"] = np.arange(len(prices), dtype=float)

    env = build_env(prices, feats, max_position_pct=1.0)
    obs, _ = env.reset(seed=0)
    assert obs[0] == 0.0  # t=0 has no prior row, filled with 0

    obs, *_ = env.step([0.0])
    assert obs[0] == 0.0  # at t=1 the agent sees raw row 0


def test_first_bar_gap_is_zero():
    """First bar has no previous close, so the gap should be 0."""
    prices = make_prices(np.concatenate([[100.0], np.full(29, 500.0)]))
    env = build_env(prices, max_position_pct=1.0)
    obs, _ = env.reset(seed=0)

    gap = obs[-3]  # obs tail is [gap, position_pct, time_of_day]
    assert gap == pytest.approx(0.0), f"gap on first bar was {gap}"


def test_action_space_is_long_only_and_fully_used():
    prices = make_prices(np.full(30, 100.0))
    env = build_env(prices, max_position_pct=1.0, fee_kwargs=NO_FEES)

    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0

    # Every action should give a different position
    exposures = []
    for action in (0.0, 0.25, 0.5, 0.75, 1.0):
        env = build_env(prices, max_position_pct=1.0, fee_kwargs=NO_FEES)
        roll(env, [action, action])
        exposures.append(env.shares)
    assert exposures == sorted(exposures)
    assert len(set(exposures)) == len(exposures), f"actions gave the same position: {exposures}"


def test_out_of_range_actions_are_clipped():
    prices = make_prices(np.full(20, 100.0))
    for action in (-5.0, 5.0):
        env = build_env(prices, max_position_pct=1.0, fee_kwargs=NO_FEES)
        roll(env, [action, action])
        assert 0 <= env.shares <= 100
        assert env.cash >= -1e-9
