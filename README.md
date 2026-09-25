# AlgoAgent

A production‑ready template for training a single‑ticker reinforcement learning (RL) trader
entirely via backtesting. It pulls OHLCV data from the Alpaca API, adds many technical
indicators, simulates realistic execution with **bid/ask spread** and **IBKR commission
models**, and trains a **RecurrentPPO** agent (sb3-contrib) inside a **Gymnasium** environment.

The agent sees features up to the close of bar `t-1` and trades at the open of bar `t`.
Its action is how much of the allowed position to hold, from 0 to 1, and its reward is the
open-to-open PnL of that position after spread and commission.

## Highlights

- **Single‑ticker** focus with robust walk‑forward training and evaluation.
- **Execution realism:** configurable bid/ask spread model, slippage, and IBKR‑style fees.
- **Risk-aware training:** position limits so equity is never exceeded; optional reward shapes.
- **Feature pipeline:** dozens of TA indicators, normalization, missing‑value handling.
- **Reproducible experiments:** configs, seeds, logging, and model checkpointing.
- **Tests** for the fee model, environment accounting, causality, and feature correctness.
- **Modern Python packaging** via `pyproject.toml`.

## Quickstart

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. **Set credentials** (copy `.env.example` to `.env` and fill values):

   ```bash
   cp .env.example .env
   # Edit .env to include ALPACA_API_KEY, ALPACA_API_SECRET (and set PAPER or LIVE base URL)
   ```

3. **Fetch data** (e.g., 1‑minute bars for AAPL):

   ```bash
   python -m scripts.fetch_data --symbol AAPL --start 2024-01-01 --end 2024-06-01 --timeframe 1Min
   ```

4. **Train PPO** (single split):

   ```bash
   python -m scripts.train --symbol AAPL --start 2024-01-01 --end 2024-06-01 --timeframe 1Min      --total_timesteps 200_000
   ```

5. **Walk‑forward** (rolling windows):

   ```bash
   python -m scripts.walk_train --symbol AAPL --start 2024-01-01 --end 2024-06-01 --timeframe 1Min      --train_days 30 --valid_days 7 --test_days 7 --stride_days 7 --total_timesteps 150_000
   ```

6. **Evaluate & plot**:

   Pass the symbol and date range. Data comes from the monthly cache, and saved feature
   stats and VecNormalize are loaded if they exist.

   ```bash
   python -m scripts.evaluate --model_path models/ppo_auto_single_ticker.zip \
       --symbol AAPL --start 2024-01-01 --end 2024-06-01
   ```

   Add `--live` to plot equity against buy-and-hold as it runs.

7. **Run the tests**:

   ```bash
   pytest
   ```

## Fee Model (IBKR‑style)

This project ships with a configurable **fee model**. For US equities, you can choose:

- **fixed**: \$0.005/share, \$1.00 minimum per order
- **tiered** (approx): \$0.0035/share, \$0.35 minimum per order (exchange and regulatory fees
  approximated and configurable)

Regulatory/exchange fees vary; keep them configurable and verify your own schedule if you
need precise replication for production.

## Important Notes

- This codebase **does not place live trades**. It is for backtesting / research.
- The Alpaca data function retrieves historical bars via REST. Provide your keys and a base URL.
- Spread and slippage are **simulated** and configurable.
- Rewards support multiple shapes; the default is the log return of equity open-to-open.
- The environment is **long-only**. Borrow costs and margin aren't modelled, so shorting is
  left out for now.

## What the tests check

- **No lookahead.** Changing prices after time `t` doesn't change any observation up to `t`
  (`test_observations_ignore_future_prices`), and cutting off the end of the data doesn't
  change earlier features (`test_features_do_not_use_future_bars`).
- **Costs hit the reward.** Over a full episode the rewards add up to the change in equity
  (`test_reward_stream_equals_realized_pnl`), and trading in and out on flat prices loses
  reward (`test_churn_on_flat_prices_is_punished`).
- **Parallel features match.** The threaded feature builder gives the same output as the
  sequential one (`test_parallel_matches_sequential`).
- **Limits hold.** Position size stays under `max_position_pct` and cash never goes
  negative, even with a minimum commission per order.
- **Every action counts.** Each action value gives a different position size.



## One-command end-to-end run (pure profit reward)

```bash
python -m scripts.auto_pipeline --symbol AAPL --start 2023-01-01 --end 2025-01-01
```
This will: fetch data → build features → train RecurrentPPO on the first 90% → evaluate on the last 10% → run a walk-forward study and save equity curves under `artifacts/`.
WARNING: This command will pull and cache 2 years of 1Min resolution ticker data from the AAPL stock, which may take up a non-negligible amount of space. Just please be aware of that.
