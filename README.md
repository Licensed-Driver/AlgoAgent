# AlgoAgent

A production‑ready template for training a single‑ticker reinforcement learning (RL) trader
entirely via backtesting. It pulls OHLCV data from the Alpaca API, adds many technical
indicators, simulates realistic execution with **bid/ask spread** and **IBKR commission
models**, and trains a **PPO** agent (Stable‑Baselines3) inside a **Gymnasium** environment.

## Highlights

- **Single‑ticker** focus with robust walk‑forward training and evaluation.
- **Execution realism:** configurable bid/ask spread model, slippage, and IBKR‑style fees.
- **Risk-aware training:** position limits so equity is never exceeded; optional reward shapes.
- **Feature pipeline:** dozens of TA indicators, normalization, missing‑value handling.
- **Reproducible experiments:** configs, seeds, logging, and model checkpointing.
- **Tests** for the fee model and environment accounting.
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

   Use a saved dataset and trained model file:

   ```bash
   python -m scripts.evaluate --data artifacts/data_AAPL_2024-01-01_2024-06-01_1Min.parquet \
       --model models/ppo_auto_single_ticker.zip
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
- Rewards support multiple shapes; default is step‑wise PnL change divided by starting equity.



## One-command end-to-end run (pure profit reward)

```bash
python -m scripts.auto_pipeline --symbol AAPL --start 2024-01-01 --end 2024-06-01 --timeframe 1Min   --total_timesteps 200000 --train_days 30 --valid_days 7 --test_days 7 --stride_days 7
```
This will: fetch data → build features → train PPO on the first 90% (reward = **raw profit scaled by initial equity**) → evaluate on the last 10% → run a walk-forward study and save equity curves under `artifacts/`.
