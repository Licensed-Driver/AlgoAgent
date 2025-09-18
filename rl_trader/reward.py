import numpy as np

def step_reward(prev_equity: float, new_equity: float, mode: str = "pnl_raw", scale: float | None = None):
    pnl = new_equity - prev_equity
    if mode == "pnl":
        # Relative profit and loss vs current equity
        return pnl / max(1.0, prev_equity)
    if mode == "pnl_raw":
        # Raw dollar PnL (optionally scaled by a constant if provided)
        return pnl if not scale else pnl / float(scale)
    if mode == "logpnl":
        return np.log(max(new_equity, 1e-9)) - np.log(max(prev_equity, 1e-9))
    if mode == "sharpe_step":
        # Proxy: penalize downside more than upside
        r = (new_equity - prev_equity) / max(prev_equity, 1.0)
        return r - 0.5 * max(0.0, -r)
    # Default to relative pnl
    return pnl / max(1.0, prev_equity)
