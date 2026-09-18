import numpy as np

class DifferentialSharpe:
    """
    Online differential Sharpe ratio, from Moody & Saffell (2001).
    A and B are the EWMA first and second moments of returns, decayed by eta.
    D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
    where dA_t = R_t - A_{t-1} and dB_t = R_t^2 - B_{t-1}
    """
    def __init__(self, eta: float = 0.1):
        self.eta = eta
        self.A = 0.0
        self.B = 0.0

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def step(self, ret: float) -> float:
        # Deltas and variance use the previous estimates
        delta_A = ret - self.A
        delta_B = (ret ** 2) - self.B
        var = self.B - self.A**2

        # Not enough variance yet, just update
        if var < 1e-6:
            self.A += self.eta * delta_A
            self.B += self.eta * delta_B
            return 0.0

        k = (self.B * delta_A - 0.5 * self.A * delta_B) / (var ** 1.5)

        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return k

def step_reward(prev_equity: float, new_equity: float, mode: str = "pnl_raw", scale: float | None = None):
    ret = None
    if mode == "pnl":
        # Relative profit and loss vs current equity
        ret = (new_equity - prev_equity) / max(1.0, prev_equity)

    if mode == "pnl_raw":
        ret = new_equity - prev_equity

    if mode == "logpnl":
        ret = np.log(max(new_equity, 1e-9)) - np.log(max(prev_equity, 1e-9))

    if mode == "sharpe_step":
        # Proxy: penalize downside more than upside
        ret = (new_equity - prev_equity) / max(prev_equity, 1.0)
        ret = (ret - 0.5 * max(0.0, -ret))

    # Env uses DifferentialSharpe for this, this is just a stateless fallback
    if mode == "differential_sharpe":
        ret = (np.log(max(new_equity, 1e-9)) - np.log(max(prev_equity, 1e-9))) * 100.0

    return ret / float(scale) if scale and ret else ret
