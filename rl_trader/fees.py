from dataclasses import dataclass

@dataclass
class IBKRFeeModel:
    model: str = "fixed"  # 'fixed' or 'tiered'
    per_share: float = 0.005
    min_per_order: float = 1.0
    tiered_per_share: float = 0.0035
    tiered_min_order: float = 0.35
    sec_fee_per_dollar: float = 0.0  # set if you want to approximate
    taf_fee_per_share: float = 0.0   # set if you want to approximate

    def commission(self, shares: float, price: float) -> float:
        shares = abs(shares)
        if shares <= 0:
            return 0.0
        if self.model == "fixed":
            fees = shares * self.per_share
            return max(fees, self.min_per_order)
        # tiered approx (ignores exchange rebates/fees detail)
        fees = max(shares * self.tiered_per_share, self.tiered_min_order)
        fees += (price * shares) * self.sec_fee_per_dollar
        fees += shares * self.taf_fee_per_share
        return fees
