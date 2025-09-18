from rl_trader.fees import IBKRFeeModel

def test_fixed_fee_min():
    f = IBKRFeeModel(model="fixed", per_share=0.005, min_per_order=1.0)
    assert abs(f.commission(10, 100.0) - 1.0) < 1e-9

def test_tiered_fee_min():
    f = IBKRFeeModel(model="tiered", tiered_per_share=0.0035, tiered_min_order=0.35)
    assert abs(f.commission(10, 100.0) - 0.35) < 1e-9
