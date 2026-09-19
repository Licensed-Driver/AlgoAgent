import pandas as pd
import time
from rl_trader.data import load_or_fetch_monthly
from rl_trader.features import build_feature_matrix, build_features_parallel

def main():
    print("Loading data...")
    # Fetch 3 years of 1-min data as a stress test
    df_raw = load_or_fetch_monthly("NVDA", "2021-01-01", "2024-01-01", "1Min")
    print(f"Data loaded: {len(df_raw)} rows.")

    print("\nRunning SEQUENTIAL...")
    s1 = time.time()
    df_seq = build_feature_matrix(df_raw.copy())
    t_seq = time.time() - s1
    print(f"Sequential built {len(df_seq)} rows in {t_seq:.2f} seconds.")

    print("\nRunning PARALLEL...")
    s2 = time.time()
    df_par = build_features_parallel(df_raw.copy(), num_workers=4, chunk_size=50000)
    t_par = time.time() - s2
    print(f"Parallel built {len(df_par)} rows in {t_par:.2f} seconds.")

    print(f"\nSpeedup: {t_seq / t_par:.2f}x")

    print("\nVerifying outputs...")
    try:
        pd.testing.assert_frame_equal(df_seq, df_par, check_dtype=True)
        print("SUCCESS! Output DataFrames are identical.")
    except Exception as e:
        print("FAIL! Output DataFrames are different.")
        print(e)


if __name__ == "__main__":
    main()
