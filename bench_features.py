import time
import pandas as pd
import numpy as np
from rl_trader.features import build_feature_matrix

def generate_dummy_data(n=10000):
    dates = pd.date_range(start="2020-01-01", periods=n, freq="1min")
    df = pd.DataFrame(index=dates)
    df["Open"] = np.random.uniform(100, 200, n)
    df["High"] = df["Open"] * np.random.uniform(1.0, 1.01, n)
    df["Low"] = df["Open"] * np.random.uniform(0.99, 1.0, n)
    df["Close"] = np.random.uniform(df["Low"], df["High"], n)
    df["Volume"] = np.random.uniform(1000, 10000, n)
    return df

def main():
    print("Generating dummy data (N=10000)...")
    df = generate_dummy_data(10000)

    print("Running build_feature_matrix...")
    start = time.time()
    X = build_feature_matrix(df)
    end = time.time()

    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Output shape: {X.shape}")

if __name__ == "__main__":
    main()
