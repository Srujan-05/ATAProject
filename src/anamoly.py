import pandas as pd
import numpy as np
import os

def compute_zscore_residuals(df, window=336, min_periods=168):
    df = df.copy()
    df['resid'] = df['y_true'] - df['yhat']

    roll_mean = df['resid'].rolling(window=window, min_periods=min_periods).mean()
    roll_std  = df['resid'].rolling(window=window, min_periods=min_periods).std(ddof=0)

    df['z_resid'] = (df['resid'] - roll_mean) / (roll_std + 1e-8)
    return df


def compute_flag_z(df):
    df['flag_z'] = (df['z_resid'].abs() >= 3.0).astype(int)
    return df


def compute_cusum(df, k=0.5, h=5.0):
    z = df['z_resid'].fillna(0).values

    s_pos = np.zeros_like(z)
    s_neg = np.zeros_like(z)

    for i in range(1, len(z)):
        s_pos[i] = max(0, s_pos[i-1] + z[i] - k)
        s_neg[i] = max(0, s_neg[i-1] - z[i] - k)

    df['flag_cusum'] = ((s_pos > h) | (s_neg > h)).astype(int)
    return df


def run_anomaly_part1(forecast_csv_path, output_path):
    df = pd.read_csv(forecast_csv_path)

    # required columns
    for col in ['y_true','yhat','lo','hi','timestamp']:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")

    df = compute_zscore_residuals(df)
    df = compute_flag_z(df)
    df = compute_cusum(df)

    df[['timestamp','y_true','yhat','lo','hi','z_resid','flag_z','flag_cusum']].to_csv(
        output_path, index=False
    )

    print("Saved anomalies:", output_path)


if __name__ == "__main__":
    cc = ["CH", "FR", "AT"]
    for c in cc:
        run_anomaly_part1("../outputs/{}_forecast_test.csv".format(c), "../outputs/{}_anomalies.csv".format(c))