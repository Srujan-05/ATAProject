import numpy as np
import pandas as pd
import statsmodels.api as models
from load_opsd import load_data
import pmdarima as pm
from scipy.stats import norm

p = 1
q = 1
P = 1
Q = 2
d = 1
D = 1
m = 24
PI = 0.8
H = 24   # forecast horizon

country_codes = ["CH", "FR", "AT"]

def compute_conf(conf_int):
    lo = conf_int[:, 0]
    hi = conf_int[:, 1]
    return lo, hi

for cc in country_codes:
    print("\n===============================")
    print("Processing country:", cc)
    print("===============================")

    raw = load_data(cc, "../data")
    data = raw["load_actual_entsoe_transparency"].dropna().values.astype(float)
    timestamps = raw["timestamp"].values

    n = len(data)
    train_split = int(n * 0.8)
    dev_split = train_split + int(n * 0.1)
    test_split = n

    train_data = data[:train_split]
    dev_data = data[train_split:dev_split]
    test_data = data[dev_split:test_split]

    # ---------------------------
    # DEV BACKTESTING
    # ---------------------------
    print("Starting Dev Backtesting...")

    if cc != "CH":
        model = pm.arima.ARIMA(
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            maxiter=20,
            method='lbfgs',
            enforce_stationarity=False,
            enforce_invertibility=False,
            suppress_warnings=True
        )
        model.fit(train_data)

        yhat = []
        lo_list = []
        hi_list = []
        train_end_list = []

        for i in range(0, len(dev_data) - H, H):
            print("Backtesting {} Started...".format(i // 24 + 1))
            forecast, conf_int = model.predict(H, return_conf_int=True, alpha=1 - PI)
            lo, hi = compute_conf(conf_int)

            yhat.extend(forecast)
            lo_list.extend(lo)
            hi_list.extend(hi)

            # timestamp of last point used for training
            train_end_list.extend([timestamps[train_split + i + (H - 1)]] * H)

            model.update(dev_data[i:i+H])

        # leftover dev block
        if len(yhat) < len(dev_data):
            remaining = len(dev_data) - len(yhat)
            forecast, conf_int = model.predict(remaining, return_conf_int=True, alpha=1 - PI)
            lo, hi = compute_conf(conf_int)

            yhat.extend(forecast)
            lo_list.extend(lo)
            hi_list.extend(hi)
            train_end_list.extend([timestamps[dev_split - 1]] * remaining)

        # Save DEV CSV (no prefix; names fixed)
        dev_df = pd.DataFrame({
            "timestamp": timestamps[train_split:dev_split],
            "yhat": yhat,
            "y_true": dev_data,
            "lo": lo_list,
            "hi": hi_list,
            "horizon": [H] * len(dev_data),
            "train_end": train_end_list
        })

        dev_df.to_csv(f"../outputs/{cc}_forecast_dev.csv", index=False)
        print(f"Saved: ../outputs/{cc}_forecast_dev.csv")

    # ---------------------------
    # TEST BACKTESTING
    # ---------------------------
    print("Starting Test Backtesting...")

    model = pm.arima.ARIMA(
        order=(p, d, q),
        seasonal_order=(P, D, Q, m),
        maxiter=20,
        method='lbfgs',
        enforce_stationarity=False,
        enforce_invertibility=False,
        suppress_warnings=True
    )
    model.fit(np.hstack((train_data, dev_data)))

    yhat = []
    lo_list = []
    hi_list = []
    train_end_list = []

    for i in range(0, len(test_data) - H, H):
        print("Forecasting {} Started...".format(i // 24 + 1))
        forecast, conf_int = model.predict(H, return_conf_int=True, alpha=1 - PI)
        lo, hi = compute_conf(conf_int)

        yhat.extend(forecast)
        lo_list.extend(lo)
        hi_list.extend(hi)
        train_end_list.extend([timestamps[dev_split + i + (H - 1)]] * H)

        model.update(test_data[i:i+H])

    # leftover test block
    if len(yhat) < len(test_data):
        remaining = len(test_data) - len(yhat)
        forecast, conf_int = model.predict(remaining, return_conf_int=True, alpha=1 - PI)
        lo, hi = compute_conf(conf_int)

        yhat.extend(forecast)
        lo_list.extend(lo)
        hi_list.extend(hi)
        train_end_list.extend([timestamps[test_split - 1]] * remaining)

    # Save TEST CSV
    test_df = pd.DataFrame({
        "timestamp": timestamps[dev_split:test_split],
        "yhat": yhat,
        "y_true": test_data,
        "lo": lo_list,
        "hi": hi_list,
        "horizon": [H] * len(test_data),
        "train_end": train_end_list
    })

    test_df.to_csv(f"../outputs/{cc}_forecast_test.csv", index=False)
    print(f"Saved: ../outputs/{cc}_forecast_test.csv")
