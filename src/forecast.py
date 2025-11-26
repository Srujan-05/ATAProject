import numpy as np
import pandas as pd
from load_opsd import load_data
import pmdarima as pm
from metrics import rmse, smape, mse, mape, mase, prediction_interval_coverage


def compute_conf(conf_int):
    lo = conf_int[:, 0]
    hi = conf_int[:, 1]
    return lo, hi


def compute_metrics_from_csv(path, seasonality=24):
    df = pd.read_csv(path)

    y_true = df["y_true"].values
    yhat = df["yhat"].values

    lo = df["lo"].values if "lo" in df else None
    hi = df["hi"].values if "hi" in df else None

    results = {}

    results["MASE"] = mase(y_true, yhat, seasonality)
    results["sMAPE"] = smape(y_true, yhat)
    results["MSE"] = mse(y_true, yhat)
    results["RMSE"] = rmse(y_true, yhat)
    results["MAPE"] = mape(y_true, yhat)

    # If intervals available
    if lo is not None and hi is not None:
        results["PI80_Coverage"] = prediction_interval_coverage(y_true, lo, hi)
    else:
        results["PI80_Coverage"] = None

    return results


def compute_dev_test_metrics(country_code, directory="../outputs"):
    dev_path = f"{directory}/{country_code}_forecast_dev.csv"
    test_path = f"{directory}/{country_code}_forecast_test.csv"

    dev_metrics = compute_metrics_from_csv(dev_path)
    test_metrics = compute_metrics_from_csv(test_path)

    return dev_metrics, test_metrics


def compute_all_metrics(country_codes, directory="../outputs"):
    all_results = {}

    for cc in country_codes:
        all_results[cc] = compute_dev_test_metrics(cc, directory)

    return all_results

def build_comparison_tables(country_codes, directory="../outputs"):
    dev_rows = {}
    test_rows = {}

    for cc in country_codes:
        dev_metrics, test_metrics = compute_dev_test_metrics(cc, directory)
        dev_rows[cc] = dev_metrics
        test_rows[cc] = test_metrics

    dev_df = pd.DataFrame.from_dict(dev_rows, orient="index")
    test_df = pd.DataFrame.from_dict(test_rows, orient="index")

    return dev_df, test_df


def backtesting_and_forecasting(country_codes, order, seasonal_order, horizon, PI):
    p, d, q = (1, 1, 1) if order is None else [order[0], order[1], order[2]]
    P, D, Q, m = (1, 1, 2, 24) if seasonal_order is None else [seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3]]
    PI = 0.8 if PI is None else PI
    H = 24 if horizon is None else horizon
    print(p, d, q, P, D, Q, m)
    country_codes = ["CH", "FR", "AT"] if country_codes is None else country_codes

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
                forecast, conf_int = model.predict(H, return_conf_int=True, alpha=1 - PI)
                lo, hi = compute_conf(conf_int)

                yhat.extend(forecast)
                lo_list.extend(lo)
                hi_list.extend(hi)

                train_end_list.extend([timestamps[train_split + i + (H - 1)]] * H)

                model.update(dev_data[i:i+H])

            if len(yhat) < len(dev_data):
                remaining = len(dev_data) - len(yhat)
                forecast, conf_int = model.predict(remaining, return_conf_int=True, alpha=1 - PI)
                lo, hi = compute_conf(conf_int)

                yhat.extend(forecast)
                lo_list.extend(lo)
                hi_list.extend(hi)
                train_end_list.extend([timestamps[dev_split - 1]] * remaining)

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
            forecast, conf_int = model.predict(H, return_conf_int=True, alpha=1 - PI)
            lo, hi = compute_conf(conf_int)

            yhat.extend(forecast)
            lo_list.extend(lo)
            hi_list.extend(hi)
            train_end_list.extend([timestamps[dev_split + i + (H - 1)]] * H)

            model.update(test_data[i:i+H])

        if len(yhat) < len(test_data):
            remaining = len(test_data) - len(yhat)
            forecast, conf_int = model.predict(remaining, return_conf_int=True, alpha=1 - PI)
            lo, hi = compute_conf(conf_int)

            yhat.extend(forecast)
            lo_list.extend(lo)
            hi_list.extend(hi)
            train_end_list.extend([timestamps[test_split - 1]] * remaining)

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


if __name__ == "__main__":
    dev_table, test_table = build_comparison_tables(["CH", "FR", "AT"])

    print("\n===== DEV METRICS COMPARISON =====")
    print(dev_table)

    print("\n===== TEST METRICS COMPARISON =====")
    print(test_table)
