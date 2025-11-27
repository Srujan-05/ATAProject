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


country_codes = ["CH", "FR", "AT"]
for country_code in country_codes:
    print(country_code)
    raw_data = load_data(country_code, "../data")
    data = raw_data["load_actual_entsoe_transparency"].dropna()
    data = data.values.astype(float)

    n = len(data)
    train_split = int(n*0.8)
    dev_split = train_split + int(n*0.1)
    test_split = dev_split + (n-dev_split)
    # print(n, train_split, test_split, dev_split)
    train_data = data[:train_split]
    dev_data = data[train_split: dev_split]
    test_data = data[dev_split: test_split]
    print("Starting BackTesting for Dev: ", len(train_data), len(dev_data), len(test_data), train_split, dev_split, test_split)
    # dev back-testing
    forecasts = np.array([])
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
    print(len(train_data), len(dev_data), len(test_data))
    for i in range(0, len(dev_data) - 24, 24):
        print("Backtesting {} Started...".format(i // 24 + 1))
        forecast, conf_int = model.predict(m, return_conf_int=True, alpha=1 - PI)
        model.update(dev_data[i:i + 24])
        forecasts = np.hstack((forecasts, forecast))
        z = norm.ppf(1 - (1 - PI) / 2)
        lo = conf_int[:, 0]
        hi = conf_int[:, 1]
        forecast_std = (hi - lo) / (2 * z)
        print(np.vstack(
            (raw_data["timestamp"].values[train_split + i:train_split + i + 24], forecast, lo, hi, forecast_std,
             np.full(len(forecast), raw_data["timestamp"].values[train_split + i + 23]))))
    if len(forecasts) < len(dev_data):
        forecast, conf_int = model.predict(len(dev_data) - len(forecasts), return_conf_int=True, alpha=1 - PI)
        z = norm.ppf(1 - (1 - PI) / 2)
        lo = conf_int[:, 0]
        hi = conf_int[:, 1]
        forecast_std = (hi - lo) / (2 * z)
        print(np.vstack(
            (raw_data["timestamp"].values[train_split + len(forecasts):dev_split], forecast, lo, hi, forecast_std,
             np.full(len(forecast), raw_data["timestamp"].values[-1]))))
        forecasts = np.hstack((forecasts, forecast))

    pd.DataFrame(np.vstack((raw_data["timestamp"].values[train_split:dev_split], forecasts, dev_data)),
                 columns=["timestamp", country_code + "_forecast", country_code + "_actual"]).to_csv(
        "../outputs/" + country_code + "_forecast_dev.csv", index=False)

    # else:
        # forecasts = np.array([])
        # model = pm.arima.ARIMA(
        #     order=(p, d, q),
        #     seasonal_order=(P, D, Q, m),
        #     maxiter=20,
        #     method='lbfgs',
        #     enforce_stationarity=False,
        #     enforce_invertibility=False,
        #     suppress_warnings=True
        # )
        # i = (len(dev_data)//24)*24
        # print(i, list(range(train_split+i, dev_split)))
        # model.fit(np.hstack((train_data, dev_data[i-24:i])))
        # forecast = model.predict(len(dev_data) - i)
        # print(np.vstack((raw_data["timestamp"].values[train_split + i:dev_split], forecast)))
        # data = pd.read_csv("../outputs/"+country_code+"_forecast_dev.csv")
        # data = data.drop(columns="Unnamed: 0")
        # # data["{}_actual".format(country_code)] = dev_data
        # data.to_csv("../outputs/" + country_code + "_forecast_dev.csv", index=False)

    print("Starting BackTesting for Test: ", len(train_data), len(dev_data), len(test_data), train_split, dev_split, test_split)
    # forecasting test
    forecasts = np.array([])
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
    for i in range(0, len(test_data) - 24, 24):
        print("Forecasting {} Started...".format(i // 24 + 1))
        forecast, conf_int = model.predict(m, return_conf_int=True, alpha=1-PI)
        model.update(test_data[i:i+24])
        z = norm.ppf(1 - (1 - PI) / 2)
        lo = conf_int[:, 0]
        hi = conf_int[:, 1]
        forecast_std = (hi - lo) / (2 * z)
        forecasts = np.hstack((forecasts, forecast))
        print(np.vstack((raw_data["timestamp"].values[dev_split + i:dev_split + i + 24], forecast, lo, hi, forecast_std, np.full(len(forecast), raw_data["timestamp"].values[dev_split+i+23]))))

    if len(forecasts) < len(test_data):
        forecast, conf_int = model.predict(len(test_data)-len(forecasts), return_conf_int=True, alpha=1-PI)
        z = norm.ppf(1 - (1 - PI) / 2)
        lo = conf_int[:, 0]
        hi = conf_int[:, 1]
        forecast_std = (hi - lo) / (2 * z)
        print(np.vstack((raw_data["timestamp"].values[dev_split + len(forecasts):], forecast, lo, hi, forecast_std, np.full(len(forecast), raw_data["timestamp"].values[-1]))))
        forecasts = np.hstack((forecasts, forecast))

    pd.DataFrame(np.vstack((raw_data["timestamp"].values[dev_split:], forecasts, test_data)),
                 columns=["timestamp", country_code + "_forecast", country_code + "_actual"]).to_csv(
        "../outputs/" + country_code + "_forecast_test.csv")
