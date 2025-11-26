import numpy as np

def mase(y_true, yhat, seasonality):
    errors = np.abs(y_true - yhat)

    denom = np.abs(y_true[seasonality:] - y_true[:-seasonality]).mean()

    return errors.mean() / denom


def smape(y_true, yhat):
    return (2 * np.mean(np.abs(yhat - y_true) / (np.abs(y_true) + np.abs(yhat) + 1e-8)))


def mse(y_true, yhat):
    return np.mean((y_true - yhat) ** 2)


def rmse(y_true, yhat):
    return np.sqrt(mse(y_true, yhat))


def mape(y_true, yhat):
    return np.mean(np.abs((y_true - yhat) / (y_true + 1e-8)))


def prediction_interval_coverage(y_true, lo, hi):
    inside = ((y_true >= lo) & (y_true <= hi)).astype(int)
    return inside.mean()