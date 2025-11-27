import os.path
import numpy
import numpy as np
from load_opsd import load_data
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import statsmodels.api as sm

def retrieve_order(series, m):
    nlags = len(series)
    threshlod = 2/np.sqrt(nlags)

    latest = 0
    for lag in range(nlags):
        if abs(series[lag]) > threshlod and lag % m != 0:
            latest = lag

    return latest

def adf_test(series, title=""):
    result = adfuller(series, autolag="AIC")
    print(f"\n=== ADF Test: {title} ===")
    print(f"  Test Statistic = {result[0]:.6f}")
    print(f"  p-value        = {result[1]:.8f}")
    print(f"  Stationary?    → {'YES' if result[1] < 0.05 else 'NO'}")

def differencing(series, s = 1):
    return series[s:] - series[:-s]

def acf_pacf_plot(acf_values, title, save_dest=None):
    plt.stem(range(len(acf_values)), acf_values)
    plt.title(title)
    if save_dest is not None:
        plt.savefig(save_dest)
    plt.show()

def grid_search(data, p, q, P, Q, d, D, m):
    start_p = 0
    max_p = 2
    start_q = 0
    max_q = 2
    start_P = 0
    max_P = 1
    start_Q = 0
    max_Q = 1
    start_d = 0
    max_d = 1
    start_D = 0
    max_D = 1

    def expand(x, lo, hi):
        vals = [x]
        for v in [x - 1, x + 1]:
            if lo <= v <= hi and v >= 0:
                vals.append(v)
        return sorted(set(vals))

    p_vals = expand(p, start_p, max_p)
    q_vals = expand(q, start_q, max_q)

    P_vals = expand(P, start_P, max_P)
    Q_vals = expand(Q, start_Q, max_Q)

    d_vals = expand(d, start_d, max_d)
    D_vals = expand(D, start_D, max_D)

    cominations = {}
    for p in p_vals:
        for q in q_vals:
            for P in P_vals:
                for Q in Q_vals:
                    for d in d_vals:
                        for D in D_vals:
                            model = sm.tsa.SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, m))
                            res = model.fit()
                            cominations[(p, q, d, P, D, Q, m)] = (res.bic, res.aic)
                            print("SARIMA({} {} {})x({} {} {} {}): BIC:{}, AIC:{}".format(p, d, q, P, D, Q, m, res.bic, res.aic))

    values = numpy.array(list(cominations.values()))
    values.sort(axis=0)

    for order in cominations:
        if cominations[order][0] == values[0][0]:
            return list(order)

def get_model_params(country_code="AT", data_dir="../data", data_col="load_actual_entsoe_transparency", m = 24, D = 1, d = 1, output_dir="../outputs"):
    raw_data = load_data(country_code, data_dir)[data_col].dropna()
    raw_data = raw_data.values.astype(float)

    stl_decomp = seasonal_decompose(raw_data[-1000:], model="additive", period=m)

    trend = stl_decomp.trend
    seasonal = stl_decomp.seasonal
    level = stl_decomp.resid

    fig, [a1, a2, a3] = plt.subplots(3, 1)
    a1.plot(range(len(trend)), trend)
    a1.set_title(country_code + " Trend from STL decomposition")
    a2.plot(range(len(seasonal)), seasonal)
    a2.set_title(country_code + " Seasonal from STL decomposition")
    a3.plot(range(len(level)), level)
    a3.set_title(country_code + " Level from STL decomposition")
    plt.savefig(os.path.join(output_dir, country_code + "_stl_decomposition_for_past_1000_steps"))
    plt.show()

    # res = STL(fr_raw_data, period=m).fit()
    # res.plot()
    # plt.show()

    adf_test(raw_data, "RAW DATA")
    # print(adf_raw[1])

    # Seasonal Differencing:-
    i = 0
    season_diff = raw_data
    while i < D:
        season_diff = differencing(season_diff, m) # Daily Differencing
        # week_season_diff = differencing(season_diff, m*7) # Weekly Differencing
        # season_diff = week_season_diff
        i += 1

    adf_test(season_diff, "POST SEASONAL DIFFERENCING")

    # Trend Differencing:-
    j = 0
    trend_diff = season_diff
    while j < d:
        trend_diff = differencing(trend_diff)
        j += 1

    adf_test(trend_diff, "POST TREND DIFFERENCING")


    acf_vals = acf(trend_diff, nlags=48)
    pacf_vals = pacf(trend_diff, nlags=48)

    sacf_vals = acf(trend_diff[::m], nlags=48)
    spacf_vals = pacf(trend_diff[::m], nlags=48)

    p = retrieve_order(pacf_vals, m)
    q = retrieve_order(acf_vals, m)
    P = retrieve_order(sacf_vals, m)
    Q = retrieve_order(spacf_vals, m)

    print("Suggested p:{}, q:{}, P:{}, Q:{}".format(p, q, P, Q))

    acf_pacf_plot(acf_vals, "ACF after trend differencing, d={}".format(d), os.path.join(output_dir, country_code+"_acf_plot"))

    acf_pacf_plot(pacf_vals, "PACF after trend differencing, d={}".format(d), os.path.join(output_dir, country_code+"_pacf_plot"))

    acf_pacf_plot(sacf_vals, "ACF after seasonal differencing, D={}".format(D), os.path.join(output_dir, country_code+"_seasonal_acf_plot"))

    acf_pacf_plot(spacf_vals, "PACF after seasonal differencing, D={}".format(D), os.path.join(output_dir, country_code+"_seasonal_pacf_plot"))

    # fig, [[a1, a2], [a3, a4]] = plt.subplots(2, 2)
    #
    # a1.stem(range(len(acf_vals)), acf_vals)
    # a1.set_title("ACF after trend differencing, d={}".format(d))
    #
    # a2.stem(range(len(pacf_vals)), pacf_vals)
    # a2.set_title("PACF after trend differencing, d={}".format(d))
    #
    # a3.stem(range(len(sacf_vals)), sacf_vals)
    # a3.set_title("ACF after seasonal differencing, D={}".format(D))
    #
    # a4.stem(range(len(spacf_vals)), spacf_vals)
    # a4.set_title("PACF after seasonal differencing, D={}".format(D))
    #
    # plt.show()

    data = load_data(country_code, data_dir)[data_col].dropna()
    data = data.values.astype(float)
    p_suit, q_suit, d_suit, P_suit, D_suit, Q_suit, _ = grid_search(data, p= p, P=P, Q=Q, q=q, D=D, d=d, m=m)
    print("Best Fitting Model orders: ({}, {}, {})x({}, {}, {}, {})".format(p_suit, d_suit, q_suit, P_suit, D_suit, Q_suit, m))
    return p_suit, q_suit, d_suit, P_suit, D_suit, Q_suit, m

if __name__ == "__main__":
    get_model_params()