# live_online_sarima_correct_hourly.py
import pandas as pd
import numpy as np
from pmdarima.arima import ARIMA
import warnings, os, time
from datetime import timedelta
warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
CC = "FR"          # "AT", "FR", or "CH"
DATA_PATH = f"../data/{CC}_preprocessed_data.csv"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, f"{CC}_online_updates.csv")

SIMULATED_HOURS = 2500
INITIAL_HISTORY_DAYS = 120
REFIT_WINDOW_DAYS = 90

# Drift detection
EWMA_ALPHA = 0.1
Z_PERCENTILE_THRESHOLD = 0.95
Z_HISTORY_DAYS = 30

# Hard-coded orders (change here)
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 24)

# ===========================================================

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df.asfreq('h')
target_col = next(c for c in df.columns if "load_actual_entsoe_transparency" in c.lower())
y_full = df[target_col].ffill()

start_idx = 24 * INITIAL_HISTORY_DAYS
end_idx = start_idx + SIMULATED_HOURS
y_hist = y_full.iloc[:start_idx].copy()
y_future = y_full.iloc[start_idx:end_idx]

# Log
log = [["timestamp", "strategy", "reason", "duration_s"]]

# Initial model
train_init = y_hist[-24*REFIT_WINDOW_DAYS:]
model = ARIMA(order=ORDER, seasonal_order=SEASONAL_ORDER,
              suppress_warnings=True, trend='c')
model.fit(train_init)
print(f"Initial model {ORDER}x{SEASONAL_ORDER} fitted")

# Drift detection state
ewma_z = 0.0
z_abs_all = []           # all |z| ever seen in simulation
z_abs_recent = []        # last 30 days only
z_threshold = 2.0        # will be updated dynamically

# Metric buffers (only collected at 00:00)
mase_before, mase_after = [], []
cov_before, cov_after = [], []

def mase(y_true, y_pred, y_train):
    naive = np.mean(np.abs(np.diff(y_train, 24)))
    return np.mean(np.abs(y_true - y_pred)) / naive if naive > 0 else np.mean(np.abs(y_true - y_pred))

def coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

# ======================= TRUE HOURLY LIVE LOOP =======================
for i, (ts, true_val) in enumerate(y_future.items()):
    # 1. Ingest new observation
    y_hist[ts] = true_val

    # 2. Make 1-step-ahead prediction for the observation we just ingested
    pred_1step = model.predict(n_periods=1)[0]
    resid_std = max(np.std(model.resid()), 1e-6)
    z = abs(true_val - pred_1step) / resid_std

    # 3. Update drift detection (every hour!)
    z_abs_all.append(z)
    z_abs_recent.append(z)

    # Update EWMA of |z|
    ewma_z = EWMA_ALPHA * z + (1 - EWMA_ALPHA) * (ewma_z or z)

    # Keep only last 30 days for percentile
    cutoff = ts - timedelta(days=Z_HISTORY_DAYS)
    z_abs_recent = [z for t, z in zip(y_future.index[:i+1], z_abs_recent) if t >= cutoff]

    # Update dynamic threshold
    if len(z_abs_recent) >= 200:
        z_threshold = np.percentile(z_abs_recent, Z_PERCENTILE_THRESHOLD * 100)

    # 4. At 00:00 UTC → issue 24h forecast + record metrics
    if ts.hour == 0 and ts.minute == 0:
        fc24 = model.predict(n_periods=24)
        ci24 = model.predict(n_periods=24, return_conf_int=True)[1]
        lower24, upper24 = ci24[:, 0], ci24[:, 1]
        actual24 = y_full[ts:ts + timedelta(hours=23)].values

        mase_pre = mase(actual24, fc24, y_hist[-24*100:-24])
        cov_pre = coverage(actual24, lower24, upper24)
        mase_before.append(mase_pre)
        cov_before.append(cov_pre)

        print(f"[{ts.date()}] 24h forecast issued | EWMA|z|={ewma_z:.3f} | thresh={z_threshold:.3f}")

    # 5. Drift check — can trigger ANY hour
    drift_triggered = len(z_abs_recent) >= 200 and ewma_z > z_threshold

    # 6. Adaptation: either scheduled (00:00) OR drift (any hour)
    do_adapt = (ts.hour == 0 and ts.minute == 0) or drift_triggered
    reason = "drift" if drift_triggered else "scheduled" if ts.hour == 0 else None

    if do_adapt:
        start_t = time.time()
        window_start = ts - timedelta(days=REFIT_WINDOW_DAYS)
        train_win = y_hist[window_start:ts]

        try:
            model.update(train_win, maxiter=15)
        except:
            model = ARIMA(order=ORDER, seasonal_order=SEASONAL_ORDER,
                          suppress_warnings=True, trend='c')
            model.fit(train_win)

        duration = time.time() - start_t

        # If we adapted at 00:00, also record post-adaptation metrics
        if ts.hour == 0 and ts.minute == 0:
            fc24_post = model.predict(n_periods=24)
            ci_post = model.predict(n_periods=24, return_conf_int=True)[1]
            lower_p, upper_p = ci_post[:, 0], ci_post[:, 1]
            actual24 = y_full[ts:ts + timedelta(hours=23)].values

            mase_after.append(mase(actual24, fc24_post, train_win))
            cov_after.append(coverage(actual24, lower_p, upper_p))

        log.append([ts.strftime("%Y-%m-%d %H:%M"), f"SARIMA{ORDER}x{SEASONAL_ORDER}", reason, f"{duration:.3f}"])
        print(f"  ADAPTED at {ts} | reason: {reason.upper()} | {duration:.2f}s")

    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{SIMULATED_HOURS} hours")

# Save log + final metrics
pd.DataFrame(log[1:], columns=log[0]).to_csv(LOG_FILE, index=False)
print(f"\nLog saved: {LOG_FILE}")

if mase_before:
    roll = 7 * 24
    print("\n" + "="*70)
    print("FINAL 7-DAY ROLLING METRICS (collected at 00:00)")
    print("="*70)
    print(f"{'':<20} {'Before':>15} {'After':>15}")
    print(f"MASE            {pd.Series(mase_before).rolling(roll,min_periods=1).mean().iloc[-1]:15.4f} "
          f"{pd.Series(mase_after).rolling(roll,min_periods=1).mean().iloc[-1]:15.4f}")
    print(f"80% Coverage    {pd.Series(cov_before).mean().iloc[-1]*100:14.1f}% "
          f"{pd.Series(cov_after).mean().iloc[-1]*100:14.1f}%")
    print("="*70)

print("True hourly live simulation completed correctly!")