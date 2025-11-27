"""
Step 4: Live ingestion + online adaptation (simulated stream)

This module simulates a live data feed and implements online model adaptation.
Picks ONE adaptation strategy:
  - Strategy A: Rolling SARIMA refit (simple & robust)
  - Strategy B: Tiny neural fine-tune (GRU/LSTM)

Logs all updates to outputs/<CC>_online_updates.csv with:
  - timestamp, strategy, reason, duration_s
"""

import os
import time
import pandas as pd
import numpy as np
import pmdarima as pm
from metrics import mase
from load_opsd import load_data


class LiveAdaptationLoop:
    """Simulated live data ingestion with online model adaptation."""

    def __init__(
        self,
        country_code,
        strategy="rolling_sarima",
        adaptation_window_days=90,
        refit_frequency_hours=24,
        drift_threshold_multiplier=1.2,
        data_dir="../data",
        outputs_dir="../outputs"
    ):
        """
        Initialize the live adaptation loop.

        Args:
            country_code: Country code (e.g., "AT", "CH", "FR")
            strategy: "rolling_sarima" or "tiny_neural"
            adaptation_window_days: Days of history to use for refit (default 90)
            refit_frequency_hours: Hours between scheduled refits (default 24)
            drift_threshold_multiplier: Multiplier for drift threshold (default 1.2)
            data_dir: Directory with preprocessed data
            outputs_dir: Directory to save outputs
        """
        self.country_code = country_code
        self.strategy = strategy
        self.adaptation_window_days = adaptation_window_days
        self.refit_frequency_hours = refit_frequency_hours
        self.drift_threshold_multiplier = drift_threshold_multiplier
        self.data_dir = data_dir
        self.outputs_dir = outputs_dir

        # SARIMA parameters (from decompose_acf_pacf analysis)
        self.sarima_order = (1, 1, 1)
        self.sarima_seasonal_order = (1, 1, 2, 24)
        self.pi_level = 0.8

        # Drift detection
        self.alpha_ewma = 0.1
        self.drift_window_hours = 30 * 24
        self.z_scores_history = []
        self.ewma_z = None
        self.z_threshold_95th = None

        # Model and data
        self.model = None
        self.historical_data = None
        self.historical_timestamps = None
        self.last_refit_time = None
        self.current_hour = 0

        # Logging
        self.updates_log = []
        self.metrics_before_after = []
        self.forecast_history = []  # Cache forecasts for metrics

        # Load data
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical data and prepare for live simulation."""
        print(f"Loading historical data for {self.country_code}...")
        raw = load_data(self.country_code, self.data_dir)

        self.historical_data = raw["load_actual_entsoe_transparency"].dropna().values.astype(float)
        self.historical_timestamps = raw["timestamp"].values

        # Calculate 80/10/10 split
        n = len(self.historical_data)
        self.train_split = int(n * 0.8)
        self.dev_split = self.train_split + int(n * 0.1)
        self.test_split = n

        print(f"  Total: {n} hours | Train: {self.train_split} | Dev: {self.dev_split-self.train_split} | Test: {self.test_split-self.dev_split}")

        # Initialize model on training data
        self._fit_initial_model()

    def _fit_initial_model(self):
        """Fit initial SARIMA model on training data (80% split)."""
        print(f"Fitting initial SARIMA model on training data...")
        train_data = self.historical_data[:self.train_split]

        self.model = pm.arima.ARIMA(
            order=self.sarima_order,
            seasonal_order=self.sarima_seasonal_order,
            maxiter=20,
            method='lbfgs',
            enforce_stationarity=False,
            enforce_invertibility=False,
            suppress_warnings=True
        )
        self.model.fit(train_data)
        self.last_refit_time = 0

    def _compute_forecast(self, horizon=24):
        """Generate forecast with prediction intervals."""
        try:
            forecast, conf_int = self.model.predict(horizon, return_conf_int=True, alpha=1 - self.pi_level)
            lo = conf_int[:, 0]
            hi = conf_int[:, 1]
            return forecast, lo, hi
        except Exception as e:
            print(f"  Warning: Forecast failed ({e}), using NaN")
            return np.full(horizon, np.nan), np.full(horizon, np.nan), np.full(horizon, np.nan)

    def _compute_single_zscore(self, residual):
        """
        Compute z-score for a single residual using rolling window statistics.
        Uses accumulated residual history for better estimation.
        """
        if len(self.z_scores_history) == 0:
            return residual  # Return raw residual if no history

        # Use the entire history to compute rolling stats
        window = 336
        min_periods = 168

        residual_series = np.array(self.z_scores_history + [residual])
        roll_mean = pd.Series(residual_series).rolling(window=window, min_periods=min_periods).mean().iloc[-1]
        roll_std = pd.Series(residual_series).rolling(window=window, min_periods=min_periods).std(ddof=0).iloc[-1]

        z_score = (residual - roll_mean) / (roll_std + 1e-8)
        return z_score

    def _detect_drift(self, z_score):
        """
        Detect drift using EWMA of z-scores.
        Returns True if EWMA(|z|) > threshold based on recent z-score percentiles.
        """
        # Update z-score history
        self.z_scores_history.append(z_score)
        if len(self.z_scores_history) > self.drift_window_hours:
            self.z_scores_history.pop(0)

        # Compute EWMA on absolute z-scores
        abs_z = np.abs(z_score)
        if self.ewma_z is None:
            self.ewma_z = abs_z
        else:
            self.ewma_z = self.alpha_ewma * abs_z + (1 - self.alpha_ewma) * self.ewma_z

        # Compute adaptive threshold based on recent z-score distribution
        min_history = 240  # 10 days minimum for threshold

        if len(self.z_scores_history) >= min_history:
            abs_z_history = np.abs(np.array(self.z_scores_history))

            # Use 90th percentile for more sensitivity (was 95th)
            threshold_base = np.percentile(abs_z_history, 90)

            # Adaptive multiplier: higher if recent volatility is low, lower if high
            recent_abs_z = np.abs(np.array(self.z_scores_history[-168:]))  # Last 7 days
            recent_median = np.median(recent_abs_z)
            recent_std = np.std(recent_abs_z, ddof=0)

            # If volatility is low, be more sensitive to changes
            if recent_std < 0.5:
                adaptive_multiplier = 0.8
            elif recent_std < 1.0:
                adaptive_multiplier = 1.0
            else:
                adaptive_multiplier = 1.2

            self.z_threshold_95th = threshold_base * adaptive_multiplier
            drift_triggered = self.ewma_z > self.z_threshold_95th

        else:
            # During warmup period (< 10 days), use looser threshold
            if len(self.z_scores_history) >= 48:  # At least 2 days
                abs_z_history = np.abs(np.array(self.z_scores_history))
                self.z_threshold_95th = np.percentile(abs_z_history, 85)  # Very loose
                drift_triggered = self.ewma_z > self.z_threshold_95th * 1.5
            else:
                self.z_threshold_95th = np.inf
                drift_triggered = False

        return drift_triggered

    def _refit_rolling_sarima(self):
        """
        Refit SARIMA on last N days of data.
        Uses two strategies:
        1. Try incremental update first (faster, more stable)
        2. If that fails, do full refit on rolling window
        """
        lookback_hours = self.adaptation_window_days * 24

        if self.current_hour >= lookback_hours:
            start_idx = self.current_hour - lookback_hours
        else:
            start_idx = 0

        end_idx = self.current_hour + 1
        data_to_refit = self.historical_data[start_idx:end_idx]

        print(f"  [Refit] Adapting SARIMA on {len(data_to_refit)} hours (window={self.adaptation_window_days}d)")

        try:
            # Strategy 1: Try incremental update (more stable)
            if self.last_refit_time is not None and self.current_hour > self.last_refit_time:
                hours_since_refit = self.current_hour - self.last_refit_time
                if hours_since_refit <= 24:  # Only if less than 1 day has passed
                    new_data = self.historical_data[self.last_refit_time:end_idx]
                    if len(new_data) > 0:
                        try:
                            self.model.update(new_data)
                            self.last_refit_time = self.current_hour
                            print(f"    ✓ Incremental update on {len(new_data)} new points")
                            return True
                        except Exception as e:
                            print(f"    ⚠ Incremental update failed ({e}), trying full refit...")

            # Strategy 2: Full refit on rolling window
            self.model = pm.arima.ARIMA(
                order=self.sarima_order,
                seasonal_order=self.sarima_seasonal_order,
                maxiter=15,  # Reduced iterations to prevent overfitting
                method='lbfgs',
                enforce_stationarity=False,
                enforce_invertibility=False,
                suppress_warnings=True
            )
            self.model.fit(data_to_refit)
            self.last_refit_time = self.current_hour
            print(f"    ✓ Full refit completed")
            return True

        except Exception as e:
            print(f"    ✗ Refit failed ({e}), keeping previous model")
            return False

    def _compute_rolling_metrics(self, window_hours=7*24):
        """Compute rolling metrics (MASE, PI coverage) over last N hours."""
        if len(self.forecast_history) < window_hours:
            return {"MASE": np.nan, "PI_Coverage": np.nan}

        # Use cached forecast history for metrics
        recent_forecasts = self.forecast_history[-window_hours:]

        y_true_window = np.array([f['y_true'] for f in recent_forecasts])
        yhat_window = np.array([f['yhat'] for f in recent_forecasts])
        lo_window = np.array([f['lo'] for f in recent_forecasts])
        hi_window = np.array([f['hi'] for f in recent_forecasts])

        # Filter out NaNs
        valid_mask = ~(np.isnan(yhat_window) | np.isnan(y_true_window))
        if valid_mask.sum() < 10:
            return {"MASE": np.nan, "PI_Coverage": np.nan}

        y_true_valid = y_true_window[valid_mask]
        yhat_valid = yhat_window[valid_mask]
        lo_valid = lo_window[valid_mask]
        hi_valid = hi_window[valid_mask]

        try:
            mase_val = mase(y_true_valid, yhat_valid, seasonality=24)
            pi_coverage = np.mean((y_true_valid >= lo_valid) & (y_true_valid <= hi_valid))
            return {"MASE": mase_val, "PI_Coverage": pi_coverage}
        except:
            return {"MASE": np.nan, "PI_Coverage": np.nan}

    def _log_update(self, timestamp, strategy, reason, duration_s):
        """Log an online update event."""
        log_entry = {
            "timestamp": timestamp,
            "strategy": strategy,
            "reason": reason,
            "duration_s": duration_s
        }
        self.updates_log.append(log_entry)

    def run_live_simulation(self, max_hours=2000):
        """
        Run live simulation on test data.

        Args:
            max_hours: Maximum hours to simulate (default 2000)
        """
        print(f"\n{'='*70}")
        print(f"Starting Live Simulation for {self.country_code}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*70}\n")

        # Simulate starting from test split
        start_idx = self.dev_split
        end_idx = min(self.dev_split + max_hours, self.test_split)

        num_hours = end_idx - start_idx
        print(f"Simulating {num_hours} hours from index {start_idx} to {end_idx}\n")

        # Main loop
        yhat_24h = None
        lo_24h = None
        hi_24h = None
        residual_history = []  # Track residuals for z-score computation

        for step, hour_idx in enumerate(range(start_idx, end_idx)):
            self.current_hour = hour_idx
            timestamp = self.historical_timestamps[hour_idx]
            y_current = self.historical_data[hour_idx]

            start_time = time.time()

            # ============ FORECAST ============
            # At 00:00 UTC, generate 24-hour forecast
            ts = pd.to_datetime(timestamp)
            is_midnight = (ts.hour == 0)

            if is_midnight or step == 0:
                yhat_24h, lo_24h, hi_24h = self._compute_forecast(horizon=24)

            # ============ COMPUTE RESIDUAL & Z-SCORE ============
            # Get the corresponding forecast value for this hour
            hour_of_day = ts.hour
            yhat_1step = yhat_24h[hour_of_day] if yhat_24h is not None and hour_of_day < len(yhat_24h) else np.nan

            if not np.isnan(yhat_1step):
                residual = y_current - yhat_1step
                residual_history.append(residual)

                # Keep rolling history
                if len(residual_history) > 336:
                    residual_history.pop(0)

                # Compute z-score using rolling window
                if len(residual_history) >= 168:
                    roll_mean = np.mean(residual_history[-336:]) if len(residual_history) >= 336 else np.mean(residual_history)
                    roll_std = np.std(residual_history[-336:], ddof=0) if len(residual_history) >= 336 else np.std(residual_history, ddof=0)
                    z_current = (residual - roll_mean) / (roll_std + 1e-8)
                else:
                    z_current = residual

                # ============ DETECT DRIFT ============
                drift_triggered = self._detect_drift(z_current)

                # ============ CACHE FORECAST FOR METRICS ============
                self.forecast_history.append({
                    'timestamp': timestamp,
                    'y_true': y_current,
                    'yhat': yhat_1step,
                    'lo': lo_24h[hour_of_day] if lo_24h is not None and hour_of_day < len(lo_24h) else np.nan,
                    'hi': hi_24h[hour_of_day] if hi_24h is not None and hour_of_day < len(hi_24h) else np.nan,
                    'z_score': z_current
                })
            else:
                drift_triggered = False

            # ============ DETERMINE ADAPTATION REASON ============
            reason = None
            if step == 0:
                reason = "initial"
            elif is_midnight and step > 0:
                reason = "scheduled"
            elif drift_triggered and len(residual_history) >= 168:
                reason = "drift"

            # ============ ADAPT MODEL ============
            if reason is not None:
                if self.strategy == "rolling_sarima":
                    adapt_start = time.time()
                    self._refit_rolling_sarima()
                    duration_s = time.time() - adapt_start
                    self._log_update(timestamp, self.strategy, reason, duration_s)
                    print(f"[{step:5d}] {timestamp} | Adapted (reason={reason}) | elapsed={duration_s:.3f}s")

            # ============ LOG ROLLING METRICS ============
            if step % 168 == 0 and step > 0:  # Every 7 days (168 hours)
                metrics_snapshot = self._compute_rolling_metrics(window_hours=7*24)
                self.metrics_before_after.append({
                    "hour": step,
                    "timestamp": timestamp,
                    "metrics": metrics_snapshot
                })
                if not np.isnan(metrics_snapshot['MASE']):
                    print(f"  [Metrics] MASE: {metrics_snapshot['MASE']:.4f}, PI Coverage: {metrics_snapshot['PI_Coverage']:.4f}")

            # Progress indicator
            if (step + 1) % 500 == 0:
                print(f"[Progress] {step+1}/{num_hours} hours completed")

        print(f"\nLive simulation completed: {num_hours} hours, {len(self.updates_log)} adaptations logged")

    def save_results(self):
        """Save online updates log and metrics to CSV."""
        # Save updates log
        updates_df = pd.DataFrame(self.updates_log)
        updates_path = os.path.join(self.outputs_dir, f"{self.country_code}_online_updates.csv")
        updates_df.to_csv(updates_path, index=False)
        print(f"Saved: {updates_path}")

        # Save rolling metrics
        metrics_path = os.path.join(self.outputs_dir, f"{self.country_code}_rolling7d_metrics.csv")
        if self.metrics_before_after:
            metrics_df = pd.DataFrame(self.metrics_before_after)
            metrics_df.to_csv(metrics_path, index=False)
            print(f"Saved: {metrics_path}")

        # Summary stats
        print(f"\n{'='*70}")
        print(f"Live Simulation Summary for {self.country_code}")
        print(f"{'='*70}")
        print(f"Total updates logged: {len(self.updates_log)}")
        print(f"Strategy used: {self.strategy}")

        if self.updates_log:
            reasons = pd.DataFrame(self.updates_log)['reason'].value_counts()
            print(f"\nAdaptation reasons:")
            for reason, count in reasons.items():
                print(f"  {reason}: {count}")


def main():
    """Main entry point for live loop simulation."""
    # Hard coded parameters
    country_codes = ["AT", "CH", "FR"][:2]
    strategy = "rolling_sarima"
    max_hours = 2000
    adaptation_window_days = 90
    data_dir = "../data"
    outputs_dir = "../outputs"

    # Run live simulation for all countries
    for country_code in country_codes:
        print(f"\n\n{'#'*70}")
        print(f"# Processing Country: {country_code}")
        print(f"{'#'*70}\n")

        live_loop = LiveAdaptationLoop(
            country_code=country_code,
            strategy=strategy,
            adaptation_window_days=adaptation_window_days,
            data_dir=data_dir,
            outputs_dir=outputs_dir
        )

        live_loop.run_live_simulation(max_hours=max_hours)
        live_loop.save_results()


if __name__ == "__main__":
    main()

