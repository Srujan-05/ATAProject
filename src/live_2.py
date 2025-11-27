import pandas as pd
import numpy as np
import warnings
import os
import time
from datetime import timedelta
import pmdarima as pm
from pmdarima import ARIMA

warnings.filterwarnings('ignore')


def load_country_data(country_code):
    """Load preprocessed data for a specific country"""
    file_path = f'../data/{country_code}_preprocessed_data.csv'
    try:
        data = pd.read_csv(file_path)
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')

        # Ensure we have the target column
        if 'load_actual_entsoe_transparency' not in data.columns:
            raise KeyError(f"Target column 'load_actual_entsoe_transparency' not found in data for {country_code}")

        # Return only the target column for simulation
        return data[['load_actual_entsoe_transparency']]

    except FileNotFoundError:
        print(f"Data file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data for {country_code}: {e}")
        return None


class ImprovedLiveSARIMAAdapter:
    def __init__(self, country_data, country_code, history_days=120):
        self.country_code = country_code
        self.data = country_data.copy()
        self.history_days = history_days

        # More stable SARIMA parameters for electricity load
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 2, 24)
        self.rolling_window = 90 * 24  # 90 days in hours

        # Adaptation parameters
        self.adaptation_frequency = 24  # daily at 00:00 UTC
        self.last_adaptation = 0

        # Improved drift detection
        self.alpha = 0.05  # Slower EWMA for stability
        self.drift_window = 30 * 24  # 30 days in hours
        self.z_scores = []
        self.ewma_z = None
        self.recent_errors = []
        self.error_window = 7 * 24  # 7 days for error statistics

        # Model stability
        self.model = None
        self.is_fitted = False
        self.best_model = None
        self.best_mase = float('inf')

        # Logging
        self.log = []
        self.performance_metrics = []

    def initial_training(self, training_data):
        """Initial training on historical data with stability checks"""
        print(f"Performing initial SARIMA training for {self.country_code}...")

        if len(training_data) < self.rolling_window:
            print(f"Not enough data for initial training. Need {self.rolling_window}, got {len(training_data)}")
            return False

        try:
            # Use the last rolling_window points for initial training
            train_data = training_data.tail(self.rolling_window)

            print(f"Training SARIMA{self.order}{self.seasonal_order} on {len(train_data)} points...")

            # Fit SARIMA model with better stability settings
            self.model = ARIMA(
                order=self.order,
                seasonal_order=self.seasonal_order,
                suppress_warnings=True,
                error_action='ignore',
                maxiter=30,  # Fewer iterations for stability
                method='lbfgs'  # More stable optimization
            )

            start_time = time.time()
            self.model.fit(train_data)
            training_time = time.time() - start_time

            # Validate model quality
            try:
                # Quick validation forecast
                val_forecast = self.model.predict(n_periods=24)
                val_actual = train_data.tail(24).values
                val_mase = self.calculate_mase(val_actual, val_forecast, train_data.iloc[:-24])

                if val_mase > 5.0:  # Very poor performance
                    print(f"Initial model validation failed (MASE: {val_mase:.3f}). Using fallback.")
                    return self._create_fallback_model(train_data)

            except:
                print("Model validation failed. Using fallback.")
                return self._create_fallback_model(train_data)

            self.is_fitted = True
            self.best_model = self.model
            self.best_mase = val_mase

            print(f"Initial training completed in {training_time:.2f}s (Val MASE: {val_mase:.3f})")

            # Initialize drift detection
            self._initialize_drift_detection(training_data)

            return True

        except Exception as e:
            print(f"Initial training failed: {e}")
            return self._create_fallback_model(training_data.tail(self.rolling_window))

    def _create_fallback_model(self, data):
        """Create a simple fallback model when SARIMA fails"""
        print("Creating fallback model...")
        try:
            # Use simpler model
            self.model = ARIMA(
                order=(1, 1, 0),
                seasonal_order=(1, 1, 0, 24),
                suppress_warnings=True,
                error_action='ignore',
                maxiter=20
            )
            self.model.fit(data)
            self.is_fitted = True
            self.best_model = self.model
            return True
        except:
            print("Fallback model also failed. Simulation cannot proceed.")
            return False

    def _initialize_drift_detection(self, training_data):
        """Initialize drift detection with better calibration"""
        try:
            # Use more conservative initialization
            self.z_scores = [1.0] * 168  # Start with 1 week of moderate z-scores
            self.ewma_z = 1.0
            self.recent_errors = [training_data.std()] * 24

            print("Drift detection initialized with conservative defaults")

        except Exception as e:
            print(f"Drift detection initialization failed: {e}")
            self.z_scores = [1.0] * 168
            self.ewma_z = 1.0

    def rolling_refit(self, recent_data):
        """Improved rolling refit with quality checks"""
        if len(recent_data) < self.rolling_window:
            print(f"Not enough data for refit. Need {self.rolling_window}, got {len(recent_data)}")
            return False

        try:
            print(f"Refitting SARIMA on {len(recent_data)} points...")

            # Split for validation
            train_data = recent_data.tail(self.rolling_window)
            val_data = train_data.tail(48)  # Last 2 days for validation
            train_subset = train_data.iloc[:-48]

            if len(train_subset) < self.rolling_window - 48:
                train_subset = train_data

            # Fit new model
            new_model = ARIMA(
                order=self.order,
                seasonal_order=self.seasonal_order,
                suppress_warnings=True,
                error_action='ignore',
                maxiter=30,
                method='lbfgs'
            )

            start_time = time.time()
            new_model.fit(train_subset)
            refit_time = time.time() - start_time

            # Validate new model quality
            try:
                val_forecast = new_model.predict(n_periods=24)
                val_actual = val_data.tail(24).values
                new_mase = self.calculate_mase(val_actual, val_forecast, train_subset)

                # Only update if new model is better or not much worse
                improvement_threshold = 0.1  # Allow 10% degradation
                current_performance = self.best_mase if self.best_mase < float('inf') else 5.0

                if new_mase <= current_performance * (1 + improvement_threshold):
                    self.model = new_model
                    if new_mase < self.best_mase:
                        self.best_model = new_model
                        self.best_mase = new_mase
                    print(f"Refit successful in {refit_time:.2f}s (MASE: {new_mase:.3f})")
                    return True
                else:
                    print(f"Refit rejected - performance degraded: {new_mase:.3f} vs {current_performance:.3f}")
                    return False

            except Exception as e:
                print(f"Refit validation failed: {e}")
                return False

        except Exception as e:
            print(f"Rolling refit failed: {e}")
            return False

    def forecast_next_24h(self, historical_data):
        """Robust forecasting with fallbacks"""
        if not self.is_fitted or self.model is None:
            return self._naive_forecast(historical_data)

        try:
            # Try main model first
            forecast, conf_int = self.model.predict(
                n_periods=24,
                return_conf_int=True,
                alpha=0.2
            )

            # Validate forecast
            if len(forecast) != 24 or np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                raise ValueError("Invalid forecast values")

            # Calculate prediction interval width
            if conf_int is not None and len(conf_int) == 24:
                pi_width = (conf_int[:, 1] - conf_int[:, 0]) / 2
                # Ensure reasonable PI width
                data_std = historical_data.tail(168).std()
                pi_width = np.clip(pi_width, data_std * 0.5, data_std * 3.0)
            else:
                pi_width = historical_data.tail(168).std() * np.ones(24)

            return forecast, pi_width

        except Exception as e:
            print(f"Forecasting failed: {e}. Using fallback.")
            return self._naive_forecast(historical_data)

    def _naive_forecast(self, historical_data):
        """Naive seasonal forecast as fallback"""
        last_week = historical_data.tail(168)  # Last week
        if len(last_week) >= 24:
            # Use same hour from previous days
            forecast = []
            for hour in range(24):
                same_hour_values = [last_week.iloc[i] for i in range(hour, len(last_week), 24)]
                forecast.append(np.mean(same_hour_values[-3:]))  # Average of last 3 same hours
            forecast = np.array(forecast)
        else:
            forecast = np.tile(historical_data.tail(24).mean(), 24)

        pi_width = historical_data.tail(168).std() * np.ones(24)
        return forecast, pi_width

    def calculate_mase(self, actuals, forecasts, historical_data):
        """Calculate MASE metric"""
        if len(historical_data) < 48 or len(actuals) != 24:
            return float('inf')

        # Seasonal naive errors (24-hour seasonality)
        if len(historical_data) > 24:
            naive_forecast = historical_data.shift(24)
            naive_errors = np.abs(historical_data - naive_forecast).dropna()
            if len(naive_errors) > 0:
                scale = np.mean(naive_errors)
            else:
                scale = historical_data.std()
        else:
            scale = historical_data.std()

        forecast_errors = np.abs(actuals - forecasts)
        return np.mean(forecast_errors) / (scale + 1e-8)

    def calculate_pi_coverage(self, actuals, forecasts, pi_width):
        """Calculate prediction interval coverage"""
        if len(actuals) != len(forecasts) or len(actuals) != len(pi_width):
            return 0.5  # Default coverage

        lower_bound = forecasts - pi_width
        upper_bound = forecasts + pi_width

        coverage = np.mean((actuals >= lower_bound) & (actuals <= upper_bound))
        return coverage

    def update_drift_detection(self, actuals, forecasts):
        """More conservative drift detection"""
        if len(actuals) != len(forecasts):
            return False

        errors = actuals - forecasts
        error_std = np.std(errors)

        if error_std < 1e-8:
            return False

        current_z_scores = errors / error_std
        current_abs_z = np.abs(current_z_scores)

        # Update z-score history
        self.z_scores.extend(current_abs_z)
        if len(self.z_scores) > self.drift_window:
            self.z_scores = self.z_scores[-self.drift_window:]

        # Conservative EWMA update
        if self.ewma_z is None:
            self.ewma_z = np.mean(current_abs_z)
        else:
            self.ewma_z = self.alpha * np.mean(current_abs_z) + (1 - self.alpha) * self.ewma_z

        # Update recent errors
        self.recent_errors.extend(np.abs(errors))
        if len(self.recent_errors) > self.error_window:
            self.recent_errors = self.recent_errors[-self.error_window:]

        # More conservative drift condition
        if len(self.z_scores) >= 168:  # Need at least 1 week of data
            # Use longer window for more stable threshold
            recent_z = self.z_scores[-min(720, len(self.z_scores)):]  # Last 30 days
            z_threshold = np.percentile(recent_z, 95)

            # Only trigger if significantly above threshold
            drift_margin = 1.1  # 10% margin
            drift_detected = self.ewma_z > z_threshold * drift_margin

            if drift_detected:
                print(f"🚨 DRIFT DETECTED! EWMA(|z|)={self.ewma_z:.3f} > {z_threshold:.3f}×{drift_margin}")

            return drift_detected

        return False

    def simulate_live_ingestion(self, start_hour, total_hours=2000):
        """Run live ingestion with better adaptation logic"""
        print(f"Starting improved live ingestion simulation for {self.country_code}...")

        # Get initial historical data
        initial_data = self.data.iloc[:start_hour]

        # Perform initial training
        success = self.initial_training(initial_data)
        if not success:
            print("Initial training failed. Cannot proceed with simulation.")
            return

        current_hour = start_hour
        adaptation_count = 0
        successful_adaptations = 0

        for hour in range(total_hours):
            if current_hour + 24 >= len(self.data):
                print(f"Reached end of data at hour {hour}")
                break

            current_timestamp = self.data.index[current_hour]
            historical_data = self.data.iloc[:current_hour + 1]

            # Check if it's 00:00 UTC for forecasting
            should_forecast = current_timestamp.hour == 0 and current_timestamp.minute == 0

            adaptation_reason = None
            adaptation_duration = 0
            adaptation_success = False

            if should_forecast:
                # Forecast next 24 hours
                forecasts, pi_width = self.forecast_next_24h(historical_data.iloc[:-1])
                actuals = self.data.iloc[current_hour + 1:current_hour + 25].values.flatten()

                if len(actuals) == len(forecasts):
                    # Update drift detection
                    drift_detected = self.update_drift_detection(actuals, forecasts)

                    # Calculate metrics before potential adaptation
                    mase_before = self.calculate_mase(actuals, forecasts, historical_data.iloc[:-1])
                    coverage_before = self.calculate_pi_coverage(actuals, forecasts, pi_width)

                    # Check if adaptation is needed
                    needs_adaptation = False

                    # Scheduled adaptation
                    days_since_adaptation = (hour - self.last_adaptation) // 24
                    if days_since_adaptation >= 1:
                        adaptation_reason = "scheduled"
                        needs_adaptation = True

                    # Drift-triggered adaptation
                    elif drift_detected:
                        adaptation_reason = "drift"
                        needs_adaptation = True

                    # Perform adaptation with quality check
                    if needs_adaptation and len(historical_data) >= self.rolling_window:
                        adaptation_start_time = time.time()
                        adaptation_success = self.rolling_refit(historical_data)
                        adaptation_duration = time.time() - adaptation_start_time

                        if adaptation_success:
                            self.last_adaptation = hour
                            adaptation_count += 1
                            successful_adaptations += 1

                            # Get metrics after adaptation
                            forecasts_after, pi_width_after = self.forecast_next_24h(historical_data.iloc[:-1])
                            mase_after = self.calculate_mase(actuals, forecasts_after, historical_data.iloc[:-1])
                            coverage_after = self.calculate_pi_coverage(actuals, forecasts_after, pi_width_after)

                            # Only log if adaptation was actually performed
                            self.performance_metrics.append({
                                'timestamp': current_timestamp,
                                'mase_before': mase_before,
                                'mase_after': mase_after,
                                'coverage_before': coverage_before,
                                'coverage_after': coverage_after,
                                'reason': adaptation_reason,
                                'improvement': mase_before - mase_after
                            })

                            print(f"Hour {hour}: {adaptation_reason} adaptation - "
                                  f"MASE: {mase_before:.3f}→{mase_after:.3f} "
                                  f"({'✓' if mase_after < mase_before else '✗'})")

                # Log the update
                self.log.append({
                    'timestamp': current_timestamp,
                    'strategy': 'rolling_sarima',
                    'reason': adaptation_reason if adaptation_success else 'no_adaptation',
                    'duration_s': adaptation_duration if adaptation_success else 0
                })

            current_hour += 1

            # Progress update
            if hour % 500 == 0:
                print(f"Progress: {hour}/{total_hours} hours completed")

        print(f"\nSimulation completed!")
        print(f"Adaptation attempts: {adaptation_count}")
        print(f"Successful adaptations: {successful_adaptations}")
        print(f"Success rate: {successful_adaptations / max(1, adaptation_count) * 100:.1f}%")

    def save_results(self):
        """Save logs and results"""
        os.makedirs('../outputs', exist_ok=True)

        # Save update log
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(f'../outputs/{self.country_code}_online_updates.csv', index=False)

        # Save performance metrics
        if self.performance_metrics:
            perf_df = pd.DataFrame(self.performance_metrics)
            perf_df.to_csv(f'../outputs/{self.country_code}_performance_metrics.csv', index=False)

        return log_df


def run_improved_live_simulation():
    """Run the improved live simulation"""
    online_country = 'FR'

    print(f"Selected {online_country} for live simulation")

    # Load data
    country_data = load_country_data(online_country)
    if country_data is None:
        return

    print(f"Loaded data for {online_country}")
    print(f"Data shape: {country_data.shape}")

    # Check data sufficiency
    required_hours = 120 * 24 + 2000
    if len(country_data) < required_hours:
        simulation_hours = min(2000, len(country_data) - 120 * 24)
        if simulation_hours <= 0:
            print("Not enough data for simulation.")
            return
    else:
        simulation_hours = 2000

    print(f"Simulation hours: {simulation_hours}")

    # Start simulation
    start_hour = 120 * 24
    adapter = ImprovedLiveSARIMAAdapter(
        country_data['load_actual_entsoe_transparency'],
        online_country,
        history_days=120
    )
    adapter.simulate_live_ingestion(start_hour, total_hours=simulation_hours)

    # Save and report results
    log_df = adapter.save_results()

    print("\n" + "=" * 50)
    print("IMPROVED SIMULATION SUMMARY")
    print("=" * 50)

    if adapter.performance_metrics:
        perf_df = pd.DataFrame(adapter.performance_metrics)

        # Calculate improvements only for successful adaptations
        improvements = perf_df[perf_df['improvement'] != 0]
        if len(improvements) > 0:
            avg_mase_improvement = improvements['improvement'].mean()
            positive_improvements = len(improvements[improvements['improvement'] > 0])
            improvement_rate = positive_improvements / len(improvements) * 100

            print(f"Successful adaptations: {len(improvements)}")
            print(f"Average MASE change: {avg_mase_improvement:+.4f}")
            print(f"Improvement rate: {improvement_rate:.1f}%")
            print(f"Final average coverage: {improvements['coverage_after'].mean():.3f}")

    return adapter


# Run the improved simulation
if __name__ == "__main__":
    print("Improved Live Ingestion + Online Adaptation")
    print("Strategy: Quality-Checked Rolling SARIMA Refit")
    print("=" * 50)

    adapter = run_improved_live_simulation()