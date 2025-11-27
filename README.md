# OPSD PowerDesk - Time Series Forecasting & Anomaly Detection

## Overview

OPSD PowerDesk is a comprehensive time series forecasting and anomaly detection system for European power load data (Austria, Switzerland, France). It combines SARIMA modeling, online adaptation with drift detection, and machine learning-based anomaly classification.

**Key Features:**
- ⚡ **Hourly Load Forecasting**: SARIMA models with 24-hour ahead predictions
- 🔍 **Anomaly Detection**: Statistical (z-score, CUSUM) + ML-based detection
- 🔄 **Online Adaptation**: Real-time model refit with drift detection
- 📊 **Live Dashboard**: Streamlit-based monitoring interface
- 📈 **Comprehensive Metrics**: MASE, sMAPE, PI coverage, and more

---

## Project Structure

```
ATA Project/
├── src/                          # Source code
│   ├── load_opsd.py             # Data loading and preprocessing
│   ├── forecast.py              # SARIMA forecasting (dev/test)
│   ├── anamoly.py               # Statistical anomaly detection
│   ├── anamoly_ml.py            # ML-based anomaly classification
│   ├── decompose_acf_pacf.py    # ACF/PACF analysis and SARIMA parameter tuning
│   ├── metrics.py               # Evaluation metrics (MASE, sMAPE, etc.)
│   ├── live_loop.py             # Step 4: Live ingestion + online adaptation
│   └── dashboard_app.py         # Step 5: Streamlit dashboard
│
├── data/                         # Input data
│   ├── time_series_60min_singleindex_filtered.csv
│   ├── AT_preprocessed_data.csv # Austria
│   ├── CH_preprocessed_data.csv # Switzerland
│   └── FR_preprocessed_data.csv # France
│
├── outputs/                      # Output files (forecasts, anomalies, metrics)
│   ├── {CC}_forecast_dev.csv
│   ├── {CC}_forecast_test.csv
│   ├── {CC}_anomalies.csv
│   ├── {CC}_online_updates.csv
│   └── ... (metrics, plots, etc.)
│
├── notebooks/                    # Jupyter notebooks
│   └── gru_lstm_lstm-attention.ipynb
│
├── config.yaml                   # Configuration file (countries, thresholds, etc.)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Countries & Data

The system processes data for **three European countries**:

| Country | Code | Data Column |
|---------|------|---|
| Austria | AT | `load_actual_entsoe_transparency` |
| Switzerland | CH | `load_actual_entsoe_transparency` |
| France | FR | `load_actual_entsoe_transparency` |

Data granularity: **Hourly (60-minute intervals)**

---

## Environment Setup

### Prerequisites
- Python 3.8+
- macOS/Linux/Windows
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "/Users/venkatanagasaisrujantallam/PyCharm Projects/ATA Project"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # OR on Windows:
   # .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, pmdarima, streamlit; print('✓ All dependencies installed')"
   ```

---

## How to Run

### Step 1: Data Preprocessing

Preprocess raw data and split into separate country files:

```bash
cd src
python -c "from load_opsd import preprocess_data; preprocess_data()"
```

This creates:
- `AT_preprocessed_data.csv`
- `CH_preprocessed_data.csv`
- `FR_preprocessed_data.csv`

### Step 2: ACF/PACF Analysis & SARIMA Parameter Tuning

Analyze time series properties and determine optimal SARIMA parameters:

```bash
python -c "from decompose_acf_pacf import get_model_params; get_model_params(country_code='AT')"
```

Outputs:
- ACF/PACF plots
- STL decomposition plots
- Suggested SARIMA parameters

Repeat for other countries (CH, FR).

### Step 3: Backtesting & Forecasting

Generate forecasts on dev/test sets with 80% prediction intervals:

```bash
python -c "
from forecast import backtesting_and_forecasting
backtesting_and_forecasting(
    country_codes=['AT', 'CH', 'FR'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 2, 24),
    horizon=24,
    PI=0.8
)
"
```

Outputs:
- `{CC}_forecast_dev.csv` (10% validation set)
- `{CC}_forecast_test.csv` (10% test set)

### Step 4: Anomaly Detection (Statistical)

Detect anomalies using z-score residuals and CUSUM:

```bash
python -c "
from anamoly import run_anomaly_part1
for cc in ['AT', 'CH', 'FR']:
    run_anomaly_part1(
        forecast_csv_path=f'../outputs/{cc}_forecast_test.csv',
        output_path=f'../outputs/{cc}_anomalies.csv'
    )
"
```

Outputs:
- `{CC}_anomalies.csv` with `flag_z` and `flag_cusum` columns

### Step 5: Anomaly Detection (ML-based)

Train machine learning classifier on verified anomaly labels:

```bash
python -c "
from anamoly_ml import make_silver_labels, build_features, train_anomaly_classifier
# (Requires human-verified labels)
"
```

### Step 6: Live Ingestion & Online Adaptation

Simulate live data ingestion with online model refit and drift detection:

```bash
python live_loop.py
```

This:
- Loads historical data (80/10/10 split)
- Simulates 2000 hours of live data
- Detects drift and triggers model refits
- Logs all updates to `{CC}_online_updates.csv`
- Outputs forecasts for all countries

### Step 7: Streamlit Dashboard

Launch the live monitoring dashboard:

```bash
streamlit run dashboard_app.py
```

Dashboard features:
- Country selector (AT, CH, FR)
- Live series (last 7-14 days)
- 24-hour forecast with 80% PI band
- Anomaly highlighting
- KPI tiles (MASE, PI coverage, anomaly count)
- Update status and adaptation logs

---

## Configuration

All configuration parameters are in **`config.yaml`**:

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|---|
| `countries` | [AT, CH, FR] | Countries to process |
| `load_column` | `load_actual_entsoe_transparency` | Data column name |
| `seasonality` | 24 | Hours per day |
| `forecast_horizon` | 24 | Hours ahead |
| `pi_level` | 0.8 | 80% prediction interval |
| `train_ratio` | 0.8 | 80% train data |
| `order` | (1,1,1) | SARIMA (p,d,q) |
| `seasonal_order` | (1,1,2,24) | SARIMA (P,D,Q,m) |
| `adaptation_window_days` | 90 | Days for refit |
| `zscore_threshold` | 3.0 | Anomaly threshold |
| `drift_threshold_multiplier` | 1.2 | Drift sensitivity |

To modify, edit `config.yaml` and the system will use updated values.

---

## Main Workflows

### Workflow A: Initial Setup (One-time)
```
Data Preprocessing → ACF/PACF Analysis → Backtesting & Forecasting → Statistical Anomaly Detection
```

### Workflow B: Live Monitoring (Continuous)
```
Load Historical Data → Live Ingestion Loop → Drift Detection → Model Refit → Dashboard
```

### Workflow C: Full Pipeline (Complete Analysis)
```
A + B + ML Anomaly Classification
```

---

## Output Files

### Forecast Files
- `{CC}_forecast_dev.csv`: Validation forecasts
- `{CC}_forecast_test.csv`: Test forecasts with actual values

### Anomaly Files
- `{CC}_anomalies.csv`: Statistical anomalies (z-score, CUSUM flags)
- `{CC}_anomaly_labels_verified.csv`: Human-verified labels
- `{CC}_anomaly_silver_labels.csv`: Auto-generated ML training labels

### Metrics Files
- `{CC}_rolling7d_metrics.csv`: Rolling 7-day metrics
- `{CC}_anomaly_ml_eval.json`: ML model evaluation metrics

### Live Ingestion Files
- `{CC}_online_updates.csv`: Drift detections and model refits

### Plots
- `{CC}_acf_plot.png`, `{CC}_pacf_plot.png`
- `{CC}_seasonal_acf_plot.png`, `{CC}_seasonal_pacf_plot.png`
- `{CC}_stl_decomposition_for_past_1000_steps.png`

---

## Key Thresholds & Horizons

### Forecasting
- **Horizon**: 24 hours ahead
- **PI Level**: 80%
- **Seasonality**: 24 (hourly data)

### Anomaly Detection
- **Z-score Threshold**: 3.0 (statistical)
- **Z-score Window**: 336 hours (14 days)
- **CUSUM Parameters**: k=0.5, h=5.0

### Drift Detection
- **EWMA Alpha**: 0.1
- **Drift Window**: 720 hours (30 days)
- **Threshold Multiplier**: 1.2 (adaptive)

### Online Adaptation
- **Refit Frequency**: 24 hours
- **Adaptation Window**: 90 days
- **Strategy**: Rolling SARIMA

---

## Dependencies

Core libraries:
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML classification
- **statsmodels**: Statistical modeling
- **pmdarima**: Auto ARIMA
- **matplotlib, seaborn**: Plotting
- **streamlit**: Web dashboard
- **tensorflow, torch**: Neural networks
- **lightgbm**: Gradient boosting

See `requirements.txt` for full list with versions.

---

## Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Data file not found"
**Solution**: Verify data files exist in `./data/`:
```bash
ls -la data/
```

### Issue: Dashboard not loading
**Solution**: Run from project root, not src directory:
```bash
cd "/Users/venkatanagasaisrujantallam/PyCharm Projects/ATA Project"
streamlit run src/dashboard_app.py
```

### Issue: Low forecast accuracy
**Solution**: Check SARIMA parameters in `config.yaml`. Run ACF/PACF analysis:
```bash
python src/decompose_acf_pacf.py
```

---

## Performance Metrics

Expected metrics on test set (80% PI):

| Metric | Austria | Switzerland | France |
|--------|---------|-------------|--------|
| MASE | < 0.5 | < 0.5 | < 0.5 |
| sMAPE | 2-5% | 2-5% | 2-5% |
| PI Coverage | 75-85% | 75-85% | 75-85% |

---

## Citation & References

- **OPSD Dataset**: Open Power System Data (https://open-power-system-data.org/)
- **SARIMA**: Time Series forecasting using AutoRegressive Integrated Moving Average
- **Drift Detection**: Exponential Weighted Moving Average (EWMA) for concept drift
- **Anomaly Detection**: Tukey's fences (3σ) + CUSUM control chart + ML classification

---

## License

This project is part of the ATA (Advanced Time Series Analysis) curriculum project.

---

## Contact & Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

**Last Updated**: November 2025
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
scipy==1.16.3
statsmodels==0.14.5
pmdarima==2.1.1
matplotlib==3.10.7
seaborn==0.13.2
streamlit==1.51.0
tensorflow==2.20.0
torch==2.9.1
lightgbm==4.6.0
python-dateutil==2.9.0.post0
pytz==2025.2
pdfplumber==0.11.8
joblib==1.5.2

