"""
Step 5: Streamlit Dashboard for Live Monitoring & Dashboarding

This dashboard displays real-time forecasting, anomaly detection, and online
adaptation metrics for the selected country across three European regions
(AT, CH, FR).

Required elements:
i. Country selector (preselect live country)
ii. Live series: last 7–14 days of y_true & yhat (line chart)
iii. Forecast cone: next 24h mean with 80% PI (shaded)
iv. Anomaly tape: highlight hours with flag_z=1
v. KPI tiles: rolling-7d MASE, PI coverage, anomaly hours today, last update
vi. Update status: last online update timestamp + reason
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import os

# Set page configuration
st.set_page_config(
    page_title="OPSD PowerDesk - Live Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

@st.cache_data
def load_forecast_data(country_code):
    """Load forecast test data."""
    path = os.path.join(OUTPUTS_DIR, f"{country_code}_forecast_test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_anomalies_data(country_code):
    """Load anomalies data."""
    path = os.path.join(OUTPUTS_DIR, f"{country_code}_anomalies.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_online_updates(country_code):
    """Load online update logs."""
    path = os.path.join(OUTPUTS_DIR, f"{country_code}_online_updates.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_rolling_metrics(country_code):
    """Load rolling 7-day metrics."""
    path = os.path.join(OUTPUTS_DIR, f"{country_code}_rolling7d_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ============================================================================
# DASHBOARD HEADER
# ============================================================================

st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>⚡ OPSD PowerDesk</h1>
        <h3>Live Forecasting, Anomaly Detection & Online Adaptation</h3>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - COUNTRY SELECTOR
# ============================================================================

st.sidebar.markdown("### ⚙️ Configuration")
country_code = st.sidebar.selectbox(
    "Select Country",
    options=["FR", "AT", "CH"],
    index=0,
    help="Choose a country to monitor. FR (France) is preselected."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.write(f"**Country:** {country_code}")
st.sidebar.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Debug Info")
st.sidebar.write(f"**Data Dir:** `{OUTPUTS_DIR}`")
forecast_exists = os.path.exists(os.path.join(OUTPUTS_DIR, f"{country_code}_forecast_test.csv"))
st.sidebar.write(f"**Forecast File:** {'✅ Found' if forecast_exists else '❌ Not Found'}")

# ============================================================================
# LOAD DATA
# ============================================================================

forecast_df = load_forecast_data(country_code)
anomalies_df = load_anomalies_data(country_code)
online_updates_df = load_online_updates(country_code)
rolling_metrics_df = load_rolling_metrics(country_code)

# Check if data exists
if forecast_df is None:
    st.error(f"❌ No forecast data found for {country_code}")
    st.stop()

# Convert timestamp columns to datetime
if 'timestamp' in forecast_df.columns:
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])

if anomalies_df is not None and 'timestamp' in anomalies_df.columns:
    anomalies_df['timestamp'] = pd.to_datetime(anomalies_df['timestamp'])

if online_updates_df is not None and 'timestamp' in online_updates_df.columns:
    online_updates_df['timestamp'] = pd.to_datetime(online_updates_df['timestamp'])

if rolling_metrics_df is not None and 'timestamp' in rolling_metrics_df.columns:
    rolling_metrics_df['timestamp'] = pd.to_datetime(rolling_metrics_df['timestamp'])

# ============================================================================
# SECTION 1: KPI TILES
# ============================================================================

st.markdown("## 📈 Key Performance Indicators (Last 7 Days)")

# Calculate metrics
latest_7d_data = forecast_df.tail(7 * 24)  # Last 7 days (168 hours)

if len(latest_7d_data) > 0:
    # MASE calculation
    y_true = latest_7d_data['y_true'].values
    yhat = latest_7d_data['yhat'].values

    if len(y_true) > 24:
        errors = np.abs(y_true - yhat)
        denom = np.abs(y_true[24:] - y_true[:-24]).mean()
        mase_7d = errors.mean() / (denom + 1e-8)
    else:
        mase_7d = np.nan

    # PI Coverage
    if 'lo' in latest_7d_data.columns and 'hi' in latest_7d_data.columns:
        lo = latest_7d_data['lo'].values
        hi = latest_7d_data['hi'].values
        pi_coverage_7d = np.mean((y_true >= lo) & (y_true <= hi))
    else:
        pi_coverage_7d = np.nan

    # Anomaly count for last day of available data
    if anomalies_df is not None and len(anomalies_df) > 0:
        last_day = anomalies_df['timestamp'].dt.normalize().max()
        anomalies_today = anomalies_df[
            anomalies_df['timestamp'].dt.normalize() == last_day
        ]['flag_z'].sum() if 'flag_z' in anomalies_df.columns else 0
    else:
        anomalies_today = 0

    # Last update
    if online_updates_df is not None and len(online_updates_df) > 0:
        last_update = online_updates_df.iloc[-1]
        last_update_time = last_update['timestamp']
        last_update_reason = last_update['reason']
    else:
        last_update_time = "Never"
        last_update_reason = "N/A"

    # Display KPI tiles
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 MASE (7d)",
            value=f"{mase_7d:.3f}" if not np.isnan(mase_7d) else "N/A",
            help="Mean Absolute Scaled Error - lower is better"
        )

    with col2:
        st.metric(
            label="🎯 PI Coverage (7d)",
            value=f"{pi_coverage_7d:.1%}" if not np.isnan(pi_coverage_7d) else "N/A",
            help="80% Prediction Interval Coverage - target is 80%"
        )

    with col3:
        st.metric(
            label="⚠️ Anomalies Today",
            value=f"{int(anomalies_today)}",
            help="Number of anomalous hours detected today"
        )

    with col4:
        st.metric(
            label="🔄 Last Update",
            value=str(last_update_time)[:10] if last_update_time != "Never" else "Never",
            help=f"Last adaptation: {last_update_reason}"
        )

# ============================================================================
# SECTION 2: LIVE SERIES (Last 7-14 Days) - COMBINED WITH ANOMALIES & PI
# ============================================================================

st.markdown("## 📉 Live Load Series with Anomalies & 80% Prediction Interval (Last 14 Days)")

# Get last 14 days of data
last_14d_data = forecast_df.tail(14 * 24)

if len(last_14d_data) > 0:
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot prediction intervals as shaded area first (background)
    if 'lo' in last_14d_data.columns and 'hi' in last_14d_data.columns:
        ax.fill_between(
            last_14d_data['timestamp'],
            last_14d_data['lo'],
            last_14d_data['hi'],
            alpha=0.2,
            color='#3498db',
            label='80% Prediction Interval',
            zorder=1
        )

    # Plot forecast (orange dashed line)
    ax.plot(
        last_14d_data['timestamp'],
        last_14d_data['yhat'],
        label='Forecast',
        color='#e67e22',
        linewidth=2.5,
        linestyle='--',
        marker='s',
        markersize=2.5,
        zorder=2
    )

    # Plot actual load (blue solid line)
    ax.plot(
        last_14d_data['timestamp'],
        last_14d_data['y_true'],
        label='Actual Load',
        color='#2980b9',
        linewidth=2.5,
        marker='o',
        markersize=2.5,
        zorder=3
    )

    # Highlight anomalies on the plot
    if anomalies_df is not None and 'flag_z' in anomalies_df.columns:
        # Filter anomalies for last 14 days
        last_14d_min = last_14d_data['timestamp'].min()
        last_14d_max = last_14d_data['timestamp'].max()

        anomalies_14d = anomalies_df[
            (anomalies_df['timestamp'] >= last_14d_min) &
            (anomalies_df['timestamp'] <= last_14d_max) &
            (anomalies_df['flag_z'] == 1)
        ]

        if len(anomalies_14d) > 0:
            # Anomalies already contain y_true values, use them directly
            ax.scatter(
                anomalies_14d['timestamp'],
                anomalies_14d['y_true'],
                color='#e74c3c',
                s=200,
                marker='X',
                label='Anomalies Detected (Z-Score)',
                zorder=5,
                edgecolors='darkred',
                linewidths=2.5,
                alpha=0.9
            )

    ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Load (MW)', fontsize=12, fontweight='bold')
    ax.set_title(f'{country_code} - Live Load with Forecast, 80% PI & Anomalies (Last 14 Days)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


# ============================================================================
# SECTION 4: ANOMALY STATISTICS (Last 7 Days)
# ============================================================================

st.markdown("## 🚨 Anomaly Detection Summary (Last 7 Days)")

if anomalies_df is not None:
    # Get the last 7 days (168 hours) of anomaly data
    anomalies_7d = anomalies_df.tail(7 * 24).copy()

    if len(anomalies_7d) > 0 and (anomalies_7d['flag_z'] == 1).sum() > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            total_anomalies = (anomalies_7d['flag_z'] == 1).sum()
            st.metric("Total Anomalies", f"{int(total_anomalies)}", delta="in last 7 days", help="Z-score anomalies detected")

        with col2:
            avg_z_score = anomalies_7d['z_resid'].abs().mean()
            st.metric("Avg |Z-Score|", f"{avg_z_score:.3f}", delta="residual magnitude")

        with col3:
            if 'flag_cusum' in anomalies_7d.columns:
                cusum_count = (anomalies_7d['flag_cusum'] == 1).sum()
                st.metric("CUSUM Flags", f"{int(cusum_count)}", delta="in last 7 days")
    else:
        st.info("✅ No anomalies detected in the last 7 days of data")
else:
    st.warning("Anomaly data not available")

# ============================================================================
# SECTION 5: ONLINE UPDATES STATUS
# ============================================================================

st.markdown("## 🔄 Online Ingestion & Adaptation Status")

if online_updates_df is not None and len(online_updates_df) > 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        total_adaptations = len(online_updates_df)
        st.metric("Total Adaptations", f"{total_adaptations}", help="Total model updates during live ingestion")

    with col2:
        avg_duration = online_updates_df['duration_s'].mean()
        st.metric("Avg Duration", f"{avg_duration:.3f}s", help="Average time per adaptation")

    with col3:
        drift_triggers = (online_updates_df['reason'] == 'drift').sum()
        st.metric("Drift Triggers", f"{int(drift_triggers)}", help="Adaptations triggered by drift detection")

    st.markdown("---")

    # Latest updates table
    st.subheader("📋 Recent Adaptation Events (Last 15)")
    latest_updates = online_updates_df.tail(15)[['timestamp', 'strategy', 'reason', 'duration_s']].copy()
    latest_updates.columns = ['Timestamp', 'Strategy', 'Reason', 'Duration (s)']
    latest_updates['Duration (s)'] = latest_updates['Duration (s)'].round(3)
    st.dataframe(latest_updates, use_container_width=True, hide_index=True)
else:
    st.info("No online adaptation events recorded yet")



# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem; margin-top: 2rem;'>
        <p>OPSD PowerDesk • Live Forecasting & Anomaly Detection Dashboard</p>
        <p>Data refreshes automatically when new predictions are available</p>
    </div>
""", unsafe_allow_html=True)

