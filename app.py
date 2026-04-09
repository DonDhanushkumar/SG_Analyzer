import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(layout="wide", page_title="AI Energy Dashboard")

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=80)
with col2:
    st.markdown("<h1 style='color:#2E86C1;'>🚀 AI + ML Energy Optimization Dashboard</h1>", unsafe_allow_html=True)

# ================================
# TIME FUNCTION
# ================================
def classify_time(hour):
    if 6 <= hour <= 9:
        return "Morning Peak"
    elif 18 <= hour <= 21:
        return "Evening Peak"
    elif hour == 5 or (10 <= hour <= 17):
        return "Morning Non-Peak"
    else:
        return "Evening Non-Peak"

# ================================
# FILE UPLOAD
# ================================
files = st.file_uploader("Upload Files", type=["csv","xlsx"], accept_multiple_files=True)

if files:

    all_data, total_list, instrument_list = [], [], []

    for file in files:
        name = file.name

        df = pd.read_csv(file) if name.endswith('.csv') else pd.read_excel(file)
        df = df.fillna(0)

        if 'Instrument' not in df.columns:
            st.warning(f"{name} skipped")
            continue

        df = df.melt(id_vars=['Instrument'], var_name='Datetime', value_name='Energy')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        df['Hour'] = df['Datetime'].dt.hour
        df['Time_Category'] = df['Hour'].apply(classify_time)
        df['Plant'] = name

        all_data.append(df)

        # TOTAL detection
        total_candidate = df.groupby('Instrument')['Energy'].sum().idxmax()
        total_list.append(df[df['Instrument'] == total_candidate])
        instrument_list.append(df[df['Instrument'] != total_candidate])

    df_all = pd.concat(all_data, ignore_index=True)
    total_df = pd.concat(total_list, ignore_index=True)
    instrument_df = pd.concat(instrument_list, ignore_index=True)

    # ================================
    # KPI
    # ================================
    st.subheader("📊 KPI Dashboard")

    total_energy = total_df['Energy'].sum()
    peak_avg = total_df[total_df['Time_Category'].str.contains("Peak")]['Energy'].mean()
    non_peak_avg = total_df[total_df['Time_Category'].str.contains("Non-Peak")]['Energy'].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Total Energy", round(total_energy,2))
    c2.metric("🔥 Avg Peak", round(peak_avg,2))
    c3.metric("🌙 Avg Non-Peak", round(non_peak_avg,2))

    # ================================
    # TIME DISTRIBUTION
    # ================================
    st.subheader("🥧 Energy Distribution")

    time_summary = total_df.groupby(['Plant','Time_Category'])['Energy'].sum().reset_index()

    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy', facet_col='Plant'),
                    use_container_width=True)

    st.plotly_chart(px.bar(time_summary, x='Time_Category', y='Energy', color='Plant'),
                    use_container_width=True)

    # ================================
    # AI CLASSIFICATION
    # ================================
    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    peak_data = instrument_df.groupby(['Instrument','Peak_Type'])['Energy'].sum().reset_index()

    pivot_df = peak_data.pivot(index='Instrument', columns='Peak_Type', values='Energy').fillna(0)

    if 'Peak' not in pivot_df.columns:
        pivot_df['Peak'] = 0
    if 'Non-Peak' not in pivot_df.columns:
        pivot_df['Non-Peak'] = 0

    pivot_df = pivot_df.reset_index()

    pivot_df['Category'] = pivot_df.apply(
        lambda x: "Main" if x['Peak'] > x['Non-Peak'] else "Optional",
        axis=1
    )

    st.subheader("🤖 AI Classification")
    st.dataframe(pivot_df)

    # ================================
    # 🔮 ADVANCED ML PREDICTION
    # ================================
    st.subheader("🔮 Advanced ML Prediction")

    ts_df = total_df.groupby('Datetime')['Energy'].sum().reset_index().sort_values('Datetime')

    ts_df['t'] = np.arange(len(ts_df))

    X = ts_df[['t']]
    y = ts_df['Energy']

    # Linear
    lin = LinearRegression().fit(X,y)
    future_t = np.arange(len(ts_df), len(ts_df)+24).reshape(-1,1)
    lin_pred = lin.predict(future_t)

    # XGBoost-style (GBR)
    gbr = GradientBoostingRegressor().fit(X,y)
    gbr_pred = gbr.predict(future_t)

    # LSTM-style (lag model)
    lag = 5
    lag_df = ts_df.copy()

    for i in range(1, lag+1):
        lag_df[f'lag_{i}'] = lag_df['Energy'].shift(i)

    lag_df = lag_df.dropna()

    X_lag = lag_df[[f'lag_{i}' for i in range(1, lag+1)]]
    y_lag = lag_df['Energy']

    lag_model = LinearRegression().fit(X_lag, y_lag)

    temp = list(lag_df['Energy'].values[-lag:])
    lstm_preds = []

    for _ in range(24):
        pred = lag_model.predict([temp[-lag:]])[0]
        lstm_preds.append(pred)
        temp.append(pred)

    future_dates = pd.date_range(start=ts_df['Datetime'].max(), periods=24, freq='H')

    df_plot = pd.DataFrame({
        'Datetime': list(ts_df['Datetime']) + list(future_dates)*3,
        'Value': list(ts_df['Energy']) + list(lin_pred) + list(gbr_pred) + list(lstm_preds),
        'Model': ['Actual']*len(ts_df) + ['Linear']*24 + ['XGBoost']*24 + ['LSTM']*24
    })

    st.plotly_chart(px.line(df_plot, x='Datetime', y='Value', color='Model'),
                    use_container_width=True)

    # ================================
    # SYSTEM ANALYSIS
    # ================================
    st.subheader("🎯 System Analysis")

    def map_system(name):
        name = str(name).lower()
        if "light" in name or "ac" in name:
            return "Light & AC"
        elif "mbt-a" in name:
            return "MBT-A"
        elif "mbt-b" in name:
            return "MBT-B"
        elif "process air" in name:
            return "Process Air"
        elif "x7" in name:
            return "X7 Bay"
        elif "stp" in name:
            return "STP"
        elif "cooling" in name or "bus b" in name:
            return "Cooling Circuit Bus B"
        else:
            return "Others"

    instrument_df['System'] = instrument_df['Instrument'].apply(map_system)

    system_summary = instrument_df.groupby('System')['Energy'].sum().reset_index()

    st.plotly_chart(px.bar(system_summary, x='System', y='Energy', color='System'),
                    use_container_width=True)

    # ================================
    # AI INSIGHTS
    # ================================
    st.subheader("🤖 AI Insights")

    top_peak = pivot_df.sort_values(by='Peak', ascending=False).head(5)

    for _, row in top_peak.iterrows():
        st.write(f"⚠️ {row['Instrument']} → High peak consumption")

    st.success("🚀 AI + ML System Ready!")
