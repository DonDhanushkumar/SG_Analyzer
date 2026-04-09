import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ================================
# SAFE ML IMPORT
# ================================
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(layout="wide")

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=200)
with col2:
    st.markdown("<h1 style='color:#2E86C1;'>🚀 Saint-Gobain AI Energy Dashboard</h1>", unsafe_allow_html=True)

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

        # Detect total automatically
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
    # ⏰ TIME CATEGORY ANALYSIS
    # ================================
    st.subheader("⏰ Time Category Energy Analysis")

    time_summary = (
        total_df.groupby('Time_Category')['Energy']
        .sum()
        .reset_index()
    )

    order = ["Morning Peak", "Evening Peak", "Morning Non-Peak", "Evening Non-Peak"]
    time_summary['Time_Category'] = pd.Categorical(time_summary['Time_Category'], categories=order, ordered=True)
    time_summary = time_summary.sort_values('Time_Category')

    # BAR CHART
    fig_time = px.bar(
        time_summary,
        x='Time_Category',
        y='Energy',
        color='Time_Category',
        title="Energy Consumption by Time Category"
    )

    st.plotly_chart(fig_time, use_container_width=True)

    # PERCENTAGE
    total_val = time_summary['Energy'].sum()
    time_summary['Percentage'] = (time_summary['Energy'] / total_val) * 100

    fig_percent = px.bar(
        time_summary,
        x='Time_Category',
        y='Percentage',
        color='Time_Category',
        text=time_summary['Percentage'].round(2).astype(str) + "%",
        title="Percentage Contribution"
    )

    fig_percent.update_traces(textposition='outside')

    st.plotly_chart(fig_percent, use_container_width=True)

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
        lambda x: "Main" if x.get('Peak',0) > x.get('Non-Peak',0) else "Optional",
        axis=1
    )

    st.subheader("🤖 AI Classification")
    st.dataframe(pivot_df)

    # ================================
    # 🔮 PREDICTION
    # ================================
    st.subheader("🔮 Prediction")

    ts_df = total_df.groupby('Datetime')['Energy'].sum().reset_index().sort_values('Datetime')
    ts_df['t'] = np.arange(len(ts_df))

    future_t = np.arange(len(ts_df), len(ts_df)+24).reshape(-1,1)

    if ML_AVAILABLE:
        X = ts_df[['t']]
        y = ts_df['Energy']

        lin = LinearRegression().fit(X,y)
        lin_pred = lin.predict(future_t)

        gbr = GradientBoostingRegressor().fit(X,y)
        gbr_pred = gbr.predict(future_t)
    else:
        lin_pred = np.repeat(ts_df['Energy'].mean(), 24)
        gbr_pred = lin_pred

    last_date = ts_df['Datetime'].dropna().max()

    if pd.isna(last_date):
        st.error("❌ Datetime issue")
        st.stop()

    future_dates = pd.date_range(start=last_date, periods=24, freq='h')

    df_plot = pd.DataFrame({
        'Datetime': list(ts_df['Datetime']) + list(future_dates)*2,
        'Value': list(ts_df['Energy']) + list(lin_pred) + list(gbr_pred),
        'Model': ['Actual']*len(ts_df) + ['Linear']*24 + ['XGBoost']*24
    })

    st.plotly_chart(px.line(df_plot, x='Datetime', y='Value', color='Model'),
                    use_container_width=True)

    # ================================
    # 🎯 SYSTEM ANALYSIS
    # ================================
    st.subheader("🎯 System Peak vs Non-Peak Comparison")

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

    system_peak = (
        instrument_df.groupby(['System', 'Peak_Type'])['Energy']
        .sum()
        .reset_index()
    )

    pivot_sys = system_peak.pivot(index='System', columns='Peak_Type', values='Energy').fillna(0)

    if 'Peak' not in pivot_sys.columns:
        pivot_sys['Peak'] = 0
    if 'Non-Peak' not in pivot_sys.columns:
        pivot_sys['Non-Peak'] = 0

    pivot_sys = pivot_sys.reset_index()

    plot_sys = pivot_sys.melt(
        id_vars='System',
        value_vars=['Peak', 'Non-Peak'],
        var_name='Type',
        value_name='Energy'
    )

    fig_sys = px.bar(
        plot_sys,
        x='System',
        y='Energy',
        color='Type',
        barmode='group',
        title="System-wise Peak vs Non-Peak"
    )

    st.plotly_chart(fig_sys, use_container_width=True)

    st.success("🚀 Dashboard Running Successfully!")
