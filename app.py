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
st.set_page_config(layout="wide", page_title="AI Energy Dashboard")

# ================================
# CSS (BIG TABLE)
# ================================
st.markdown("""
<style>
.stDataFrame {font-size:16px;}
.stDataFrame th {font-size:18px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=200)
with col2:
    st.markdown("<h1 style='color:#2E86C1;'>🚀 Saint-Gobain AI Energy Optimization Dashboard</h1>", unsafe_allow_html=True)

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
            continue

        df = df.melt(id_vars=['Instrument'], var_name='Datetime', value_name='Energy')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        df['Hour'] = df['Datetime'].dt.hour
        df['Date'] = df['Datetime'].dt.date
        df['Time_Category'] = df['Hour'].apply(classify_time)
        df['Plant'] = name

        all_data.append(df)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Total Energy", round(total_df['Energy'].sum(),2))
    c2.metric("🔥 Avg Peak", round(total_df[total_df['Time_Category'].str.contains("Peak")]['Energy'].mean(),2))
    c3.metric("🌙 Avg Non-Peak", round(total_df[total_df['Time_Category'].str.contains("Non-Peak")]['Energy'].mean(),2))

    # ================================
    # ENERGY DISTRIBUTION
    # ================================
    st.subheader("🥧 Energy Distribution")

    time_summary = total_df.groupby(['Plant','Time_Category'])['Energy'].sum().reset_index()

    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy', facet_col='Plant'),
                    use_container_width=True)

    st.plotly_chart(px.bar(time_summary, x='Time_Category', y='Energy', color='Plant'),
                    use_container_width=True)

    # ================================
    # DATE FILTER
    # ================================
    st.subheader("🎛️ Select Date")
    selected_date = st.selectbox("Choose Date", sorted(df_all['Date'].unique()))
    day_df = df_all[df_all['Date'] == selected_date]

    # ================================
    # HOURLY GRAPH
    # ================================
    st.subheader("📊 Hourly Energy by Time Category")

    hourly_summary = (
        day_df.groupby(['Hour','Time_Category'])['Energy']
        .sum()
        .reset_index()
    )

    st.plotly_chart(
        px.bar(hourly_summary, x='Hour', y='Energy', color='Time_Category', barmode='group'),
        use_container_width=True
    )

    # ================================
    # TIME FILTER
    # ================================
    st.subheader("⏱️ Select Time Category")

    selected_time = st.selectbox(
        "Choose Category",
        df_all['Time_Category'].unique()
    )

    filtered_df = day_df[day_df['Time_Category'] == selected_time]

    # ================================
    # TABLE
    # ================================
    st.subheader(f"📋 Instruments for {selected_time}")

    if filtered_df.empty:
        st.warning("⚠️ No data available for this selection")
    else:
        table_df = (
            filtered_df.groupby('Instrument')['Energy']
            .sum()
            .reset_index()
            .sort_values(by='Energy', ascending=False)
        )

        st.dataframe(table_df, use_container_width=True, height=700)

        st.plotly_chart(
            px.bar(table_df.head(20), x='Instrument', y='Energy'),
            use_container_width=True
        )

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
    st.dataframe(pivot_df, use_container_width=True, height=700)

    # ================================
    # ML PREDICTION
    # ================================
    st.subheader("🔮 Advanced ML Prediction")

    ts_df = total_df.groupby('Datetime')['Energy'].sum().reset_index().sort_values('Datetime')
    ts_df['t'] = np.arange(len(ts_df))

    future_t = np.arange(len(ts_df), len(ts_df)+24).reshape(-1,1)

    if ML_AVAILABLE:
        X = ts_df[['t']]
        y = ts_df['Energy']

        lin_pred = LinearRegression().fit(X,y).predict(future_t)
        gbr_pred = GradientBoostingRegressor().fit(X,y).predict(future_t)
    else:
        lin_pred = np.repeat(ts_df['Energy'].mean(), 24)
        gbr_pred = lin_pred

    future_dates = pd.date_range(start=ts_df['Datetime'].max(), periods=24, freq='h')

    df_plot = pd.DataFrame({
        'Datetime': list(ts_df['Datetime']) + list(future_dates)*2,
        'Value': list(ts_df['Energy']) + list(lin_pred) + list(gbr_pred),
        'Model': ['Actual']*len(ts_df) + ['Linear']*24 + ['XGBoost']*24
    })

    st.plotly_chart(px.line(df_plot, x='Datetime', y='Value', color='Model'),
                    use_container_width=True)

    # ================================
    # SYSTEM ANALYSIS (OLD)
    # ================================
    st.subheader("🎯 System Analysis")

    system_summary_old = instrument_df.groupby('Instrument')['Energy'].sum().reset_index()

    st.plotly_chart(px.bar(system_summary_old.head(20), x='Instrument', y='Energy'),
                    use_container_width=True)

    # ================================
    # 🚀 SMART SYSTEM ANALYSIS (F1 / F2)
    # ================================
    st.subheader("🎯 Advanced System Analysis")

    file_names = " ".join([f.name.lower() for f in files])

    if "f1" in file_names:
        dataset_type = "F1"
    elif "f2" in file_names:
        dataset_type = "F2"
    else:
        dataset_type = "GENERAL"

    def map_system_f1(name):
        name = str(name).lower()
        if "site" in name:
            return "Site Office 1 & 2"
        elif "canteen" in name or "admin" in name:
            return "Main Canteen"
        elif "mpdp" in name:
            return "MPDP Warehouse"
        elif "mldp" in name:
            return "MLDP 4 Power Plant"
        elif "process air" in name:
            return "Process Air"
        elif "mcc2" in name or "mcc3" in name:
            return "MCC"
        elif "light" in name or "ac" in name:
            return "Light & AC"
        elif "dg" in name:
            return "DG Auxiliary"
        else:
            return "Others"

    def map_system_f2(name):
        name = str(name).lower()
        if "light" in name or "ac" in name:
            return "Light & AC"
        elif "MBT-A" in name or "MBT-B" in name:
            return "MBT"
        elif "process air" in name:
            return "Process Air"
        elif "x7" in name:
            return "X7 Bay"
        elif "stp" in name:
            return "STP"
        elif "cooling" in name:
            return "Cooling Circuit Bus B"
        else:
            return "Others"

    system_df = df_all.copy()

    if dataset_type == "F1":
        system_df['System'] = system_df['Instrument'].apply(map_system_f1)
    elif dataset_type == "F2":
        system_df['System'] = system_df['Instrument'].apply(map_system_f2)
    else:
        system_df['System'] = "General"

    st.subheader("📅 Select Date for System Analysis")

    sys_date = st.selectbox(
        "Choose Date (System View)",
        sorted(system_df['Date'].unique()),
        key="system_date"
    )

    system_day = system_df[system_df['Date'] == sys_date]

    system_summary = (
        system_day.groupby('System')['Energy']
        .sum()
        .reset_index()
        .sort_values(by='Energy', ascending=False)
    )

    st.plotly_chart(
        px.bar(system_summary, x='System', y='Energy', color='System',
               title=f"{dataset_type} System Energy Consumption"),
        use_container_width=True
    )

    st.dataframe(system_summary, use_container_width=True, height=400)

    # ================================
    # AI INSIGHTS
    # ================================
    st.subheader("🤖 AI Insights")

    top_peak = pivot_df.sort_values(by='Peak', ascending=False).head(5)

    for _, row in top_peak.iterrows():
        st.write(f"⚠️ {row['Instrument']} → High peak consumption")

    st.success("🚀 FINAL DASHBOARD READY!")
