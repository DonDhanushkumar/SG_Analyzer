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

st.set_page_config(layout="wide")

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=200)
with col2:
    st.markdown("<h1 style='color:#2E86C1;'>🚀 AI Energy Dashboard</h1>", unsafe_allow_html=True)

# ================================
# TIME CLASSIFICATION
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
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df = df.fillna(0)

        if 'Instrument' not in df.columns:
            continue

        df = df.melt(id_vars=['Instrument'], var_name='Datetime', value_name='Energy')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        df['Hour'] = df['Datetime'].dt.hour
        df['Time_Category'] = df['Hour'].apply(classify_time)

        all_data.append(df)

        total_candidate = df.groupby('Instrument')['Energy'].sum().idxmax()
        total_list.append(df[df['Instrument'] == total_candidate])
        instrument_list.append(df[df['Instrument'] != total_candidate])

    total_df = pd.concat(total_list)
    instrument_df = pd.concat(instrument_list)

    # ================================
    # KPI
    # ================================
    st.subheader("📊 KPI")

    st.metric("Total Energy", round(total_df['Energy'].sum(),2))

    # ================================
    # SYSTEM MAPPING
    # ================================
    def map_system(name):
        name = str(name).lower()
        if "light" in name or "ac" in name:
            return "Light & AC"
        elif "process air" in name:
            return "Process Air"
        elif "stp" in name:
            return "STP"
        elif "x7" in name:
            return "X7 Bay"
        elif "cooling" in name:
            return "Cooling Circuit Bus B"
        else:
            return "Others"

    instrument_df['System'] = instrument_df['Instrument'].apply(map_system)

    # ================================
    # FIXED PEAK / NON-PEAK LOGIC
    # ================================
    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    # Group
    system_group = instrument_df.groupby(['System','Peak_Type'])['Energy'].sum().reset_index()

    # Pivot (IMPORTANT FIX)
    pivot_sys = system_group.pivot(index='System', columns='Peak_Type', values='Energy').fillna(0)

    # Ensure BOTH columns exist
    if 'Peak' not in pivot_sys.columns:
        pivot_sys['Peak'] = 0
    if 'Non-Peak' not in pivot_sys.columns:
        pivot_sys['Non-Peak'] = 0

    pivot_sys = pivot_sys.reset_index()

    # Convert to long format
    plot_sys = pivot_sys.melt(
        id_vars='System',
        value_vars=['Peak','Non-Peak'],
        var_name='Type',
        value_name='Energy'
    )

    # ================================
    # FINAL CHART (SIDE BY SIDE)
    # ================================
    st.subheader("🎯 System Peak vs Non-Peak (Fixed)")

    fig = px.bar(
        plot_sys,
        x='System',
        y='Energy',
        color='Type',
        barmode='group',   # 👈 SIDE BY SIDE
        text_auto=True
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-20
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Now showing BOTH Peak & Non-Peak correctly!")
