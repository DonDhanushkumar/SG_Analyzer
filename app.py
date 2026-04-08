import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# 📌 PAGE CONFIG
# ================================
st.set_page_config(layout="wide", page_title="SG Energy Dashboard")

# ================================
# 📌 HEADER (LOGO + TITLE)
# ================================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("SG logo1.jpg", width=200)

with col2:
    st.markdown(
        "<h1 style='color:#2E86C1;'>⚡ SG Smart Energy Optimization Dashboard</h1>",
        unsafe_allow_html=True
    )

# ================================
# 📌 TIME FUNCTION
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
# 📌 FILE UPLOAD
# ================================
files = st.file_uploader(
    "Upload Multiple Files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ================================
# 📌 MAIN PROCESS
# ================================
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

        # ================================
        # 📌 SMART TOTAL DETECTION
        # ================================
        total_candidate = df.groupby('Instrument')['Energy'].sum().idxmax()

        total_list.append(df[df['Instrument'] == total_candidate])
        instrument_list.append(df[df['Instrument'] != total_candidate])

    df_all = pd.concat(all_data, ignore_index=True)
    total_df = pd.concat(total_list, ignore_index=True)
    instrument_df = pd.concat(instrument_list, ignore_index=True)

    # ================================
    # 🥧 TIME DISTRIBUTION
    # ================================
    st.subheader("🥧 Energy Distribution (Total Only)")

    time_summary = total_df.groupby(['Plant', 'Time_Category'])['Energy'].sum().reset_index()

    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy', facet_col='Plant'),
                    use_container_width=True)

    st.subheader("🍩 Donut Chart")
    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy',
                           hole=0.5, facet_col='Plant'),
                    use_container_width=True)

    st.subheader("📊 Time Category Comparison")
    st.plotly_chart(px.bar(time_summary, x='Time_Category', y='Energy',
                           color='Plant', barmode='group'),
                    use_container_width=True)

    # ================================
    # 🏭 ALL INSTRUMENTS
    # ================================
    st.subheader("🏭 All Instruments")

    instrument_summary = (
        instrument_df.groupby('Instrument')['Energy']
        .sum()
        .reset_index()
        .sort_values(by='Energy', ascending=False)
    )

    top_n = st.slider("Select Instruments", 10, len(instrument_summary), 50)

    fig_inst = px.bar(instrument_summary.head(top_n), x='Instrument', y='Energy')
    fig_inst.update_layout(xaxis_tickangle=-90, height=600, width=max(1200, top_n * 20))

    st.plotly_chart(fig_inst, use_container_width=True)

    # ================================
    # ⚖️ INSTRUMENT PEAK ANALYSIS
    # ================================
    st.subheader("⚖️ Instrument-wise Peak vs Non-Peak")

    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    peak_compare = instrument_df.groupby(['Instrument', 'Peak_Type'])['Energy'].sum().reset_index()

    pivot_df = peak_compare.pivot(index='Instrument', columns='Peak_Type', values='Energy').fillna(0).reset_index()

    if 'Peak' not in pivot_df:
        pivot_df['Peak'] = 0
    if 'Non-Peak' not in pivot_df:
        pivot_df['Non-Peak'] = 0

    top_n_peak = st.slider("Peak Analysis Instruments", 10, len(pivot_df), 50)

    melt_df = pivot_df.head(top_n_peak).melt(
        id_vars='Instrument',
        value_vars=['Peak', 'Non-Peak'],
        var_name='Type',
        value_name='Energy'
    )

    fig_peak = px.bar(melt_df, x='Instrument', y='Energy', color='Type', barmode='group')
    fig_peak.update_layout(xaxis_tickangle=-90, height=600, width=max(1200, top_n_peak * 20))

    st.plotly_chart(fig_peak, use_container_width=True)

    # ================================
    # 🎯 SYSTEM ANALYSIS
    # ================================
    SYSTEMS = [
        "Light & AC",
        "MBT-A",
        "MBT-B",
        "Process Air",
        "X7 Bay",
        "STP",
        "Cooling Circuit Bus B"
    ]

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

    # ================================
    # 📊 SYSTEM TOTAL
    # ================================
    st.subheader("🎯 All Key Systems Consumption")

    system_summary = (
        instrument_df.groupby('System')['Energy']
        .sum()
        .reindex(SYSTEMS, fill_value=0)
        .reset_index()
    )

    st.plotly_chart(px.bar(system_summary, x='System', y='Energy', color='System'),
                    use_container_width=True)

    st.plotly_chart(px.pie(system_summary, names='System', values='Energy'),
                    use_container_width=True)

    # ================================
    # ⚖️ SYSTEM PEAK ANALYSIS
    # ================================
    st.subheader("⚖️ System-wise Peak vs Non-Peak")

    peak_sys = instrument_df.groupby(['System', 'Peak_Type'])['Energy'].sum().reset_index()

    st.plotly_chart(px.bar(peak_sys, x='System', y='Energy',
                           color='Peak_Type', barmode='group'),
                    use_container_width=True)

    # ================================
    # 📢 INSIGHTS
    # ================================
    st.subheader("📢 Insights")

    for plant in time_summary['Plant'].unique():
        max_cat = time_summary[time_summary['Plant'] == plant].loc[
            time_summary['Energy'].idxmax(), 'Time_Category'
        ]
        st.write(f"🔹 {plant} → Highest Consumption: **{max_cat}**")

    st.success("""
    💡 Recommendations:
    ✔ Shift peak load  
    ✔ Optimize systems  
    ✔ Reduce energy cost  
    ✔ Focus on high-load systems  
    """)
