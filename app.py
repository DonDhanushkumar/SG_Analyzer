import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# 📌 PAGE CONFIG
# ================================
st.set_page_config(layout="wide", page_title="SG Energy Dashboard")

# ================================
# 📌 LOGO + HEADER
# ================================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("SG logo1.jpg", width=150)  # 👉 place logo.png in same folder

with col2:
    st.markdown(
        "<h1 style='color:#2E86C1;'>⚡ SG Smart Energy Optimization Dashboard</h1>",
        unsafe_allow_html=True
    )

# ================================
# 📌 TIME CLASSIFICATION
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

    all_data = []
    total_list = []
    instrument_list = []

    for file in files:
        name = file.name

        # Load file
        if name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        df = df.fillna(0)

        # Reshape
        if 'Instrument' in df.columns:
            df = df.melt(id_vars=['Instrument'], var_name='Datetime', value_name='Energy')
        else:
            st.warning(f"{name} skipped (no Instrument column)")
            continue

        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        df['Hour'] = df['Datetime'].dt.hour
        df['Time_Category'] = df['Hour'].apply(classify_time)
        df['Plant'] = name

        all_data.append(df)

        # ================================
        # 📌 SMART TOTAL DETECTION
        # ================================
        total_candidate = (
            df.groupby('Instrument')['Energy']
            .sum()
            .idxmax()
        )

        total_temp = df[df['Instrument'] == total_candidate].copy()
        instrument_temp = df[df['Instrument'] != total_candidate].copy()

        total_list.append(total_temp)
        instrument_list.append(instrument_temp)

    # Combine all
    df_all = pd.concat(all_data, ignore_index=True)
    total_df = pd.concat(total_list, ignore_index=True)
    instrument_df = pd.concat(instrument_list, ignore_index=True)

    # ================================
    # 🥧 PIE CHART
    # ================================
    st.subheader("🥧 Energy Distribution (Total Only)")

    time_summary = (
        total_df.groupby(['Plant', 'Time_Category'])['Energy']
        .sum()
        .reset_index()
    )

    fig_pie = px.pie(
        time_summary,
        names='Time_Category',
        values='Energy',
        facet_col='Plant'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ================================
    # 🍩 DONUT CHART
    # ================================
    st.subheader("🍩 Donut Chart")

    fig_donut = px.pie(
        time_summary,
        names='Time_Category',
        values='Energy',
        hole=0.5,
        facet_col='Plant'
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # ================================
    # 📊 TIME BAR CHART
    # ================================
    st.subheader("📊 Time Category Comparison")

    fig_bar = px.bar(
        time_summary,
        x='Time_Category',
        y='Energy',
        color='Plant',
        barmode='group'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ================================
    # 🏭 ALL INSTRUMENTS
    # ================================
    st.subheader("🏭 All Instruments (Scrollable View)")

    instrument_summary = (
        instrument_df.groupby('Instrument')['Energy']
        .sum()
        .reset_index()
        .sort_values(by='Energy', ascending=False)
    )

    top_n = st.slider(
        "Select Number of Instruments",
        10,
        len(instrument_summary),
        50
    )

    fig_inst = px.bar(
        instrument_summary.head(top_n),
        x='Instrument',
        y='Energy'
    )

    fig_inst.update_layout(
        xaxis_tickangle=-90,
        height=600,
        width=max(1200, top_n * 20)
    )

    st.plotly_chart(fig_inst, use_container_width=True)

    # ================================
    # ⚖️ PEAK vs NON-PEAK
    # ================================
    st.subheader("⚖️ Instrument-wise Peak vs Non-Peak")

    instrument_df = instrument_df.copy()

    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    peak_compare = (
        instrument_df.groupby(['Instrument', 'Peak_Type'])['Energy']
        .sum()
        .reset_index()
    )

    pivot_df = peak_compare.pivot(
        index='Instrument',
        columns='Peak_Type',
        values='Energy'
    ).fillna(0).reset_index()

    # Safety fix
    if 'Peak' not in pivot_df.columns:
        pivot_df['Peak'] = 0
    if 'Non-Peak' not in pivot_df.columns:
        pivot_df['Non-Peak'] = 0

    pivot_df = pivot_df.sort_values(by='Peak', ascending=False)

    top_n_peak = st.slider(
        "Top Instruments for Peak Analysis",
        10,
        len(pivot_df),
        50
    )

    melt_df = pivot_df.head(top_n_peak).melt(
        id_vars='Instrument',
        value_vars=['Peak', 'Non-Peak'],
        var_name='Type',
        value_name='Energy'
    )

    fig_peak = px.bar(
        melt_df,
        x='Instrument',
        y='Energy',
        color='Type',
        barmode='group'
    )

    fig_peak.update_layout(
        xaxis_tickangle=-90,
        height=600,
        width=max(1200, top_n_peak * 20)
    )

    st.plotly_chart(fig_peak, use_container_width=True)

    # ================================
    # 📢 INSIGHTS
    # ================================
    st.subheader("📢 Insights")

    for plant in time_summary['Plant'].unique():
        temp = time_summary[time_summary['Plant'] == plant]
        max_cat = temp.loc[temp['Energy'].idxmax(), 'Time_Category']

        st.write(f"🔹 {plant} → Highest Consumption: **{max_cat}**")

    st.success("""
    💡 Recommendations:
    
    ✔ Shift peak load to non-peak  
    ✔ Optimize scheduling  
    ✔ Monitor high energy instruments  
    ✔ Reduce electricity cost  
    """)
