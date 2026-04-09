import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(layout="wide", page_title="AI Energy Dashboard")

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=250)
with col2:
    st.markdown("<h1 style='color:#2E86C1;'>🤖 AI Energy Optimization Dashboard</h1>", unsafe_allow_html=True)

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
    # TIME DISTRIBUTION
    # ================================
    st.subheader("🥧 Energy Distribution")

    time_summary = total_df.groupby(['Plant','Time_Category'])['Energy'].sum().reset_index()

    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy', facet_col='Plant'),
                    use_container_width=True)

    st.plotly_chart(px.pie(time_summary, names='Time_Category', values='Energy', hole=0.5, facet_col='Plant'),
                    use_container_width=True)

    st.plotly_chart(px.bar(time_summary, x='Time_Category', y='Energy', color='Plant'),
                    use_container_width=True)

    # ================================
    # INSTRUMENT ANALYSIS
    # ================================
    st.subheader("🏭 All Instruments")

    inst_summary = instrument_df.groupby('Instrument')['Energy'].sum().reset_index().sort_values(by='Energy', ascending=False)

    top_n = st.slider("Select Instruments", 10, len(inst_summary), 50)

    fig = px.bar(inst_summary.head(top_n), x='Instrument', y='Energy')
    fig.update_layout(xaxis_tickangle=-90, height=600, width=max(1200, top_n*20))

    st.plotly_chart(fig, use_container_width=True)

    # ================================
    # PEAK vs NON-PEAK
    # ================================
    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    peak_data = instrument_df.groupby(['Instrument','Peak_Type'])['Energy'].sum().reset_index()

    pivot_df = peak_data.pivot(index='Instrument', columns='Peak_Type', values='Energy').fillna(0)

    # 🔥 FIX (NO ERROR)
    if 'Peak' not in pivot_df.columns:
        pivot_df['Peak'] = 0
    if 'Non-Peak' not in pivot_df.columns:
        pivot_df['Non-Peak'] = 0

    pivot_df = pivot_df.reset_index()

    # ================================
    # AI CLASSIFICATION
    # ================================
    pivot_df['Category'] = pivot_df.apply(
        lambda x: "Main" if x.get('Peak',0) > x.get('Non-Peak',0) else "Optional",
        axis=1
    )

    st.subheader("🤖 AI Classification")
    st.dataframe(pivot_df)

    # ================================
    # COST MODEL
    # ================================
    st.subheader("💰 Cost Calculation")

    peak_rate = st.number_input("Peak Rate ₹", value=10)
    non_peak_rate = st.number_input("Non-Peak Rate ₹", value=6)

    pivot_df['Peak_Cost'] = pivot_df['Peak'] * peak_rate
    pivot_df['NonPeak_Cost'] = pivot_df['Non-Peak'] * non_peak_rate

    total_cost = pivot_df['Peak_Cost'].sum() + pivot_df['NonPeak_Cost'].sum()

    st.metric("Total Cost", f"₹ {round(total_cost,2)}")

    cost_df = pivot_df.melt(
        id_vars='Instrument',
        value_vars=['Peak_Cost','NonPeak_Cost'],
        var_name='Type',
        value_name='Cost'
    )

    st.plotly_chart(px.bar(cost_df, x='Instrument', y='Cost', color='Type'),
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

    st.success("""
    🤖 Recommendations:
    ✔ Shift optional loads  
    ✔ Reduce peak demand  
    ✔ Optimize system usage  
    ✔ Save electricity cost  
    """)
