import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="AI Energy Dashboard")

# ================================
# HEADER
# ================================
col1, col2 = st.columns([1,6])
with col1:
    st.image("SG logo1.jpg", width=200)
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

        df = df.melt(id_vars=['Instrument'], var_name='Datetime', value_name='Energy')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna()

        df['Hour'] = df['Datetime'].dt.hour
        df['Time_Category'] = df['Hour'].apply(classify_time)
        df['Plant'] = name

        all_data.append(df)

        total_candidate = df.groupby('Instrument')['Energy'].sum().idxmax()
        total_list.append(df[df['Instrument'] == total_candidate])
        instrument_list.append(df[df['Instrument'] != total_candidate])

    df_all = pd.concat(all_data)
    total_df = pd.concat(total_list)
    instrument_df = pd.concat(instrument_list)

    # ================================
    # AI PEAK CLASSIFICATION
    # ================================
    instrument_df['Peak_Type'] = instrument_df['Time_Category'].apply(
        lambda x: 'Peak' if 'Peak' in x else 'Non-Peak'
    )

    peak_data = instrument_df.groupby(['Instrument','Peak_Type'])['Energy'].sum().reset_index()

    pivot_df = peak_data.pivot(index='Instrument', columns='Peak_Type', values='Energy').fillna(0)

    pivot_df['Category'] = pivot_df.apply(
        lambda x: "Main" if x['Peak'] > x['Non-Peak'] else "Optional",
        axis=1
    )

    st.subheader("🤖 AI Classification (Main vs Optional)")
    st.dataframe(pivot_df)

    # ================================
    # COST MODEL
    # ================================
    st.subheader("💰 Cost Optimization")

    peak_rate = st.number_input("Peak Rate ₹/kWh", value=10)
    non_peak_rate = st.number_input("Non-Peak Rate ₹/kWh", value=6)

    pivot_df['Peak_Cost'] = pivot_df['Peak'] * peak_rate
    pivot_df['NonPeak_Cost'] = pivot_df['Non-Peak'] * non_peak_rate

    total_cost = pivot_df['Peak_Cost'].sum() + pivot_df['NonPeak_Cost'].sum()

    st.metric("💰 Total Energy Cost", f"₹ {round(total_cost,2)}")

    # ================================
    # VISUALIZATION
    # ================================
    st.subheader("📊 Cost Breakdown")

    cost_df = pivot_df.reset_index().melt(
        id_vars='Instrument',
        value_vars=['Peak_Cost','NonPeak_Cost'],
        var_name='Type',
        value_name='Cost'
    )

    st.plotly_chart(px.bar(cost_df, x='Instrument', y='Cost', color='Type'))

    # ================================
    # AI INSIGHTS
    # ================================
    st.subheader("🤖 AI Insights")

    high_peak = pivot_df.sort_values(by='Peak', ascending=False).head(5)

    for i, row in high_peak.iterrows():
        st.write(f"⚠️ {i} consumes high energy in peak → consider shifting")

    st.success("""
    🤖 AI Recommendations:
    
    ✔ Shift Optional loads to non-peak  
    ✔ Reduce peak demand charges  
    ✔ Optimize scheduling  
    ✔ Monitor high peak instruments  
    """)
