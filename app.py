import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Energy Dashboard")

st.markdown("""
<style>
.stDataFrame { font-size: 15px; }
.stDataFrame th { font-size: 17px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color:#2E86C1;'>🚀 Saint-Gobain AI Energy Optimization Dashboard</h1>",
    unsafe_allow_html=True,
)

# ── TIME CLASSIFIER ────────────────────────────────────────────────────────
def classify_time(hour: int) -> str:
    if 6 <= hour <= 9:
        return "Morning Peak"
    elif 18 <= hour <= 21:
        return "Evening Peak"
    elif hour == 5 or (10 <= hour <= 17):
        return "Morning Non-Peak"
    else:
        return "Evening Non-Peak"

# ── SYSTEM MAPPERS ─────────────────────────────────────────────────────────
def map_system_f1(name: str) -> str:
    n = str(name).lower()
    if "site" in n:                    return "Site Office 1 & 2"
    if "canteen db - incomer - 1" in n or "admin incomer-1" in n: return "Main Canteen"
    if "mpdp - 3 warehouse" in n:    return "MPDP Warehouse"
    if "mpdb - 4 - power plant" in n:  return "MPDP 4 Power Plant"
    if "process air" in n:             return "Process Air"
    if "mcc2" in n or "mcc3" in n:     return "MCC"
    if "light" in n or " ac" in n:     return "Light & AC"
    if "dg1 aux - incomer 1 & 2 (a)" in n: return "DG Auxiliary"
    return "Others"

def map_system_f2(name: str) -> str:
    n = str(name).lower()
    if "light" in n or " ac" in n:     return "Light & AC"
    if "mbt-a" in n or "mbt-b" in n:  return "MBT"
    if "process air" in n:             return "Process Air"
    if "x7" in n:                      return "X7 Bay"
    if "stp" in n:                     return "STP"
    if "cooling" in n:                 return "Cooling Circuit Bus B"
    return "Others"

# ── FILE UPLOAD ────────────────────────────────────────────────────────────
files = st.file_uploader(
    "Upload F1 and/or F2 hourly report files (.xlsx or .csv)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

if not files:
    st.info("Please upload one or more hourly report files to begin.")
    st.stop()

# ── DATA LOADING ───────────────────────────────────────────────────────────
TC_ORDER = ["Morning Peak", "Evening Peak", "Morning Non-Peak", "Evening Non-Peak"]

all_total_melted = []
all_instr_melted = []
dataset_types    = []
file_names       = []

def melt_rows(df_wide, plant_label, dtype_label):
    melted = df_wide.melt(id_vars=["Instrument"], var_name="Datetime", value_name="Energy")
    melted["Datetime"] = pd.to_datetime(melted["Datetime"], format="%b %d %Y %H:%M", errors="coerce")
    melted = melted.dropna(subset=["Datetime"])
    melted["Hour"]          = melted["Datetime"].dt.hour
    melted["Date"]          = melted["Datetime"].dt.normalize()
    melted["Time_Category"] = melted["Hour"].apply(classify_time)
    melted["Plant"]         = plant_label
    melted["Dataset"]       = dtype_label
    melted["Energy"]        = pd.to_numeric(melted["Energy"], errors="coerce").fillna(0)
    return melted

for file in files:
    name = file.name
    raw  = pd.read_csv(file) if name.lower().endswith(".csv") else pd.read_excel(file)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw = raw.fillna(0)

    if "Instrument" not in raw.columns:
        st.warning(f"⚠️ '{name}' has no 'Instrument' column — skipped.")
        continue

    fname_lower = name.lower()
    dtype = "F1" if "f1" in fname_lower else ("F2" if "f2" in fname_lower else "GENERAL")
    dataset_types.append(dtype)
    file_names.append(name)

    all_total_melted.append(melt_rows(raw.iloc[[0]].copy(),  name, dtype))
    if len(raw) > 1:
        all_instr_melted.append(melt_rows(raw.iloc[1:].copy(), name, dtype))

if not all_total_melted:
    st.error("No valid data loaded. Please check your files.")
    st.stop()

df = pd.concat(all_total_melted, ignore_index=True)
df["Date_Label"] = df["Datetime"].dt.strftime("%d-%b")
df["Sort_Date"]  = df["Datetime"].dt.date
df["Year"]       = df["Date"].dt.year
df["Month"]      = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")

if all_instr_melted:
    df_all = pd.concat([df] + all_instr_melted, ignore_index=True)
else:
    df_all = df.copy()
df_all["Date_Label"] = df_all["Datetime"].dt.strftime("%d-%b")
df_all["Sort_Date"]  = df_all["Datetime"].dt.date

primary_dtype = dataset_types[0] if dataset_types else "GENERAL"

# ── KPI ────────────────────────────────────────────────────────────────────
st.subheader("📊 KPI Dashboard")

peak_e     = df[df["Time_Category"].isin(["Morning Peak", "Evening Peak"])]["Energy"]
non_peak_e = df[df["Time_Category"].isin(["Morning Non-Peak", "Evening Non-Peak"])]["Energy"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ Total Energy (kWh)",    f"{df['Energy'].sum():,.2f}")
c2.metric("🔥 Peak Energy (kWh)",     f"{peak_e.sum():,.2f}"      if not peak_e.empty     else "N/A")
c3.metric("🌙 Non-Peak Energy (kWh)", f"{non_peak_e.sum():,.2f}"  if not non_peak_e.empty else "N/A")
c4.metric("📈 Avg Hourly (kWh)",      f"{df['Energy'].mean():,.2f}")

# ── ENERGY DISTRIBUTION PIE ────────────────────────────────────────────────
st.subheader("🥧 Energy Distribution by Time Category (Total Consumption)")

time_summary = df.groupby(["Plant", "Time_Category"])["Energy"].sum().reset_index()

if len(files) > 1:
    fig_pie = px.pie(
        time_summary, names="Time_Category", values="Energy", facet_col="Plant",
        title="Energy Distribution by Time Category per Plant"
    )
else:
    fig_pie = px.pie(
        time_summary, names="Time_Category", values="Energy",
        title=f"Energy Distribution — {file_names[0]}"
    )
st.plotly_chart(fig_pie, use_container_width=True)

# ── AVERAGE ENERGY BY TIME CATEGORY BAR (with Date Range Slicer) ───────────
st.subheader("📊 Average Energy by Time Category (Total Consumption)")

# Date Range Slicer
all_dates = sorted(df["Sort_Date"].unique())
col_start, col_end = st.columns(2)
with col_start:
    slicer_start = st.date_input(
        "📅 From Date",
        value=all_dates[0],
        min_value=all_dates[0],
        max_value=all_dates[-1],
        key="avg_bar_start",
    )
with col_end:
    slicer_end = st.date_input(
        "📅 To Date",
        value=all_dates[-1],
        min_value=all_dates[0],
        max_value=all_dates[-1],
        key="avg_bar_end",
    )

if slicer_start > slicer_end:
    st.warning("⚠️ 'From Date' must be before or equal to 'To Date'.")
    df_sliced = df.copy()
else:
    df_sliced = df[
        (df["Sort_Date"] >= slicer_start) &
        (df["Sort_Date"] <= slicer_end)
    ]

# AVERAGE (mean) energy per time category
time_avg_bar = (
    df_sliced.groupby("Time_Category")["Energy"]
    .mean()
    .reindex(TC_ORDER)
    .fillna(0)
    .reset_index()
)
time_avg_bar.rename(columns={"Energy": "Avg_Energy"}, inplace=True)

fig_avg_bar = px.bar(
    time_avg_bar,
    x="Time_Category",
    y="Avg_Energy",
    text_auto=".3s",
    color="Time_Category",
    title=f"Average Energy Consumption by Time Category  ({slicer_start} → {slicer_end})",
    labels={"Avg_Energy": "Average Energy (kWh)", "Time_Category": "Time_Category"},
    color_discrete_map={
        "Morning Peak":     "lightskyblue",
        "Evening Peak":     "royalblue",
        "Morning Non-Peak": "lightcoral",
        "Evening Non-Peak": "red",
    },
)
fig_avg_bar.update_traces(textposition="outside")
fig_avg_bar.update_layout(showlegend=True)
st.plotly_chart(fig_avg_bar, use_container_width=True)

# ── MONTHLY STACKED BAR ────────────────────────────────────────────────────
st.subheader("📊 Daily Energy Distribution — Stacked by Time Category (Total Consumption)")

sort_order     = df[["Date_Label", "Sort_Date"]].drop_duplicates().sort_values("Sort_Date")
monthly_detail = df.groupby(["Date_Label", "Time_Category"])["Energy"].sum().reset_index()
monthly_detail = monthly_detail.merge(sort_order, on="Date_Label").sort_values("Sort_Date")

st.plotly_chart(
    px.bar(monthly_detail, x="Date_Label", y="Energy", color="Time_Category",
           barmode="stack",
           title="Daily Total Consumption — All Days",
           category_orders={"Time_Category": TC_ORDER}),
    use_container_width=True,
)

# ── DAILY ENERGY TABLE ─────────────────────────────────────────────────────
st.subheader("📋 Daily Energy Table (Total Consumption)")

table_daily = (
    monthly_detail.pivot(index="Date_Label", columns="Time_Category", values="Energy")
    .fillna(0)
    .reset_index()
)
for col in TC_ORDER:
    if col not in table_daily.columns:
        table_daily[col] = 0
table_daily = table_daily[["Date_Label"] + TC_ORDER]
table_daily["Total"] = table_daily[TC_ORDER].sum(axis=1)
table_daily = (
    table_daily.merge(sort_order, on="Date_Label")
    .sort_values("Sort_Date")
    .drop(columns="Sort_Date")
)
st.dataframe(table_daily, use_container_width=True, height=500)

# ── DATE / TIME FILTER ─────────────────────────────────────────────────────
st.subheader("🎛️ Smart Date & Time Filter")

month_options = (
    df[["Year", "Month", "Month_Name"]]
    .drop_duplicates()
    .sort_values(["Year", "Month"])
)
month_labels  = month_options.apply(lambda x: f"{x['Month_Name']} {x['Year']}", axis=1).tolist()
selected_month = st.selectbox("Choose Month", month_labels)
sel_row        = month_options.iloc[month_labels.index(selected_month)]

month_df = df[
    (df["Year"] == sel_row["Year"]) &
    (df["Month"] == sel_row["Month"])
]

min_date      = month_df["Date"].min().date()
max_date      = month_df["Date"].max().date()
selected_date = st.date_input("Pick Date", value=min_date, min_value=min_date, max_value=max_date)

day_df        = month_df[month_df["Date"] == pd.Timestamp(selected_date)]
time_options  = ["All"] + TC_ORDER
selected_time = st.selectbox("Choose Time Category", time_options)

filtered_df   = day_df if selected_time == "All" else day_df[day_df["Time_Category"] == selected_time]

# ── HOURLY BAR (FILTERED DAY) ──────────────────────────────────────────────
st.subheader(f"📊 Hourly Total Consumption — {selected_date}  |  {selected_time}")

if filtered_df.empty:
    st.warning("⚠️ No data for this date / time selection.")
else:
    hourly_summary = filtered_df.groupby("Hour", as_index=False)["Energy"].sum()
    st.plotly_chart(
        px.bar(hourly_summary, x="Hour", y="Energy", text_auto=True,
               title=f"Hourly Total Consumption on {selected_date}"),
        use_container_width=True,
    )

# ── HOURLY LINE TREND ──────────────────────────────────────────────────────
st.subheader("📈 Hourly Energy Trend (Total Consumption)")

hourly_trend = df.groupby("Hour", as_index=False)["Energy"].mean()
st.plotly_chart(
    px.line(hourly_trend, x="Hour", y="Energy", markers=True,
            title="Average Total Consumption by Hour of Day"),
    use_container_width=True,
)

# ── MULTI-PLANT COMPARISON ─────────────────────────────────────────────────
if len(all_total_melted) > 1:
    st.subheader("🔀 Plant-wise Daily Comparison (Total Consumption)")
    plant_daily = df.groupby(["Date_Label", "Plant", "Sort_Date"])["Energy"].sum().reset_index()
    plant_daily = plant_daily.sort_values("Sort_Date")
    st.plotly_chart(
        px.line(plant_daily, x="Date_Label", y="Energy", color="Plant", markers=True,
                title="Daily Total Consumption — F1 vs F2"),
        use_container_width=True,
    )

# ── ADVANCED SYSTEM ANALYSIS ───────────────────────────────────────────────
st.subheader("🎯 Advanced System Analysis")

sys_date_options = sorted(df_all["Date"].dt.date.unique())
sys_date = st.selectbox("Choose Date (System View)", sys_date_options, key="system_date")

system_df = df_all[df_all["Date"] == pd.Timestamp(sys_date)].copy()
if primary_dtype == "F1":
    system_df["System"] = system_df["Instrument"].apply(map_system_f1)
elif primary_dtype == "F2":
    system_df["System"] = system_df["Instrument"].apply(map_system_f2)
else:
    system_df["System"] = "General"

system_summary = (
    system_df.groupby(["System", "Time_Category"])["Energy"].mean().reset_index()
)
system_summary.rename(columns={"Energy": "Avg_Energy"}, inplace=True)

# Exclude 'Others' from the chart
system_summary_chart = system_summary[system_summary["System"] != "Others"]

st.plotly_chart(
    px.bar(
        system_summary_chart,
        x="System",
        y="Avg_Energy",
        color="Time_Category",
        barmode="group",
        text_auto=".2f",
        title=f"{primary_dtype} System — Average Energy by Time Category ({sys_date})",
        labels={"Avg_Energy": "Avg Energy (kWh)", "Time_Category": "Time Category"},
        category_orders={"Time_Category": TC_ORDER},
        color_discrete_map={
            "Morning Peak":     "lightskyblue",
            "Evening Peak":     "royalblue",
            "Morning Non-Peak": "lightpink",
            "Evening Non-Peak": "red",
        },
    ),
    use_container_width=True,
)

pivot_sys = (
    system_summary.pivot(index="System", columns="Time_Category", values="Avg_Energy")
    .fillna(0)
    .reset_index()
)
for col in TC_ORDER:
    if col not in pivot_sys.columns:
        pivot_sys[col] = 0
pivot_sys = pivot_sys[["System"] + TC_ORDER]
pivot_sys["Avg Total"] = pivot_sys[TC_ORDER].sum(axis=1)
pivot_sys = pivot_sys.sort_values("Avg Total", ascending=False)
st.dataframe(pivot_sys, use_container_width=True, height=400)

# ── ML PREDICTION ──────────────────────────────────────────────────────────
st.subheader("🔮 ML Prediction — Next 24 Hours (Total Consumption)")

ts_df = df.groupby("Datetime")["Energy"].sum().reset_index().sort_values("Datetime")
ts_df["t"] = np.arange(len(ts_df))
future_t   = np.arange(len(ts_df), len(ts_df) + 24).reshape(-1, 1)

if ML_AVAILABLE and len(ts_df) >= 2:
    X, y     = ts_df[["t"]], ts_df["Energy"]
    lin_pred = LinearRegression().fit(X, y).predict(future_t)
    gbr_pred = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, y).predict(future_t)
else:
    mean_e   = ts_df["Energy"].mean()
    lin_pred = np.full(24, mean_e)
    gbr_pred = np.full(24, mean_e)

future_dates = pd.date_range(start=ts_df["Datetime"].max(), periods=25, freq="h")[1:]

df_plot = pd.DataFrame({
    "Datetime": list(ts_df["Datetime"]) + list(future_dates) * 2,
    "Value":    list(ts_df["Energy"])   + list(lin_pred)     + list(gbr_pred),
    "Model":    ["Actual"] * len(ts_df) + ["Linear"] * 24    + ["GradientBoost"] * 24,
})

st.plotly_chart(
    px.line(df_plot, x="Datetime", y="Value", color="Model",
            title="Total Consumption Forecast — Next 24 Hours"),
    use_container_width=True,
)

# ── AI INSIGHTS ────────────────────────────────────────────────────────────
st.subheader("🤖 AI Insights (Total Consumption)")

peak_rows    = df[df["Time_Category"].isin(["Morning Peak", "Evening Peak"])]
nonpeak_rows = df[df["Time_Category"].isin(["Morning Non-Peak", "Evening Non-Peak"])]

peak_total    = peak_rows["Energy"].sum()
nonpeak_total = nonpeak_rows["Energy"].sum()
grand_total   = df["Energy"].sum()

if grand_total > 0:
    st.write(f"⚡ **Peak hours** account for **{peak_total / grand_total * 100:.1f}%** of total consumption ({peak_total:,.2f} kWh)")
    st.write(f"🌙 **Non-peak hours** account for **{nonpeak_total / grand_total * 100:.1f}%** of total consumption ({nonpeak_total:,.2f} kWh)")

daily_totals = df.groupby("Date_Label")["Energy"].sum()
if not daily_totals.empty:
    max_day = daily_totals.idxmax()
    min_day = daily_totals.idxmin()
    st.write(f"📈 **Highest consumption day:** {max_day} — {daily_totals[max_day]:,.2f} kWh")
    st.write(f"📉 **Lowest consumption day:**  {min_day} — {daily_totals[min_day]:,.2f} kWh")

peak_hour = df.groupby("Hour")["Energy"].mean().idxmax()
st.write(f"🕐 **Highest average consumption hour:** {peak_hour}:00")

st.success("✅ Dashboard loaded successfully!")
