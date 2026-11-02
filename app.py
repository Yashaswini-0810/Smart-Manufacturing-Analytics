import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Smart Manufacturing Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme ----------
st.markdown("""
<style>
    .stApp { background: #080d18; color: #eef3ff; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stMetric"] {
        background: #151f31; border: 1px solid #263650;
        padding: 16px; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: #9db4d8; }
    [data-testid="stMetricValue"] { color: #f4f7ff; }
    .hero {
        background: linear-gradient(110deg,#11283a,#11182b);
        border: 1px solid #263b56; padding: 24px 28px;
        border-radius: 16px; margin-bottom: 18px;
    }
    .hero h1 { color: #f3f7ff; margin: 0; font-size: 2rem; }
    .hero p { color: #9db9df; margin: 8px 0 0 0; }
    .note {
        background: #182438; border-left: 4px solid #22c9c3;
        padding: 12px 16px; border-radius: 8px; color: #dbe8fa;
    }
    h1, h2, h3 { color: #f3f7ff; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "manufacturing_cleaned.csv"
RAW_FILE = "manufacturing_original.csv"
MODEL_FILE = "best_model_final.joblib"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Missing {DATA_FILE}. Place it in the same folder as app.py.")
        st.stop()
    data = pd.read_csv(DATA_FILE)
    for col in ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data

@st.cache_data
def load_original():
    if os.path.exists(RAW_FILE):
        original = pd.read_csv(RAW_FILE)
        for col in ["Scheduled_Start", "Scheduled_End", "Actual_Start", "Actual_End"]:
            if col in original.columns:
                original[col] = pd.to_datetime(original[col], errors="coerce")
        return original
    return None

df = load_data()
raw_df = load_original()

required = {
    "Job_ID", "Machine_ID", "Operation_Type", "Material_Used",
    "Processing_Time", "Energy_Consumption", "Machine_Availability",
    "Scheduled_Start", "Job_Status", "Optimization_Category"
}
missing = required.difference(df.columns)
if missing:
    st.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

# Use original records for actual timestamp completeness, where available.
report_df = raw_df.copy() if raw_df is not None else df.copy()

# ---------- Sidebar ----------
st.sidebar.markdown("## 🏭 Smart Mfg")
st.sidebar.caption("Manufacturing Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Executive Dashboard", "Machine Health", "Production Analysis",
     "Quality & Job Status", "Correlation Analysis", "Job Status Prediction"]
)
st.sidebar.divider()
st.sidebar.markdown("### Filters")

machines = sorted(df["Machine_ID"].dropna().astype(str).unique())
operations = sorted(df["Operation_Type"].dropna().astype(str).unique())
statuses = sorted(df["Job_Status"].dropna().astype(str).unique())

selected_machines = st.sidebar.multiselect("Machine ID", machines, default=machines)
selected_operations = st.sidebar.multiselect("Operation Type", operations, default=operations)
selected_statuses = st.sidebar.multiselect("Job Status", statuses, default=statuses)

filtered = df[
    df["Machine_ID"].astype(str).isin(selected_machines)
    & df["Operation_Type"].astype(str).isin(selected_operations)
    & df["Job_Status"].astype(str).isin(selected_statuses)
].copy()

# ---------- Helpers ----------
def moneyless_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def style_chart(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8efff",
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        legend_title_text=""
    )
    return fig

def empty_message():
    st.warning("No records match the selected filters. Please adjust the filters.")

st.markdown("""
<div class="hero">
  <h1>Smart Manufacturing Production Analytics and Performance Dashboard</h1>
  <p>Interactive overview of machine performance, production operations,
  job outcomes and predictive analytics.</p>
</div>
""", unsafe_allow_html=True)

if filtered.empty and page != "Job Status Prediction":
    empty_message()
    st.stop()

# ---------- Executive Dashboard ----------
if page == "Executive Dashboard":
    st.subheader("Executive Overview")
    total = len(filtered)
    completed = int((filtered["Job_Status"] == "Completed").sum())
    delayed = int((filtered["Job_Status"] == "Delayed").sum())
    failed = int((filtered["Job_Status"] == "Failed").sum())
    completion_rate = completed / total * 100 if total else 0
    avg_time = filtered["Processing_Time"].mean() if total else 0
    avg_energy = filtered["Energy_Consumption"].mean() if total else 0
    avg_avail = filtered["Machine_Availability"].mean() if total else 0

    k = st.columns(5)
    k[0].metric("Total Jobs", f"{total:,}")
    k[1].metric("Completion Rate", f"{completion_rate:.1f}%")
    k[2].metric("Delayed Jobs", f"{delayed:,}")
    k[3].metric("Failed Jobs", f"{failed:,}")
    k[4].metric("Avg. Machine Availability", f"{avg_avail:.1f}%")

    c1, c2 = st.columns(2)
    status_counts = filtered["Job_Status"].value_counts().rename_axis("Job Status").reset_index(name="Jobs")
    fig = px.pie(status_counts, names="Job Status", values="Jobs", hole=0.58,
                 title="Job Status Distribution")
    c1.plotly_chart(style_chart(fig), use_container_width=True)

    efficiency = filtered["Optimization_Category"].value_counts().rename_axis("Category").reset_index(name="Jobs")
    fig = px.pie(efficiency, names="Category", values="Jobs", hole=0.58,
                 title="Optimization Category Mix")
    c2.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    machine_jobs = filtered.groupby("Machine_ID").size().reset_index(name="Jobs")
    fig = px.bar(machine_jobs, x="Machine_ID", y="Jobs", title="Jobs by Machine", text="Jobs")
    c3.plotly_chart(style_chart(fig), use_container_width=True)

    trend = filtered.dropna(subset=["Scheduled_Start"]).copy()
    if not trend.empty:
        trend["Month"] = trend["Scheduled_Start"].dt.to_period("M").astype(str)
        monthly = trend.groupby(["Month", "Job_Status"]).size().reset_index(name="Jobs")
        fig = px.line(monthly, x="Month", y="Jobs", color="Job_Status",
                      markers=True, title="Monthly Job Status Trend")
        c4.plotly_chart(style_chart(fig), use_container_width=True)
    else:
        c4.info("Scheduled date information is unavailable for the selected records.")

    st.markdown("#### Key production averages")
    a, b, c = st.columns(3)
    a.metric("Average Processing Time", f"{avg_time:.2f}")
    b.metric("Average Energy Consumption", f"{avg_energy:.2f}")
    c.metric("Average Machine Availability", f"{avg_avail:.2f}")

# ---------- Machine Health ----------
elif page == "Machine Health":
    st.subheader("Machine Health & Utilization")
    machine_summary = filtered.groupby("Machine_ID").agg(
        Jobs=("Job_ID", "count"),
        Avg_Processing_Time=("Processing_Time", "mean"),
        Avg_Energy=("Energy_Consumption", "mean"),
        Avg_Availability=("Machine_Availability", "mean")
    ).reset_index()
    c1, c2 = st.columns(2)
    fig = px.bar(machine_summary, x="Machine_ID", y="Avg_Availability",
                 title="Average Machine Availability", text_auto=".1f")
    c1.plotly_chart(style_chart(fig), use_container_width=True)
    fig = px.bar(machine_summary, x="Machine_ID", y="Avg_Processing_Time",
                 title="Average Processing Time", text_auto=".1f")
    c2.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    fig = px.bar(machine_summary, x="Machine_ID", y="Avg_Energy",
                 title="Average Energy Consumption", text_auto=".2f")
    c3.plotly_chart(style_chart(fig), use_container_width=True)
    fig = px.bar(machine_summary, x="Machine_ID", y="Jobs",
                 title="Jobs Handled by Each Machine", text="Jobs")
    c4.plotly_chart(style_chart(fig), use_container_width=True)
    st.markdown("#### Machine summary")
    st.dataframe(machine_summary.round(2), use_container_width=True, hide_index=True)

# ---------- Production Analysis ----------
elif page == "Production Analysis":
    st.subheader("Production Analysis")
    c1, c2 = st.columns(2)
    op_counts = filtered.groupby("Operation_Type").size().reset_index(name="Jobs")
    fig = px.bar(op_counts, x="Operation_Type", y="Jobs", color="Operation_Type",
                 title="Jobs by Operation Type", text="Jobs")
    c1.plotly_chart(style_chart(fig), use_container_width=True)

    op_time = filtered.groupby("Operation_Type")["Processing_Time"].mean().reset_index()
    fig = px.bar(op_time, x="Operation_Type", y="Processing_Time",
                 title="Average Processing Time by Operation")
    c2.plotly_chart(style_chart(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    op_energy = filtered.groupby("Operation_Type")["Energy_Consumption"].mean().reset_index()
    fig = px.bar(op_energy, x="Operation_Type", y="Energy_Consumption",
                 title="Average Energy by Operation")
    c3.plotly_chart(style_chart(fig), use_container_width=True)

    if "Scheduled_Start" in filtered.columns:
        monthly = filtered.dropna(subset=["Scheduled_Start"]).copy()
        if not monthly.empty:
            monthly["Month"] = monthly["Scheduled_Start"].dt.to_period("M").astype(str)
            monthly_counts = monthly.groupby("Month").size().reset_index(name="Jobs")
            fig = px.line(monthly_counts, x="Month", y="Jobs", markers=True,
                          title="Scheduled Jobs by Month")
            c4.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            c4.info("No valid scheduled dates available.")
    st.markdown("#### Filtered production records")
    st.dataframe(filtered[["Job_ID", "Machine_ID", "Operation_Type", "Processing_Time",
                           "Energy_Consumption", "Job_Status"]].head(100),
                 use_container_width=True, hide_index=True)

# ---------- Quality & Job Status ----------
elif page == "Quality & Job Status":
    st.subheader("Quality & Job Status")
    c1, c2 = st.columns(2)
    status_machine = filtered.groupby(["Machine_ID", "Job_Status"]).size().reset_index(name="Jobs")
    fig = px.bar(status_machine, x="Machine_ID", y="Jobs", color="Job_Status",
                 barmode="group", title="Job Status by Machine")
    c1.plotly_chart(style_chart(fig), use_container_width=True)

    status_operation = filtered.groupby(["Operation_Type", "Job_Status"]).size().reset_index(name="Jobs")
    fig = px.bar(status_operation, x="Operation_Type", y="Jobs", color="Job_Status",
                 barmode="group", title="Job Status by Operation")
    c2.plotly_chart(style_chart(fig), use_container_width=True)

    category_status = filtered.groupby(["Optimization_Category", "Job_Status"]).size().reset_index(name="Jobs")
    fig = px.bar(category_status, x="Optimization_Category", y="Jobs", color="Job_Status",
                 barmode="stack", title="Optimization Category vs Job Status")
    st.plotly_chart(style_chart(fig, 420), use_container_width=True)

    if raw_df is not None and {"Actual_Start", "Actual_End", "Job_Status"}.issubset(raw_df.columns):
        failed = raw_df[raw_df["Job_Status"] == "Failed"]
        missing_start = int(failed["Actual_Start"].isna().sum())
        missing_end = int(failed["Actual_End"].isna().sum())
        st.markdown("#### Actual timestamp completeness")
        st.info(
            f"In the original data, {missing_start} failed jobs have no Actual_Start "
            f"and {missing_end} failed jobs have no Actual_End. These missing values "
            "are preserved for reporting and are not presented as recorded timestamps."
        )

# ---------- Correlation Analysis ----------
elif page == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    numeric = filtered.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Between Numerical Features")
        st.plotly_chart(style_chart(fig, 520), use_container_width=True)
    else:
        st.info("Not enough numerical columns are available for correlation analysis.")

    c1, c2 = st.columns(2)
    fig = px.scatter(filtered, x="Processing_Time", y="Energy_Consumption",
                     color="Job_Status", hover_data=["Machine_ID", "Operation_Type"],
                     title="Processing Time vs Energy Consumption")
    c1.plotly_chart(style_chart(fig), use_container_width=True)
    fig = px.scatter(filtered, x="Machine_Availability", y="Processing_Time",
                     color="Job_Status", hover_data=["Machine_ID"],
                     title="Machine Availability vs Processing Time")
    c2.plotly_chart(style_chart(fig), use_container_width=True)

# ---------- Prediction ----------
elif page == "Job Status Prediction":
    st.subheader("Predict Manufacturing Job Status")
    st.markdown(
        '<div class="note">Enter details for a prospective job. The model uses six inputs; '
        'Optimization Category is deliberately excluded because its relationship with '
        'the final status raised a possible target-leakage concern.</div>',
        unsafe_allow_html=True
    )

    if not os.path.exists(MODEL_FILE):
        st.error(f"Missing {MODEL_FILE}. Place it in the same folder as app.py.")
        st.stop()

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_FILE)

    model = load_model()

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        machine = c1.selectbox("Machine ID", sorted(df["Machine_ID"].dropna().unique()))
        operation = c2.selectbox("Operation Type", sorted(df["Operation_Type"].dropna().unique()))
        material = c1.number_input(
            "Material Used",
            min_value=float(df["Material_Used"].min()),
            max_value=float(df["Material_Used"].max()),
            value=float(df["Material_Used"].median())
        )
        processing = c2.number_input(
            "Processing Time",
            min_value=float(df["Processing_Time"].min()),
            max_value=float(df["Processing_Time"].max()),
            value=float(df["Processing_Time"].median())
        )
        energy = c1.number_input(
            "Energy Consumption",
            min_value=float(df["Energy_Consumption"].min()),
            max_value=float(df["Energy_Consumption"].max()),
            value=float(df["Energy_Consumption"].median())
        )
        availability = c2.number_input(
            "Machine Availability",
            min_value=float(df["Machine_Availability"].min()),
            max_value=float(df["Machine_Availability"].max()),
            value=float(df["Machine_Availability"].median())
        )
        submitted = st.form_submit_button("Predict Job Status", type="primary")

    if submitted:
        sample = pd.DataFrame([{
            "Machine_ID": machine,
            "Operation_Type": operation,
            "Material_Used": material,
            "Processing_Time": int(processing),
            "Energy_Consumption": energy,
            "Machine_Availability": int(availability)
        }])
        prediction = model.predict(sample)[0]
        st.success(f"Predicted Job Status: **{prediction}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            prob_df = pd.DataFrame({
                "Job Status": model.classes_,
                "Predicted Probability": probabilities
            }).sort_values("Predicted Probability", ascending=False)
            fig = px.bar(prob_df, x="Job Status", y="Predicted Probability",
                         title="Model Probability by Job Status",
                         text=prob_df["Predicted Probability"].map(lambda x: f"{x:.1%}"))
            st.plotly_chart(style_chart(fig), use_container_width=True)
        st.caption(
            "Model validation on the held-out test set: accuracy 57.5%, macro F1-score 0.3382. "
            "Treat this prediction as an experimental indication, not an operational decision."
        )

st.sidebar.divider()
st.sidebar.caption("Smart Manufacturing Analytics • Academic project")
st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} records")
