import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Care Transition Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

/* ================= MAIN APP ================= */

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.20);
}

[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
}


/* ================= PROFESSIONAL HEADER ================= */

.dashboard-header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    padding: 32px 38px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #cbd5e1;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.dashboard-header h1 {
    color: white !important;
    font-size: 38px;
    margin: 0;
    font-weight: 800;
}

.dashboard-header p {
    color: #dbeafe;
    font-size: 21px;
    margin: 6px 0;
    font-weight: 600;
}

.dashboard-header span {
    color: #cbd5e1;
    font-size: 14px;
}


/* ================= HEADINGS ================= */

h1 {
    color: #0f172a;
    font-weight: 800;
}

h2 {
    color: #1e293b;
    font-weight: 750;
}

h3 {
    color: #334155;
    font-weight: 700;
}


/* ================= KPI CARDS ================= */

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800;
}


/* ================= DATAFRAMES ================= */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}


/* ================= BUTTONS ================= */

.stButton > button {
    border-radius: 10px;
    border: 1px solid #cbd5e1;
    font-weight: 600;
    padding: 0.55rem 1rem;
}


/* ================= INPUTS ================= */

[data-baseweb="select"] > div {
    border-radius: 10px;
}

[data-testid="stDateInput"] input {
    border-radius: 10px;
}


/* ================= ALERTS ================= */

[data-testid="stAlert"] {
    border-radius: 12px;
}


/* ================= DIVIDERS ================= */

hr {
    border-color: #e2e8f0;
}


/* ================= INFO CARDS ================= */

.insight-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    margin-bottom: 15px;
}

.insight-card h4 {
    margin-top: 0;
    color: #1e3a5f;
}

.insight-card p {
    color: #475569;
    line-height: 1.6;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_excel(
        "HHS_Unaccompanied_Alien_Children_Program.xlsx"
    )

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    return data


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:24px;
        font-weight:800;
        margin-bottom:5px;
    ">
    📊 Care Analytics
    </div>

    <div style="
        font-size:12px;
        color:#cbd5e1;
        margin-bottom:20px;
    ">
    Unified Mentor • Data Science
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📈 Trend Analysis",
        "🔄 Transition Analysis",
        "🚨 Anomaly Detection",
        "🤖 Forecasting",
        "📋 Data Explorer",
        "ℹ️ Methodology"
    ]
)


st.sidebar.markdown("---")

st.sidebar.subheader("🔎 Filters")


min_date = df["Date"].min().date()
max_date = df["Date"].max().date()


start_date = st.sidebar.date_input(
    "Start Date",
    min_date
)

end_date = st.sidebar.date_input(
    "End Date",
    max_date
)


if start_date > end_date:

    st.sidebar.error(
        "Start date must be before end date."
    )

    st.stop()


filtered_df = df[
    (df["Date"].dt.date >= start_date)
    &
    (df["Date"].dt.date <= end_date)
].copy()


if filtered_df.empty:

    st.warning(
        "No records available for the selected dates."
    )

    st.stop()


# =========================================================
# COMMON VARIABLES
# =========================================================

cbp_col = "Children in CBP custody"

hhs_col = "Children in HHS Care"

discharge_col = "Children discharged from HHS Care"


transfer_col = None

if "Children transferred out of CBP custody" in filtered_df.columns:

    transfer_col = "Children transferred out of CBP custody"


# =========================================================
# PROFESSIONAL GLOBAL HEADER
# =========================================================

st.markdown(
"""<div class="dashboard-header">
<div class="header-badge">M.Sc. DATA SCIENCE • UNIFIED MENTOR PROJECT</div>
<h1>Care Transition Analytics</h1>
<p>Efficiency & Placement Outcome Analytics</p>
<span>Interactive data-driven analysis of care transitions, custody patterns and discharge outcomes</span>
</div>""",
unsafe_allow_html=True
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        "### 📌 Executive Overview"
    )

    st.caption(
        "Key indicators for the selected analysis period"
    )


    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    avg_cbp = filtered_df[cbp_col].mean()

    avg_hhs = filtered_df[hhs_col].mean()

    total_discharged = filtered_df[
        discharge_col
    ].sum()

    latest_cbp = filtered_df[
        cbp_col
    ].iloc[-1]

    latest_hhs = filtered_df[
        hhs_col
    ].iloc[-1]


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "📋 Total Records",
        f"{len(filtered_df):,}"
    )


    c2.metric(
        "👥 Avg. CBP Custody",
        f"{avg_cbp:,.0f}"
    )


    c3.metric(
        "🏥 Avg. HHS Care",
        f"{avg_hhs:,.0f}"
    )


    c4.metric(
        "✅ Total Discharged",
        f"{total_discharged:,.0f}"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Latest Status
    # -----------------------------------------------------

    st.markdown(
        "### 📍 Latest Available Status"
    )

    st.caption(
        f"Latest record: {filtered_df['Date'].iloc[-1].strftime('%d %B %Y')}"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "CBP Custody",
        f"{latest_cbp:,.0f}"
    )


    c2.metric(
        "HHS Care",
        f"{latest_hhs:,.0f}"
    )


    c3.metric(
        "CBP → HHS Difference",
        f"{latest_hhs - latest_cbp:,.0f}"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # Population Trends
    # -----------------------------------------------------

    st.markdown(
        "### 📈 Population Trends"
    )

    st.caption(
        "CBP custody and HHS care activity over the selected period"
    )


    trend_df = filtered_df[
        [
            "Date",
            cbp_col,
            hhs_col
        ]
    ].melt(
        id_vars="Date",
        var_name="Category",
        value_name="Children"
    )


    fig = px.line(
        trend_df,
        x="Date",
        y="Children",
        color="Category",
        markers=False,
        title="CBP Custody and HHS Care Over Time"
    )


    fig.update_layout(
        hovermode="x unified",
        height=500,
        template="plotly_white",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend_title_text=""
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


       # -----------------------------------------------------
    # Insights
    # -----------------------------------------------------

    st.subheader("💡 Automated Insights")

    max_cbp = filtered_df[cbp_col].max()
    max_hhs = filtered_df[hhs_col].max()
    min_cbp = filtered_df[cbp_col].min()
    min_hhs = filtered_df[hhs_col].min()

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
            **👥 CBP Custody**

            **Average:** {avg_cbp:,.0f}

            **Maximum:** {max_cbp:,.0f}

            **Minimum:** {min_cbp:,.0f}
            """
        )

    with col2:

        st.info(
            f"""
            **🏥 HHS Care**

            **Average:** {avg_hhs:,.0f}

            **Maximum:** {max_hhs:,.0f}

            **Minimum:** {min_hhs:,.0f}
            """
        )


# =========================================================
# TREND ANALYSIS
# =========================================================

elif page == "📈 Trend Analysis":

    st.header("📈 Detailed Trend Analysis")

    st.caption(
        "Explore changes in operational measures over time."
    )


    metric = st.selectbox(
        "Select metric",
        [
            cbp_col,
            hhs_col,
            discharge_col
        ]
    )


    fig = px.line(
        filtered_df,
        x="Date",
        y=metric,
        title=f"{metric} Over Time"
    )


    fig.update_layout(
        height=550,
        hovermode="x unified",
        template="plotly_white"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # Monthly Analysis
    # -----------------------------------------------------

    st.subheader("📅 Monthly Analysis")


    monthly = (
        filtered_df
        .set_index("Date")
        .resample("ME")[metric]
        .agg(
            [
                "mean",
                "max",
                "min",
                "sum"
            ]
        )
        .reset_index()
    )


    monthly.columns = [
        "Month",
        "Average",
        "Maximum",
        "Minimum",
        "Total"
    ]


    st.dataframe(
        monthly,
        use_container_width=True,
        hide_index=True
    )


    fig = px.bar(
        monthly,
        x="Month",
        y="Average",
        title=f"Monthly Average — {metric}"
    )


    fig.update_layout(
        template="plotly_white",
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TRANSITION ANALYSIS
# =========================================================

elif page == "🔄 Transition Analysis":

    st.header("🔄 Care Transition Analysis")

    st.caption(
        "Analyze transfers and differences between CBP custody and HHS care."
    )


    if transfer_col:

        st.subheader(
            "Children Transferred Out of CBP Custody"
        )


        total_transfer = filtered_df[
            transfer_col
        ].sum()


        avg_transfer = filtered_df[
            transfer_col
        ].mean()


        c1, c2 = st.columns(2)


        c1.metric(
            "Total Transfers",
            f"{total_transfer:,.0f}"
        )


        c2.metric(
            "Average Transfers",
            f"{avg_transfer:,.1f}"
        )


        fig = px.bar(
            filtered_df,
            x="Date",
            y=transfer_col,
            title="Transfers Over Time"
        )


        fig.update_layout(
            template="plotly_white",
            height=500
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # CBP vs HHS
    # -----------------------------------------------------

    st.subheader(
        "⚖️ CBP Custody vs HHS Care"
    )


    comparison = filtered_df[
        [
            "Date",
            cbp_col,
            hhs_col
        ]
    ].copy()


    comparison["Difference"] = (
        comparison[hhs_col]
        -
        comparison[cbp_col]
    )


    fig = px.area(
        comparison,
        x="Date",
        y="Difference",
        title="Difference Between HHS Care and CBP Custody"
    )


    fig.update_layout(
        template="plotly_white",
        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ANOMALY DETECTION
# =========================================================

elif page == "🚨 Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    st.caption(
        "Isolation Forest identifies unusual observations "
        "in the selected population metric."
    )


    anomaly_metric = st.selectbox(
        "Select metric for anomaly detection",
        [
            cbp_col,
            hhs_col,
            discharge_col
        ]
    )


    anomaly_data = filtered_df[
        [
            "Date",
            anomaly_metric
        ]
    ].copy()


    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )


    anomaly_data["Anomaly"] = model.fit_predict(
        anomaly_data[
            [anomaly_metric]
        ]
    )


    anomaly_data["Status"] = np.where(
        anomaly_data["Anomaly"] == -1,
        "⚠️ Anomaly",
        "Normal"
    )


    fig = px.scatter(
        anomaly_data,
        x="Date",
        y=anomaly_metric,
        color="Status",
        title=f"Anomaly Detection — {anomaly_metric}",
        hover_data=["Status"]
    )


    fig.update_layout(
        template="plotly_white",
        height=550
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    anomalies = anomaly_data[
        anomaly_data["Anomaly"] == -1
    ]


    st.subheader(
        f"⚠️ Detected Anomalies: {len(anomalies)}"
    )


    st.dataframe(
        anomalies,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FORECASTING
# =========================================================

elif page == "🤖 Forecasting":

    st.header("🤖 Trend Forecasting")

    st.caption(
        "Linear Regression provides a simple trend-based "
        "forecast for academic analysis."
    )


    forecast_metric = st.selectbox(
        "Select metric",
        [
            cbp_col,
            hhs_col
        ]
    )


    forecast_days = st.slider(
        "Forecast period (days)",
        min_value=7,
        max_value=90,
        value=30
    )


    model_data = filtered_df[
        [
            "Date",
            forecast_metric
        ]
    ].dropna().copy()


    model_data["Days"] = (
        model_data["Date"]
        -
        model_data["Date"].min()
    ).dt.days


    X = model_data[
        ["Days"]
    ]


    y = model_data[
        forecast_metric
    ]


    model = LinearRegression()

    model.fit(
        X,
        y
    )


    future_days = np.arange(
        model_data["Days"].max() + 1,
        model_data["Days"].max()
        +
        forecast_days
        +
        1
    )


    future_dates = pd.date_range(
        model_data["Date"].max()
        +
        pd.Timedelta(days=1),
        periods=forecast_days
    )


    predictions = model.predict(
        future_days.reshape(-1, 1)
    )


    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted": predictions
        }
    )


    # -----------------------------------------------------
    # Historical + Forecast
    # -----------------------------------------------------

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=model_data["Date"],
            y=model_data[forecast_metric],
            mode="lines",
            name="Historical"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted"],
            mode="lines",
            name="Forecast",
            line=dict(
                dash="dash"
            )
        )
    )


    fig.update_layout(
        title=f"{forecast_metric} — Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title="Children",
        height=550,
        template="plotly_white"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "📋 Forecasted Values"
    )


    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        "⚠️ This is a simple trend-based forecast for "
        "academic analysis, not an official prediction."
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "📋 Data Explorer":

    st.header("📋 Data Explorer")

    st.caption(
        "Explore and download the filtered dataset."
    )


    selected_columns = st.multiselect(
        "Select columns",
        filtered_df.columns.tolist(),
        default=filtered_df.columns.tolist()
    )


    if selected_columns:

        explorer_df = filtered_df[
            selected_columns
        ]


        st.dataframe(
            explorer_df,
            use_container_width=True,
            hide_index=True
        )


        csv = explorer_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇️ Download CSV",
            csv,
            "filtered_care_transition_data.csv",
            "text/csv"
        )


    # -----------------------------------------------------
    # Descriptive Statistics
    # -----------------------------------------------------

    st.subheader(
        "📊 Descriptive Statistics"
    )


    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )


    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    st.subheader(
        "🔗 Correlation Matrix"
    )


    numeric_df = filtered_df.select_dtypes(
        include=np.number
    )


    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()


        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Between Numeric Variables"
        )


        fig.update_layout(
            template="plotly_white",
            height=550
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# METHODOLOGY
# =========================================================

elif page == "ℹ️ Methodology":

    st.header(
        "ℹ️ Project Methodology"
    )

    st.caption(
        "Overview of the data science workflow used in this project."
    )


    st.subheader(
        "🎯 Objective"
    )


    st.write(
        "To analyze care transition patterns between CBP custody "
        "and HHS care and understand placement and discharge outcomes."
    )


    st.subheader(
        "🔄 Data Science Workflow"
    )


    st.markdown(
        """
        ### 1. Data Collection

        Excel dataset containing CBP and HHS care transition records.

        ### 2. Data Preprocessing

        - Date conversion
        - Sorting
        - Date filtering
        - Handling selected variables

        ### 3. Exploratory Data Analysis

        - Descriptive statistics
        - Averages
        - Minimum and maximum values
        - Correlation analysis

        ### 4. Data Visualization

        - Time-series analysis
        - Bar charts
        - Comparison charts
        - Interactive Plotly visualizations

        ### 5. Anomaly Detection

        Isolation Forest is used to identify unusual observations.

        ### 6. Predictive Analysis

        Linear Regression is used for basic trend forecasting.

        ### 7. Dashboard Development

        Streamlit is used to provide an interactive analytical interface.
        """
    )


    st.subheader(
        "🛠️ Technologies Used"
    )


    st.write(
        "Python • Pandas • NumPy • Plotly • "
        "Scikit-learn • Streamlit • Excel"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <b>Care Transition Efficiency & Placement Outcome Analytics</b><br>
        M.Sc. Data Science • Unified Mentor Project
    </div>
    """,
    unsafe_allow_html=True
)