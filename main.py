import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plots
import os
import warnings
from tabs import history_tab, forecast_tab, optimization_tab

# Optional heavy deps
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

# -----------------------------
# Helpers
# -----------------------------
def load_data(uploaded_file=None):
    """Load user-uploaded CSV/Excel or fall back to known Whirlpool datasets."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.read_excel(uploaded_file)
    else:
        # Prefer clustered dataset
        if os.path.exists("ML_Clustered_Database_Horizon_Global_Consulting.csv"):
            df = pd.read_csv("ML_Clustered_Database_Horizon_Global_Consulting.csv")
        elif os.path.exists("Final_Database_Horizon_Global_Consulting.csv"):
            df = pd.read_csv("Final_Database_Horizon_Global_Consulting.csv")
        else:
            raise FileNotFoundError(
                "No default dataset found. Please upload a CSV/Excel file in the sidebar."
            )

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Basic numeric cleaning matching your first dashboard
    numeric_cols = [
        "inventory",
        "quantity",
        "gross_sales",
        "year",
        "month",
        "quarter",
        "iso_week",
        "cpi",
        "price_list",
        "price_final",
        "vpc",
        "wty",
        "varfw",
        "varsga",
        "usd_to_mxn",
        "total_variable_cost",
        "real_discount_pct",
        "dcm",
        "tp_pt",
        "tp_sku",
        "demand",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "gross_sales" in df.columns:
        df["gross_sales"] = df["gross_sales"].fillna(0)
    if "price_final" in df.columns:
        df["price_final"] = df["price_final"].fillna(df["price_final"].median())
    if "dcm" in df.columns:
        df["dcm"] = df["dcm"].fillna(0)
    if "demand" in df.columns:
        df["demand"] = df["demand"].fillna(0)

    return df


def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def aggregate_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M"):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    ts = s.set_index(date_col).resample(freq).sum().reset_index()
    ts = ts.rename(columns={date_col: "ds", value_col: "y"})
    return ts

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(
        page_title="Whirlpool — Sales & Price Optimization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    warnings.filterwarnings("ignore")

    # Light / simple theme styling
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Whirlpool — Sales & Price Optimization")
    st.caption(
        "Unified dashboard with sales analytics, global demand forecasting, and advanced price optimization."
    )

    # Sidebar: data input
    st.sidebar.header("Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV/Excel (optional)", type=["csv", "xlsx", "xls"]
    )

    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(str(e))
        return

    st.sidebar.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")
    show_sample = st.sidebar.checkbox("Show data sample")

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["Sales History", "Forecast", "Price Optimization Grid"]
    )

    with tab1:
        history_tab(df)

    with tab2:
        forecast_tab(df)

    with tab3:
        optimization_tab(df)

    if show_sample:
        st.subheader("Data sample")
        st.dataframe(df.head(200))


if __name__ == "__main__":
    main()