import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import warnings
from tabs import history_tab, forecast_tab, optimization_tab
from data import get_df_sku
from components import selection_field

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
        page_title="Whirlpool - Price Optimization",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="⚡",
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

        [data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
        [data-testid="stMetricValue"] { font-size: 2rem !important; }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Whirlpool — Price Optimization Dashboard")
    st.caption(
        "A user-friendly dashboard with sales analytics, demand forecasting, and price optimization — just provide the data!"
    )

    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Whirlpool_Corporation_Logo_%28as_of_2017%29.svg/2560px-Whirlpool_Corporation_Logo_%28as_of_2017%29.svg.png" alt="Whirlpool Logo" width="150">
        </div>
        """,
        unsafe_allow_html=True,
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

    selected_product_type, selected_sku, selected_tp, selected_date = selection_field(df)
    df_sku = get_df_sku(df, selected_sku, selected_tp)

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["Sales History", "Forecast", "Price Optimization Grid"]
    )


    with tab1:
        history_tab(df_sku)

    with tab2:
        if selected_date != None:
            forecast_tab(df, df_sku, selected_sku, selected_date)
        else:
            forecast_tab(df, df_sku, selected_sku)
    with tab3:
            optimization_tab(df_sku,selected_date)


if __name__ == "__main__":
    main()