import streamlit as st
import pandas as pd

def selection_field(df: pd.DataFrame):

    product_type_list = ["All"] + sorted(df["product_type"].dropna().unique()) 
    selected_product_type = st.sidebar.selectbox("Select Product Type (optional)", product_type_list, key="product_type_selectbox")
    sku_list = sorted(df[df["product_type"] == selected_product_type]["sku"].dropna().unique()) if selected_product_type != "All" else sorted(df["sku"].dropna().unique())
    selected_sku = st.sidebar.selectbox("Select SKU", sku_list,key="sku_selectbox")
    tp_list = ["All"] + sorted(df[df["sku"] == selected_sku]["trade_partner"].dropna().unique())
    selected_tp = st.sidebar.selectbox("Select Trade Partner (optional)", tp_list, key="tp_selectbox")
    selected_date = st.sidebar.date_input(
        "Select Forecast Date (optional)",
        value=None,
        key="date_input"
    )
    return selected_product_type, selected_sku, selected_tp, selected_date