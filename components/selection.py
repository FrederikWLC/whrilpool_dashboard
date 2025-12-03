import streamlit as st
import pandas as pd

def selection_field(df: pd.DataFrame):

    sku_list = sorted(df["sku"].dropna().unique())
    tp_list = ["All"] + sorted(df["trade_partner"].dropna().unique())
    selected_sku = st.sidebar.selectbox("Select SKU", sku_list,key="sku_selectbox")
    selected_tp = st.sidebar.selectbox("Select Trade Partner (optional)", tp_list, key="tp_selectbox")
    return selected_sku, selected_tp