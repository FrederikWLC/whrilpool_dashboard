
import streamlit as st
import pandas as pd

def get_df_sku(df: pd.DataFrame, selected_sku: str, selected_tp: str) -> pd.DataFrame:    

    # Optional trade partner filter
    df_sku = df[df["sku"] == selected_sku].copy()
    if selected_tp != "All":
        df_sku = df_sku[df_sku["trade_partner"] == selected_tp]

    if df_sku.shape[0] < 10:
        st.warning("Not enough observations for this (SKU, trade partner) selection.")
        return

    # Build or reconstruct a usable date column for time series
    if "date" in df_sku.columns:
        df_sku["date"] = pd.to_datetime(df_sku["date"], errors="coerce")
    elif "year" in df_sku.columns and "iso_week" in df_sku.columns:
        df_sku["date"] = pd.to_datetime(
            df_sku["year"].astype(str) + df_sku["iso_week"].astype(str) + "1",
            format="%G%V%u",
            errors="coerce",
        )
    else:
        st.error("No usable date column found (need 'date', or 'year' + 'iso_week').")
        return

    df_sku = df_sku.dropna(subset=["date"]).sort_values("date")
    return df_sku