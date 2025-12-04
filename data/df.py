
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    df_sku = df_sku.sort_values("date")
    return df_sku

def get_demand_weekly_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate to weekly demand by cluster
    df_weekly_by_cluster = (
        df.groupby(["year", "iso_week", "cluster"], as_index=False)
        .agg({"demand": "sum"})
        .sort_values(["cluster", "year", "iso_week"])
    )
    return df_weekly_by_cluster

def get_demand_weekly(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate to weekly demand
    df_weekly = (
        df.groupby(["year", "iso_week"], as_index=False)
        .agg({"demand": "sum"})
        .sort_values(["year", "iso_week"])
    )
    return df_weekly

def get_forecast_df(df,specific=False):
    # Scale demand between 0 and 1

    if specific:
        df_weekly = get_demand_weekly_by_cluster(df)
    else:
        df_weekly = get_demand_weekly(df)

    # for calculation purposes
    df_weekly["date"] = pd.to_datetime(
        df_weekly["year"].astype(str) + df_weekly["iso_week"].astype(str) + '1',
        format='%G%V%u'
    )

    df_weekly.rename(columns={"date": "ds", "demand": "y"}, inplace=True)
    return df_weekly

    