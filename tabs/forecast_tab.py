import streamlit as st
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt

def detect_default_columns(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    res = {
        "date": None,
        "demand": None,
        "price": None,
        "profit": None,
        "sku": None,
        "trade_partner": None,
    }

    # date
    for key in ["date", "order_date", "ds", "timestamp"]:
        if key in cols_lower:
            res["date"] = cols_lower[key]
            break

    # demand / quantity
    for key in ["quantity", "qty", "demand", "units", "sales", "volume"]:
        for col in df.columns:
            if key in col.lower():
                res["demand"] = col
                break
        if res["demand"]:
            break

    # price
    for col in df.columns:
        if "price_final" == col.lower():
            res["price"] = col
            break
    if res["price"] is None:
        for col in df.columns:
            if "price" in col.lower():
                res["price"] = col
                break

    # profit / dcm
    for col in df.columns:
        if any(t in col.lower() for t in ["profit", "dcm", "margin"]):
            res["profit"] = col
            break

    # sku
    for col in df.columns:
        if any(t in col.lower() for t in ["sku", "product", "item", "material"]):
            res["sku"] = col
            break

    # trade partner
    for col in df.columns:
        if any(t in col.lower() for t in ["partner", "customer", "dealer", "distributor", "trade"]):
            res["trade_partner"] = col
            break

    return res

def forecast_tab(df: pd.DataFrame):

    defaults = detect_default_columns(df)
    date_col = defaults.get("date")
    demand_col = defaults.get("demand")

    if date_col is None or demand_col is None:
        st.error("A date column and a demand/quantity column are required for forecasting.")
        return

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Could not parse the date column: {e}")
        return

    ts = df[[date_col, demand_col]].copy()
    ts = ts.rename(columns={date_col: "ds", demand_col: "y"})
    ts = ts.dropna()

    # Aggregate to monthly to smooth noise
    ts = ts.set_index("ds").resample("M").sum().reset_index()
    ts = ts[ts["y"] > 0]

    if ts.empty:
        st.warning("No positive demand values available for forecasting.")
        return


    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        m.fit(ts)
        future = m.make_future_dataframe(periods=0, freq="M")
        fcst = m.predict(future)

        fig_components = m.plot_components(fcst, figsize=(15, 6))  # (width, height) in inches
        fig_components.tight_layout()
        st.pyplot(fig_components)
    except Exception as e:
        st.error(f"Prophet failed: {e}")
        return

    st.markdown(
        """
        **How to read this:**

        - The **trend** curve shows the long-run direction of demand across the entire Whirlpool dataset.
        - The **yearly seasonality** chart shows systematic peaks and dips across the months of the year.
        - These patterns can be used as context (or as features) for SKU-level models on the price grid.
        """
    )