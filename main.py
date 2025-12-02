import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plots
import os
import warnings

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


def detect_default_columns(df: pd.DataFrame):
    """Heuristic detection of common column roles."""
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


def aggregate_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M"):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    ts = s.set_index(date_col).resample(freq).sum().reset_index()
    ts = ts.rename(columns={date_col: "ds", value_col: "y"})
    return ts


# -----------------------------
# Tabs
# -----------------------------
def sales_dashboard_tab(df: pd.DataFrame):
    st.subheader("Sales Dashboard")

    if "date" not in df.columns:
        st.error("The dataset must contain a 'date' column for this tab.")
        return

    # Ensure date parsed
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    st.markdown("Filter the period, inspect KPIs, and compare gross sales across products, partners, or SKUs.")

    # Date range picker
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.date_input(
        "Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    df_filtered = df[
        (df["date"].dt.date >= date_range[0])
        & (df["date"].dt.date <= date_range[1])
    ].copy()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    total_gross = df_filtered["gross_sales"].sum() if "gross_sales" in df_filtered.columns else np.nan
    avg_price = df_filtered["price_final"].mean() if "price_final" in df_filtered.columns else np.nan
    total_demand = df_filtered["demand"].sum() if "demand" in df_filtered.columns else np.nan
    avg_dcm = df_filtered["dcm"].mean() if "dcm" in df_filtered.columns else np.nan

    k1.metric("Total Gross Sales", f"${total_gross:,.0f}" if not np.isnan(total_gross) else "—")
    k2.metric("Average Final Price", f"${avg_price:,.2f}" if not np.isnan(avg_price) else "—")
    k3.metric("Total Demand", f"{total_demand:,.0f}" if not np.isnan(total_demand) else "—")
    k4.metric("Average DCM", f"{avg_dcm:,.2f}" if not np.isnan(avg_dcm) else "—")

    st.markdown("### Sales Performance Comparison")
    options = []
    if "product_type" in df_filtered.columns:
        options.append("Product Type")
    if "trade_partner" in df_filtered.columns:
        options.append("Trade Partner")
    if "sku" in df_filtered.columns:
        options.append("SKU")

    if not options:
        st.info("No product_type, trade_partner, or sku columns found for breakdown.")
    else:
        view_choice = st.selectbox("View Sales By", options)

        if view_choice == "Product Type":
            df_view = (
                df_filtered.groupby("product_type", as_index=False)["gross_sales"]
                .sum()
                .sort_values("gross_sales", ascending=False)
                .rename(columns={"product_type": "Category"})
            )
            x_col = "gross_sales"
            y_col = "Category"
        elif view_choice == "Trade Partner":
            df_view = (
                df_filtered.groupby("trade_partner", as_index=False)["gross_sales"]
                .sum()
                .sort_values("gross_sales", ascending=False)
                .rename(columns={"trade_partner": "Trade Partner"})
            )
            x_col = "gross_sales"
            y_col = "Trade Partner"
        else:
            df_view = (
                df_filtered.groupby("sku", as_index=False)["gross_sales"]
                .sum()
                .sort_values("gross_sales", ascending=False)
                .head(20)
                .rename(columns={"sku": "SKU"})
            )
            x_col = "gross_sales"
            y_col = "SKU"

        fig_bar = px.bar(
            df_view,
            x=x_col,
            y=y_col,
            orientation="h",
            title="Gross Sales by " + view_choice,
            labels={x_col: "Total Gross Sales", y_col: ""},
        )
        fig_bar.update_layout(
            template="simple_white",
            height=420,
            margin=dict(l=0, r=10, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Gross Sales Over Time")
    if "gross_sales" in df_filtered.columns:
        df_time = (
            df_filtered.groupby("date", as_index=False)["gross_sales"]
            .sum()
            .sort_values("date")
        )
        fig_time = px.line(
            df_time,
            x="date",
            y="gross_sales",
            labels={"date": "Date", "gross_sales": "Gross Sales"},
        )
        fig_time.update_traces(mode="lines+markers")
        fig_time.update_layout(template="simple_white", height=380)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Column 'gross_sales' not found in dataset.")


def forecast_tab(df: pd.DataFrame):
    st.subheader("Forecast — Global Demand Trend & Seasonality")

    if not HAS_PROPHET:
        st.error("Prophet is not installed in this environment. Install `prophet` to use this tab.")
        return

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

    st.caption(
        "This forecast uses Meta's Prophet on total demand, to extract long-term trend and recurring yearly patterns."
    )

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

        # Trend + yearly seasonality components
        fig_components = m.plot_components(fcst)
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


def price_grid_tab(df: pd.DataFrame):
    st.subheader("Price Optimization Grid — XGBoost Point & Quantile Models")

    if not HAS_XGBOOST:
        st.error("XGBoost is not installed in this environment. Install `xgboost` to use this tab.")
        return

    # Required columns as in your advanced grid
    required_cols = ["sku", "trade_partner", "price_final", "dcm", "quantity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns for price optimization: {missing}")
        return

    # SKU selection
    sku_list = sorted(df["sku"].dropna().unique())
    selected_sku = st.selectbox("Select SKU", sku_list)
    df_sku = df[df["sku"] == selected_sku].copy()

    # Optional trade partner filter
    tp_list = ["All"] + sorted(df_sku["trade_partner"].dropna().unique())
    selected_tp = st.selectbox("Select Trade Partner (optional)", tp_list)
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

    # -------- Approximate variable cost --------
    base_row = df_sku.iloc[[0]].copy()
    if "mean_total_variable_cost" in df_sku.columns:
        cost = float(base_row["mean_total_variable_cost"].iloc[0])
    else:
        tmp = df_sku[df_sku["quantity"] > 0].copy()
        if not tmp.empty:
            tmp["approx_cost"] = tmp["price_final"] - tmp["dcm"] / tmp["quantity"]
            cost = float(tmp["approx_cost"].mean())
        else:
            cost = float(base_row["price_final"].iloc[0]) * 0.7

    actual_price = float(base_row["price_final"].iloc[0])
    p_min = max(actual_price * 0.5, cost)
    p_max = actual_price * 1.5
    price_grid = np.linspace(p_min, p_max, 150)

    # Feature set
    candidate_features = [
        "price_final",
        "mean_price_final",
        "mean_price_list",
        "forecasted_demand",
        "price_final_var_coeff",
        "quantity_var_coeff",
        "real_inventory",
    ]
    features = [c for c in candidate_features if c in df_sku.columns]
    if "price_final" not in features:
        features.append("price_final")

    df_model = df_sku.dropna(subset=features + ["quantity"]).copy()
    if df_model.shape[0] < 10:
        st.warning("Not enough clean rows with all required features for modeling.")
        return

    X = df_model[features]
    y = df_model["quantity"]
    has_inv = "real_inventory" in df_model.columns

    # ========= 1. Historical performance =========
    st.markdown("### 1. Historical Performance (DCM → Quantity → Price)")

    st.plotly_chart(plots.historical_figure(df_sku,selected_sku), use_container_width=True)

    # ========= 2. Point estimate XGBoost model =========
    st.markdown("### 2. Point Estimate Model (XGBoost)")

    params_point = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    point_model = xgb.XGBRegressor(**params_point)
    point_model.fit(X, y)

    df_grid_point = pd.concat([base_row] * len(price_grid), ignore_index=True)
    df_grid_point["price_final"] = price_grid
    X_grid_point = df_grid_point[features]

    q_pred_point = np.round(point_model.predict(X_grid_point))

    if has_inv:
        inv = df_grid_point["real_inventory"].values
        q_pred_point = np.round(np.minimum(np.maximum(q_pred_point, 0), inv))
    else:
        q_pred_point = np.round(np.maximum(q_pred_point, 0))

    dcm_point = (df_grid_point["price_final"].values - cost) * q_pred_point
    best_idx_point = int(np.argmax(dcm_point))
    best_price_point = float(df_grid_point["price_final"].iloc[best_idx_point])
    best_dcm_point = float(dcm_point[best_idx_point])

   

    st.metric("Optimal Price (Point Estimate)", f"{best_price_point:,.2f} MXN")
    st.plotly_chart(plots.point_estimate_figure(df_sku, price_grid, dcm_point, best_price_point, best_dcm_point, q_pred_point), use_container_width=True)

    # ========= 3. Quantile XGBoost model =========
    st.markdown("### 3. Quantile Model (Uncertainty Band)")

    quantiles = {"lower": 0.05, "median": 0.5, "upper": 0.95}
    q_models = {}
    base_params = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }

    for name_q, alpha in quantiles.items():
        params_q = base_params.copy()
        params_q.update(
            {
                "objective": "reg:quantileerror",
                "quantile_alpha": alpha,
            }
        )
        m_q = xgb.XGBRegressor(**params_q)
        m_q.fit(X, y)
        q_models[name_q] = m_q

    qL = q_models["lower"].predict(X_grid_point)
    qM = q_models["median"].predict(X_grid_point)
    qU = q_models["upper"].predict(X_grid_point)

    if has_inv:
        inv = df_grid_point["real_inventory"].values
        qL = np.round(np.minimum(np.maximum(qL, 0), inv))
        qM = np.round(np.minimum(np.maximum(qM, 0), inv))
        qU = np.round(np.minimum(np.maximum(qU, 0), inv))
    else:
        qL = np.round(np.maximum(qL, 0))
        qM = np.round(np.maximum(qM, 0))
        qU = np.round(np.maximum(qU, 0))

    dcm_L = (price_grid - cost) * qL
    dcm_M = (price_grid - cost) * qM
    dcm_U = (price_grid - cost) * qU

    best_price_quant = float(df_grid_point["price_final"].iloc[best_idx_point])

    st.metric("Optimal Price (Quantile Median)", f"{best_price_quant:,.2f} MXN")
    st.plotly_chart(plots.quantile_estimate_figure(df_sku, price_grid, dcm_L, dcm_M, dcm_U, qL, qM, qU), use_container_width=True)

    st.markdown(
        """
**Interpretation:**

- The **point model** gives a single expected outcome for each price.
- The **quantile band** adds a pessimistic (5%) and optimistic (95%) scenario for both DCM and quantity.
- The recommended price is where **median DCM** is highest, while still keeping the band in a comfortable risk zone.
"""
    )


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
        ["Sales Dashboard", "Forecast", "Price Optimization Grid"]
    )

    with tab1:
        sales_dashboard_tab(df)

    with tab2:
        forecast_tab(df)

    with tab3:
        price_grid_tab(df)

    if show_sample:
        st.subheader("Data sample")
        st.dataframe(df.head(200))


if __name__ == "__main__":
    main()