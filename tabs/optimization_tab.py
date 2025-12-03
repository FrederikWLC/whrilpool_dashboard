import pandas as pd
import numpy as np
import streamlit as st
import plots
import xgboost as xgb

def optimization_tab(df: pd.DataFrame):
    st.subheader("Price Optimization Grid — XGBoost Point & Quantile Models")

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

