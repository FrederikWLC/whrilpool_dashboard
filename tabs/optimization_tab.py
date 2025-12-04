import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import plots


def optimization_tab(df_sku: pd.DataFrame,selected_date=None):

    

    df_sku["forecasted_demand"] = df_sku["demand"]

    base_row = df_sku.iloc[[0]]
    cost = float(base_row["total_variable_cost"]) if "total_variable_cost" in base_row else 0.0

    # Pricing exploration window
    actual_price = float(base_row["price_final"])
    p_min = actual_price * 0.5
    p_max = actual_price * 1.5
    price_grid = np.linspace(p_min, p_max, 150)

    # Features for the model
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

    df_model = df_sku.dropna(subset=features + ["quantity"])
    if df_model.shape[0] < 10:
        st.warning("Not enough clean rows with required features.")
        return

    X = df_model[features]
    y = df_model["quantity"]
    has_inv = "real_inventory" in df_model.columns

    df_grid_point = pd.concat([base_row] * len(price_grid), ignore_index=True)
    df_grid_point["price_final"] = price_grid
    X_grid_point = df_grid_point[features]
    X_grid_point["forecasted_demand"].values[:] = st.session_state.get("specific_forecasted_demand",0) if selected_date else df_sku["demand"].mean()

    df_sku["hover_text"] = (
        "Y" + df_sku["year"].astype(str) +
        " W" + df_sku["iso_week"].astype(str)
    )

    # Recency color scaling (if 'ds' + selected cutoff exist)
    if selected_date is not None and "ds" in df_sku.columns:
        sel_date = pd.to_datetime(selected_date)
        recency = np.abs((df_sku["ds"] - sel_date).dt.days)
        rec_norm = 1 - (recency / recency.max()).fillna(0)  # closer → higher
    else:
        rec_norm = np.full(len(df_sku), 0.5)  # neutral color fallback

    df_sku["recency_norm"] = rec_norm

    # Model choice UI
    model_type = st.radio(
        "Choose estimator:",
        ["Point", "Quantile"],
        horizontal=True
    )

    if model_type == "Point":
        point_tab(
            df_sku=df_sku,
            price_grid=price_grid,
            X=X,
            y=y,
            cost=cost,
            has_inv=has_inv,
            df_grid_point=df_grid_point,
            X_grid_point=X_grid_point,
        )
    else:
        quant_tab(
            df_sku=df_sku,
            price_grid=price_grid,
            X=X,
            y=y,
            cost=cost,
            has_inv=has_inv,
            df_grid_point=df_grid_point,
            X_grid_point=X_grid_point,
        )


def point_tab(df_sku, price_grid, X, y, cost, has_inv,
              df_grid_point, X_grid_point):

    params_point = dict(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model = xgb.XGBRegressor(**params_point).fit(X, y)

    q_pred = model.predict(X_grid_point)
    q_pred = np.maximum(q_pred, 0)
    if has_inv:
        q_pred = np.minimum(q_pred, df_grid_point["real_inventory"].values)
    q_pred = np.round(q_pred)

    dcm = (price_grid - cost) * q_pred
    best_idx = int(np.argmax(dcm))
    best_price = float(price_grid[best_idx])
    best_dcm = float(dcm[best_idx])

    st.metric("Estimated Optimal Price", f"{best_price:,.2f} MXN")

    fig = plots.point_estimate_figure(df_sku, price_grid, dcm,
                                      best_price, best_dcm, q_pred)
    st.plotly_chart(fig, use_container_width=True)

def quant_tab(df_sku, price_grid, X, y, cost, has_inv,
              df_grid_point, X_grid_point):

    quantiles = {"lower": 0.05, "median": 0.5, "upper": 0.95}
    base_params = dict(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    q_pred = {}
    for name, alpha in quantiles.items():
        params = base_params.copy()
        params.update(objective="reg:quantileerror", quantile_alpha=alpha)
        model = xgb.XGBRegressor(**params).fit(X, y)
        q_pred[name] = model.predict(X_grid_point)

    qL, qM, qU = (np.maximum(q_pred[k], 0) for k in ["lower", "median", "upper"])

    if has_inv:
        inv = df_grid_point["real_inventory"].values
        qL = np.minimum(qL, inv)
        qM = np.minimum(qM, inv)
        qU = np.minimum(qU, inv)

    qL, qM, qU = np.round(qL), np.round(qM), np.round(qU)

    dcm_L = (price_grid - cost) * qL
    dcm_M = (price_grid - cost) * qM
    dcm_U = (price_grid - cost) * qU

    best_price = float(price_grid[np.argmax(dcm_M)])
    st.metric("Estimated Optimal Price", f"{best_price:,.2f} MXN")

    fig = plots.quantile_estimate_figure(
        df_sku, price_grid,
        dcm_L, dcm_M, dcm_U,
        qL, qM, qU
    )
    st.plotly_chart(fig, use_container_width=True)