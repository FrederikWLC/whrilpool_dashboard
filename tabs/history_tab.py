import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def history_tab(df: pd.DataFrame):
    st.subheader("Sales History")

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