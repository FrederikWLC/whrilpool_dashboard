import plotly.graph_objects as go
from plotly.subplots import make_subplots

def historical_figure(df_sku, selected_sku):
    fig_hist = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.33, 0.33, 0.33],
    )

    # DCM
    fig_hist.add_trace(
        go.Scatter(
            x=df_sku["date"],
            y=df_sku["dcm"],
            mode="lines+markers",
            name="DCM",
            line=dict(color="lightgreen", width=3),
            marker=dict(size=7),
            hovertemplate="Date: %{x}<br>DCM: %{y:.2f} MXN<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_hist.update_yaxes(title_text="DCM (MXN)", row=1, col=1)

    # Quantity
    fig_hist.add_trace(
        go.Scatter(
            x=df_sku["date"],
            y=df_sku["quantity"],
            mode="lines+markers",
            name="Quantity",
            line=dict(color="lightblue", width=3),
            marker=dict(size=7),
            hovertemplate="Date: %{x}<br>Quantity: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_hist.update_yaxes(title_text="Quantity", row=2, col=1)

    # Price
    fig_hist.add_trace(
        go.Scatter(
            x=df_sku["date"],
            y=df_sku["price_final"],
            mode="lines+markers",
            name="Price",
            line=dict(color="darkgrey", width=3),
            marker=dict(size=7),
            hovertemplate="Date: %{x}<br>Price: %{y:.2f} MXN<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig_hist.update_yaxes(title_text="Price (MXN)", row=3, col=1)
    fig_hist.update_xaxes(title_text="Date", row=3, col=1)

    fig_hist.update_layout(
        height=780,
        template="simple_white",
        showlegend=False,
        title=dict(text=f"Historical Time Series — {selected_sku}", x=0,font=dict(size=20)),
    )
    return fig_hist