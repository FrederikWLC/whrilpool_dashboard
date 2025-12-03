import plotly.graph_objects as go
from plotly.subplots import make_subplots

def point_estimate_figure(df_sku, price_grid, dcm_point, best_price_point, best_dcm_point, q_pred_point):
    fig_point = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
    )
    # Top: DCM
    fig_point.add_trace(
        go.Scatter(
            x=df_sku["price_final"],
            y=df_sku["dcm"],
            mode="markers",
            name="Historical",
            marker=dict(size=6, opacity=0.8, color="black"),
            hovertemplate="Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_point.add_trace(
        go.Scatter(
            x=price_grid,
            y=dcm_point,
            mode="lines",
            name="Predicted DCM",
            line=dict(color="blue", width=3),
            hovertemplate="Price: %{x:.2f} MXN<br>Pred DCM: %{y:.2f} MXN<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_point.add_trace(
        go.Scatter(
            x=[best_price_point],
            y=[best_dcm_point],
            mode="markers",
            name="Optimal Price",
            marker=dict(size=12, color="green"),
            hovertemplate="Optimal Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig_point.update_layout(
        shapes=[
            # Vertical green dashed line aligned with optimal price
            dict(
                type="line",
                x0=best_price_point,
                x1=best_price_point,
                yref="paper",
                y0=0,
                y1=1.05,
                line=dict(color="green", width=7.5, dash="solid"),
            )
        ],
        height=640,
        template="simple_white",
        title="Point Estimate Price Grid — DCM & Quantity Curves",
    )

    # Bottom: Quantity
    fig_point.add_trace(
        go.Scatter(
            x=df_sku["price_final"],
            y=df_sku["quantity"],
            mode="markers",
            name="Historical",
            marker=dict(size=6, opacity=0.8, color="black"),
            hovertemplate="Price: %{x:.2f} MXN<br>Quantity: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_point.add_trace(
        go.Scatter(
            x=price_grid,
            y=q_pred_point,
            mode="lines",
            name="Predicted Quantity",
            line=dict(color="blue", width=3, dash="dash"),
            hovertemplate="Price: %{x:.2f} MXN<br>Pred Qty: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig_point.update_xaxes(title_text="Price (MXN)", row=2, col=1)
    fig_point.update_yaxes(title_text="DCM (MXN)", row=1, col=1)
    fig_point.update_yaxes(title_text="Quantity", row=2, col=1)
    fig_point.update_layout(
        height=640,
        template="simple_white",
        title="Point Estimate Price Grid — DCM & Quantity Curves",
    )
    return fig_point