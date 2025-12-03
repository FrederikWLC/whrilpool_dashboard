import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def quantile_estimate_figure(df_sku, price_grid, dcm_L, dcm_M, dcm_U, qL, qM, qU):
    best_idx_quant = int(np.argmax(dcm_M))
    best_price_quant = float(price_grid[best_idx_quant])
    best_dcm_quant = float(dcm_M[best_idx_quant])

    fig_quant = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
    )

    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=dcm_M,
            mode="lines",
            name="Predicted DCM",
            line=dict(color="blue", width=3),
            hovertemplate="Price: %{x:.2f} MXN<br>Predicted DCM: %{y:.2f} MXN<extra></extra>",
            legendrank=1,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=1,
        col=1,
    )

    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=qM,
            mode="lines",
            name="Predicted Quantity",
            line=dict(width=3, dash="dash", color="blue"),
            hovertemplate="Price: %{x:.2f} MXN<br>Predicted Qty: %{y}<extra></extra>",
            legendrank=2,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=2,
        col=1,
    )

    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=dcm_U,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,120,255,0.15)",
            line=dict(width=0),
            name="DCM Uncertainty Band",
            hoverinfo='none',
            legendrank=3,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=1,
        col=1,
    )

    fig_quant.add_trace(
        go.Scatter(
            x=[best_price_quant],
            y=[best_dcm_quant],
            mode="markers",
            name="Optimal Price",
            marker=dict(size=11, color="green"),
            hovertemplate="Optimal Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>",
            legendrank=4,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=1,
        col=1,
    )

    # Top: DCM band
    fig_quant.add_trace(
        go.Scatter(
            x=df_sku["price_final"],
            y=df_sku["dcm"],
            mode="markers",
            name="Historical DCM",
            marker=dict(size=6, opacity=0.8, color="black"),
            hovertemplate="Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>",
            legendrank=5,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=1,
        col=1,
    )

    # Bottom: Quantity band
    fig_quant.add_trace(
        go.Scatter(
            x=df_sku["price_final"],
            y=df_sku["quantity"],
            mode="markers",
            name="Historical Quantity",
            marker=dict(size=6, opacity=0.8, color="black"),
            hovertemplate="Price: %{x:.2f} MXN<br>Qty: %{y}<extra></extra>",
            legendrank=6,
            hoverlabel=dict(
                font=dict(
                    size=16,  # Set the desired font size here (e.g., 16)
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.7)', # Optional: Customize background color
                bordercolor='white' # Optional: Customize border color
            )
        ),
        row=2,
        col=1,
    )
    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=dcm_L,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo='none'
        ),
        row=1,
        col=1,
    )

    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=qL,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo='none'
        ),
        row=2,
        col=1,
    )
    fig_quant.add_trace(
        go.Scatter(
            x=price_grid,
            y=qU,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,120,255,0.15)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo='none'
        ),
        row=2,
        col=1,
    )

    fig_quant.update_xaxes(title_text="Price (MXN)", row=2, col=1)
    fig_quant.update_yaxes(title_text="DCM (MXN)", row=1, col=1)
    fig_quant.update_yaxes(title_text="Quantity", row=2, col=1)
    fig_quant.update_layout(
        height=640,
        template="simple_white",
        title="Quantile Estimate Price Grid — DCM & Quantity Uncertainty Bands",
        shapes=[
            # Vertical green dashed line aligned with optimal price
            dict(
                type="line",
                x0=best_price_quant,
                x1=best_price_quant,
                yref="paper",
                y0=0,
                y1=1.05,
                line=dict(color="green", width=7.5, dash="solid"),
            )
        ]
    )

    return fig_quant