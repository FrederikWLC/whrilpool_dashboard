import streamlit as st
import pandas as pd
from model.prophet import do_forecast
from data.df import get_forecast_df
import plots


@st.cache_data(show_spinner=False)
def cached_forecast(df_forecast: pd.DataFrame, cutoff, name: str):
    model, train, test, forecast = do_forecast(
        df_forecast, 
        cutoff=cutoff, 
        name=name
    )
    return model, train, test, forecast

def forecast_tab(df: pd.DataFrame,df_sku, selected_sku, selected_date=None):

    # Model choice UI
    model_type = st.radio(
        "Choose scope:",
        ["Specific", "General"],
        horizontal=True
    )

    if model_type == "General":

        df_forecast = get_forecast_df(df)
        model, train, test, forecast = cached_forecast(df_forecast, cutoff=selected_date, name="General Trend")
        fig = plots.forecast_figure(forecast, train, test, "General Trend")
        st.session_state["prophet_general"] = model
        st.session_state["general_forecasted_demand"] = forecast.iloc[forecast["ds"].idxmax()]["yhat"]

    if model_type == "Specific":
        df_forecast = get_forecast_df(df_sku, specific=True)
        model, train, test, forecast = cached_forecast(df_forecast, cutoff=selected_date, name=selected_sku)
        fig = plots.forecast_figure(forecast, train, test, selected_sku)
        st.session_state["prophet_specific"] = model
        st.session_state["specific_forecasted_demand"] = forecast.iloc[forecast["ds"].idxmax()]["yhat"]

    st.pyplot(fig)
    with st.expander("See Forecast Components"):
        fig_components = model.plot_components(forecast, figsize=(15, 6))  # (width, height) in inches
        fig_components.tight_layout()
        st.pyplot(fig_components)
