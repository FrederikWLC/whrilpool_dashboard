import streamlit as st
import pandas as pd
import plots

def history_tab(df_sku: pd.DataFrame):
    st.plotly_chart(plots.historical_figure(df_sku,df_sku.sku.unique()[0]), use_container_width=True)