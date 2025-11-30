import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# optional heavy deps will be imported lazily
_HAS_XGBOOST = False
_HAS_PROPHET = False
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    xgb = None

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    Prophet = None


def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.read_excel(uploaded_file)
    else:
        # prefer clustered dataset if present
        if os.path.exists('ML_Clustered_Database_Horizon_Global_Consulting.csv'):
            df = pd.read_csv('ML_Clustered_Database_Horizon_Global_Consulting.csv')
        else:
            # fallback to original default
            df = pd.read_csv('Final_Database_Horizon_Global_Consulting.csv')
    return df


def load_saved_artifacts(search_dir='.'):
    """Look for common saved artifacts (pickled models, processed csvs) in the given directory.
    Returns a dict with keys: 'models' -> {name: model}, 'data' -> {name: path}
    """
    artifacts = {'models': {}, 'data': {}}
    try:
        # search for pickles
        pkl_files = glob.glob(os.path.join(search_dir, '*.pkl')) + glob.glob(os.path.join(search_dir, '*.joblib'))
        for p in pkl_files:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                artifacts['models'][name] = joblib.load(p)
            except Exception:
                try:
                    with open(p, 'rb') as fh:
                        artifacts['models'][name] = fh.read()
                except Exception:
                    pass

        # search for csvs that might be processed data
        csv_files = glob.glob(os.path.join(search_dir, '*processed*.csv')) + glob.glob(os.path.join(search_dir, '*processed*.txt'))
        csv_files += glob.glob(os.path.join(search_dir, '*.csv'))
        for p in csv_files:
            name = os.path.splitext(os.path.basename(p))[0]
            artifacts['data'][name] = p
    except Exception:
        pass
    return artifacts


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def compute_kpis(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_simple_models(df, target_col, features=None, test_size=0.2, random_state=42, preloaded_models=None):
    """Train simple LR and RF baselines and (if provided) evaluate preloaded models on the same test split.

    preloaded_models: dict mapping name->model objects (as returned by load_saved_artifacts)
    """
    X = df[features] if features is not None else df[numeric_cols(df)].drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)

    preds = {}
    preds['LinearRegression'] = lr.predict(X_test)
    preds['RandomForest'] = rf.predict(X_test)

    # Evaluate any preloaded models on the same test split if possible
    if preloaded_models:
        for name, mdl in preloaded_models.items():
            try:
                p = predict_with_model(mdl, X_test, model_name=name)
                # ensure 1d
                p = np.asarray(p).reshape(-1,)
                if p.shape[0] == X_test.shape[0]:
                    preds[name] = p
            except Exception:
                # skip models that can't be applied to this feature set
                continue

    kpis = {name: compute_kpis(y_test, p) for name, p in preds.items()}
    return {'models': {'lr': lr, 'rf': rf}, 'X_test': X_test, 'y_test': y_test, 'preds': preds, 'kpis': kpis}


def predict_with_model(model, X, model_name=None, date_col=None):
    """Attempt to predict using a saved model object. Handles sklearn-like, xgboost.Booster, and Prophet where possible.

    - model: loaded object from joblib/pickle
    - X: DataFrame or 2D array
    - model_name: optional string filename to help heuristics
    - date_col: for prophet-style models, name of date column if needed
    Returns: 1D numpy array of predictions or raises Exception
    """
    # sklearn-like models
    if hasattr(model, 'predict') and not (Prophet and isinstance(model, Prophet)):
        try:
            return model.predict(X)
        except Exception:
            # try numpy array
            try:
                return model.predict(np.asarray(X))
            except Exception:
                pass

    # xgboost.Booster
    if _HAS_XGBOOST and hasattr(xgb, 'DMatrix') and isinstance(model, getattr(xgb, 'Booster', object)):
        try:
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)
        except Exception:
            # try numpy
            dmat = xgb.DMatrix(np.asarray(X))
            return model.predict(dmat)

    # Prophet model (expects DataFrame with ds column)
    if _HAS_PROPHET and Prophet and isinstance(model, Prophet):
        # X must contain 'ds' column: if X is a DataFrame, try to build ds
        if isinstance(X, pd.DataFrame):
            if 'ds' not in X.columns:
                if date_col and date_col in X.columns:
                    future = pd.DataFrame({'ds': pd.to_datetime(X[date_col])})
                else:
                    # if index appears to be datetime-like
                    try:
                        future = pd.DataFrame({'ds': pd.to_datetime(X.index)})
                    except Exception:
                        raise ValueError('Prophet model requires a date (ds) column or datetime index')
            else:
                future = X[['ds']].copy()
            pred = model.predict(future)
            return pred['yhat'].values

    # fallback: attempt joblib-like loading/unwrapping
    if hasattr(model, 'predict'):
        return model.predict(X)

    raise RuntimeError('Unable to predict with provided model. Missing predict method or unsupported model type.')


def get_feature_importances_from_saved_model(model, feature_names=None):
    """Extract feature importances from a saved model if possible.
    Returns DataFrame with columns ['feature','importance'] or None.
    """
    # sklearn-style
    if hasattr(model, 'feature_importances_'):
        importances = getattr(model, 'feature_importances_')
        names = feature_names if feature_names is not None else [f'f{i}' for i in range(len(importances))]
        return pd.DataFrame({'feature': names, 'importance': importances}).sort_values('importance', ascending=False)

    # xgboost
    if _HAS_XGBOOST and isinstance(model, getattr(xgb, 'Booster', object)):
        try:
            score = model.get_score(importance_type='weight')
            items = [{'feature': k, 'importance': v} for k, v in score.items()]
            df_imp = pd.DataFrame(items).sort_values('importance', ascending=False)
            if df_imp.empty and feature_names is not None:
                # maybe trained xgb scikit API
                try:
                    arr = model.feature_importances_
                    return pd.DataFrame({'feature': feature_names, 'importance': arr}).sort_values('importance', ascending=False)
                except Exception:
                    pass
            return df_imp
        except Exception:
            pass

    # not available
    return None


# ----- lightweight time-series and forecasting helpers -----
def aggregate_timeseries(df, date_col, value_col, freq='M', agg='sum'):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    if freq == 'D':
        sdf = s.set_index(date_col).resample('D').sum().reset_index()
    elif freq == 'M':
        sdf = s.set_index(date_col).resample('M').sum().reset_index()
    elif freq == 'Y':
        sdf = s.set_index(date_col).resample('Y').sum().reset_index()
    else:
        sdf = s.set_index(date_col).resample(freq).sum().reset_index()
    sdf = sdf.rename(columns={date_col: 'ds', value_col: 'y'})
    return sdf


def make_lag_features(ts_df, lags=(1,7), rolling_windows=(3,7)):
    df = ts_df.copy().set_index('ds')
    for lag in range(1, max(lags)+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(window=w, min_periods=1).mean()
    df = df.dropna().reset_index()
    return df


def train_xgb_forecast(ts_df, horizon=12, params=None):
    # prepare lag features
    df_feats = make_lag_features(ts_df, lags=range(1,8), rolling_windows=(3,7))
    feature_cols = [c for c in df_feats.columns if c not in ['ds','y']]
    X = df_feats[feature_cols]
    y = df_feats['y']
    if _HAS_XGBOOST:
        model = xgb.XGBRegressor(n_estimators=params.get('n_estimators',100), max_depth=params.get('max_depth',3), learning_rate=params.get('learning_rate',0.1), random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=params.get('n_estimators',100), random_state=0)
    model.fit(X, y)

    # iterative forecasting using last rows
    last = ts_df.set_index('ds').copy()
    preds = []
    cur = last.copy()
    for i in range(horizon):
        # build features from cur
        recent = cur['y']
        feat = {}
        for lag in range(1,8):
            feat[f'lag_{lag}'] = recent.iloc[-lag] if len(recent) >= lag else recent.mean()
        for w in (3,7):
            feat[f'roll_mean_{w}'] = recent.shift(1).rolling(window=w, min_periods=1).mean().iloc[-1]
        Xp = pd.DataFrame([feat])
        p = model.predict(Xp)[0]
        # advance
        next_idx = cur.index[-1] + (cur.index[-1] - cur.index[-2]) if len(cur) > 1 else cur.index[-1] + pd.Timedelta(days=1)
        cur.loc[next_idx] = p
        preds.append(p)
    return model, preds


def train_prophet(ts_df, periods=12, seasonality_mode='additive', changepoint_prior_scale=0.05):
    if not _HAS_PROPHET:
        raise RuntimeError('Prophet is not installed in the environment')
    m = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
    m.fit(ts_df.rename(columns={'ds':'ds','y':'y'}))
    future = m.make_future_dataframe(periods=periods, freq='M')
    f = m.predict(future)
    return m, f


def detect_default_columns(df):
    """Heuristic detection of common column roles: date, demand/quantity, price, profit, sku, trade partner."""
    cols = {c.lower(): c for c in df.columns}
    res = {'date': None, 'demand': None, 'price': None, 'profit': None, 'sku': None, 'trade_partner': None}
    # date
    for k in ['date', 'order_date', 'ds', 'timestamp']:
        if k in cols:
            res['date'] = cols[k]
            break
    # demand/quantity
    for k in ['quantity', 'qty', 'demand', 'units', 'inventory', 'sales', 'volume']:
        for col in df.columns:
            if k in col.lower():
                res['demand'] = col
                break
        if res['demand']:
            break
    # price
    for col in df.columns:
        if 'price' in col.lower() or 'list_price' in col.lower():
            res['price'] = col
            break
    # profit / dcm
    for col in df.columns:
        if any(t in col.lower() for t in ['profit', 'dcm', 'margin']):
            res['profit'] = col
            break
    # sku / product
    for col in df.columns:
        if any(t in col.lower() for t in ['sku', 'product', 'item', 'material', 'partno']):
            res['sku'] = col
            break
    # trade partner / customer
    for col in df.columns:
        if any(t in col.lower() for t in ['partner', 'customer', 'dealer', 'distributor', 'trade']):
            res['trade_partner'] = col
            break
    return res



def main():
    st.set_page_config(page_title='Whirlpool AI Insights', layout='wide', initial_sidebar_state='expanded')
    # suppress warnings in console for cleaner output
    warnings.filterwarnings('ignore')

    st.markdown("""
    <style>
    .block-container{padding:1rem 2rem}
    .stSidebar .css-1d391kg {background: #ffffff}
    </style>
    """, unsafe_allow_html=True)

    st.title('Whirlpool — AI Insights')
    st.caption('Minimal, light-themed interactive dashboard for analytics and price optimization')

    # Sidebar
    st.sidebar.header('Controls')
    uploaded_file = st.sidebar.file_uploader('Upload CSV/Excel (optional)', type=['csv', 'xlsx', 'xls'])
    prefer_saved = st.sidebar.checkbox('Prefer saved notebook artifacts (models/data) if available', value=False)
    force_retrain = st.sidebar.checkbox('Force local retrain (ignore saved artifacts)', value=True)
    notebook_only = st.sidebar.checkbox('Notebook-only mode (use saved artifacts only, no retrain)', value=False)
    page = st.sidebar.radio('Pages', ['Dashboard Overview', 'Forecasting Overview', 'Feature Importance', 'Data Insights', 'Price Optimization'])
    st.sidebar.markdown('---')
    st.sidebar.write('Dataset & Filters')
    use_cluster = st.sidebar.checkbox('Use clustered dataset (if available)', value=True)

    df = None
    try:
        # Determine whether to actually use saved artifacts (allow forcing local retrain)
        prefer_saved_effective = prefer_saved and (not force_retrain)
        # If user prefers saved artifacts and there's a processed CSV, use that as default
        saved = load_saved_artifacts('.') if prefer_saved_effective else {'models': {}, 'data': {}}
        df = None
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        else:
            # If user opts to use clustered dataset and it exists, load it
            if use_cluster and os.path.exists('ML_Clustered_Database_Horizon_Global_Consulting.csv'):
                df = pd.read_csv('ML_Clustered_Database_Horizon_Global_Consulting.csv')
            else:
                # check for a processed csv first when preferring saved artifacts
                if prefer_saved and saved.get('data'):
                    # pick a likely candidate: contains 'processed' or the first CSV available
                    candidates = [p for k, p in saved['data'].items() if 'processed' in k.lower()]
                    use_path = candidates[0] if candidates else next(iter(saved['data'].values()), None)
                    if use_path:
                        try:
                            df = pd.read_csv(use_path)
                        except Exception:
                            df = load_data(None)
                    else:
                        df = load_data(None)
                else:
                    df = load_data(None)
    except FileNotFoundError:
        st.error('Default dataset not found in workspace. Please upload a CSV file.')
        return
    except Exception as e:
        st.error(f'Error loading dataset: {e}')
        return

    st.sidebar.markdown(f'Data rows: {df.shape[0]} | cols: {df.shape[1]}')

    # common selectors
    num_cols = numeric_cols(df)
    cat_cols = categorical_cols(df)

    # detect sensible defaults so user doesn't have to pick every time
    defaults = detect_default_columns(df)

    # If user asked to prefer saved models, try to extract them for use later
    saved_models = saved.get('models', {}) if 'saved' in locals() else {}

    if page == 'Dashboard Overview':
        st.header('Overview — Forecast & Price Optimization')
        st.write('Upload your CSV (use the sidebar). Overview will show side-by-side Prophet (time-series) and price optimization for a selectable product.')

        if uploaded_file is None and df is None:
            st.info('Please upload a CSV in the sidebar to begin.')
        else:
            # minimal controls for overview
            # use detected defaults when available to avoid forcing user selection
            date_options = [c for c in df.columns if 'date' in c.lower()] + [c for c in df.columns]
            date_default = defaults.get('date') if defaults.get('date') in date_options else date_options[0]
            date_col = st.selectbox('Date column', date_options, index=date_options.index(date_default))
            demand_default = defaults.get('demand') if defaults.get('demand') in num_cols else (num_cols[0] if num_cols else None)
            demand_col = st.selectbox('Demand / quantity column', num_cols, index=num_cols.index(demand_default) if demand_default in num_cols else 0)
            sku_col = None
            if cat_cols:
                sku_default = defaults.get('sku') if defaults.get('sku') in cat_cols else None
                sku_opt = [None] + cat_cols
                sku_col = st.selectbox('SKU / product column (optional)', sku_opt, index=sku_opt.index(sku_default) if sku_default in sku_opt else 0)
            price_default = defaults.get('price') if defaults.get('price') in num_cols else (num_cols[0] if num_cols else None)
            price_col = st.selectbox('Price column (for optimization)', [c for c in num_cols], index=num_cols.index(price_default) if price_default in num_cols else 0)
            profit_default = defaults.get('profit') if defaults.get('profit') in num_cols and defaults.get('profit') != price_col else (next((c for c in num_cols if c != price_col), price_col) if num_cols else None)
            profit_col = st.selectbox('DCM (Direct Contribution Margin) column (for optimization)', [c for c in num_cols if c != price_col] or [price_col], index=([c for c in num_cols if c != price_col] or [price_col]).index(profit_default) if profit_default in ( [c for c in num_cols if c != price_col] or [price_col]) else 0)

            # forecasting tuning
            st.markdown('### Forecast tuning')
            prop_seasonality = st.selectbox('Prophet seasonality mode', ['additive', 'multiplicative'], index=0)
            prop_changepoint = st.slider('Prophet changepoint prior scale', 0.001, 0.5, 0.05)
            xgb_estimators = st.slider('XGBoost / RF n_estimators', 10, 500, 100)
            xgb_depth = st.slider('XGBoost max_depth', 1, 10, 3)

            # seasonality/date range selectors
            st.markdown('### Seasonality & date range')
            # ensure date parsed
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                st.warning('Could not parse the selected date column. Forecasting may fail.')

            min_date = df[date_col].min()
            max_date = df[date_col].max()
            drange = st.date_input('Select date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)

            # choose product for price optimization (random default)
            product_value = None
            if sku_col:
                unique_skus = df[sku_col].dropna().unique().tolist()
                # prefer detected sku value if present
                sku_default_value = None
                if defaults.get('sku') and defaults.get('sku') == sku_col and len(unique_skus) > 0:
                    sku_default_value = unique_skus[0]
                product_value = st.selectbox('Select product for price optimization', ['<random>'] + unique_skus, index=0 if sku_default_value is None else (['<random>'] + unique_skus).index(sku_default_value))
                if product_value == '<random>' and unique_skus:
                    import random
                    product_value = random.choice(unique_skus)

            # aggregation frequency for price optimization (Overview)
            agg_freq = st.selectbox('Aggregate by (Overview)', ['M', 'Y'], index=0)

            # trade partner filter defaults for Overview (safe defaults so variables exist)
            trade_partner_col = defaults.get('trade_partner') if defaults.get('trade_partner') in cat_cols else None
            trade_partner_value = None
            if trade_partner_col:
                tp_vals = df[trade_partner_col].dropna().unique().tolist()
                if tp_vals:
                    trade_partner_value = st.selectbox('Select trade partner (optional)', ['<all>'] + tp_vals)

            # run forecasts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Prophet forecast (demand)')
                try:
                    ts = aggregate_timeseries(df[(df[date_col] >= pd.to_datetime(drange[0])) & (df[date_col] <= pd.to_datetime(drange[1]))], date_col, demand_col, freq='M')
                    m, forecast_df = train_prophet(ts, periods=12, seasonality_mode=prop_seasonality, changepoint_prior_scale=prop_changepoint)
                    fig_p = px.line(forecast_df, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title='Prophet forecast (yhat, lower, upper)')
                    st.plotly_chart(fig_p, use_container_width=True)
                except Exception as e:
                    st.error(f'Prophet forecast failed: {e}')

            with col2:
                st.subheader('Price optimization (sample product)')
                try:
                    # filter product and trade partner
                    pop = df.copy()
                    if sku_col and product_value:
                        pop = pop[pop[sku_col] == product_value]
                    if trade_partner_col and trade_partner_value and trade_partner_value != '<all>':
                        pop = pop[pop[trade_partner_col] == trade_partner_value]
                    # aggregate monthly
                    resample_rule = 'M' if agg_freq == 'M' else 'Y'
                    pop_agg = pop.set_index(pd.to_datetime(pop[date_col])).resample(resample_rule).agg({price_col:'mean', profit_col:'sum'})
                    pop_agg = pop_agg.dropna()
                    if pop_agg.empty:
                        st.warning('Not enough data for selected product to run price optimization.')
                    else:
                        X = pop_agg[[price_col]].values.reshape(-1,1)
                        y = pop_agg[profit_col].values
                        # simple model choice
                        if _HAS_XGBOOST:
                            price_model = xgb.XGBRegressor(n_estimators=xgb_estimators, max_depth=xgb_depth, random_state=0)
                        else:
                            price_model = RandomForestRegressor(n_estimators=xgb_estimators, random_state=0)
                        price_model.fit(X, y)
                        grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
                        preds = price_model.predict(grid)
                        opt = grid[np.argmax(preds)][0]
                        viz = pd.DataFrame({'price':grid.flatten(), 'pred_profit':preds})
                        fig_price = px.line(viz, x='price', y='pred_profit', title=f'Price optimization for product {product_value}')
                        fig_price.add_scatter(x=[opt], y=[preds.max()], mode='markers', name='Optimal', marker={'size':12, 'color':'green'})
                        st.plotly_chart(fig_price, use_container_width=True)
                        st.metric('Optimal price (est.)', f'{float(opt):.2f}')
                except Exception as e:
                    st.error(f'Price optimization failed: {e}')

    elif page == "Forecasting Overview":
        st.header("Forecasting Overview — Global Trend & Seasonality (Prophet)")

        st.markdown("""
        <div style="padding:16px; border-radius:10px; background-color:#111827; border:1px solid #1f2937;">
        <p style="color:#E5E7EB;">
        This page uses <b>Meta’s Prophet</b> to extract the <b>overall demand trend</b> and 
        <b>yearly seasonality</b> across the entire dataset. This is not a SKU-level forecast.  
        Prophet here acts only as a <b>feature generator</b> for the XGBoost price model and as
        a visual context tool for Alejandro.
        </p>

        <p style="color:#E5E7EB;">
        The goal is simply to understand whether Whirlpool's demand overall follows a 
        predictable seasonal rhythm (e.g., peaks in certain months), or if it behaves irregularly.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Detect date + demand cols
        # -----------------------------
        date_col = defaults.get("date")
        demand_col = defaults.get("demand")

        if date_col is None or demand_col is None:
            st.error("A date column and demand/quantity column are required.")
            st.stop()

        # Convert date
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            st.error("Could not parse the date column.")
            st.stop()

        # Global aggregated demand (monthly)
        ts = df[[date_col, demand_col]].copy()
        ts = ts.rename(columns={date_col: "ds", demand_col: "y"})
        ts = ts.dropna()

        # Aggregate to monthly (to calm noise)
        ts = ts.set_index("ds").resample("M").sum().reset_index()

        # Remove zeros or negatives (Prophet hates them)
        ts = ts[ts["y"] > 0]

        # -----------------------------
        # Run Prophet (trend + yearly only)
        # -----------------------------
        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive"
            )
            m.fit(ts)
            future = m.make_future_dataframe(periods=0, freq="M")
            fcst = m.predict(future)

            # Trend + yearly seasonality plots
            st.subheader("Trend and Yearly Seasonality")

            fig_components = m.plot_components(fcst)
            st.pyplot(fig_components)

        except Exception as e:
            st.error(f"Prophet failed: {e}")
            st.stop()

        # -----------------------------
        # Final explanation box
        # -----------------------------
        st.markdown("""
        <div style="padding:14px; border-radius:10px; background-color:#111827; border:1px solid #1f2937;">
        
        <p style="color:#E5E7EB;">
        <b>How this is used:</b><br>
        The yearly seasonality extracted here is included as a feature in the 
        <b>XGBoost price–quantity model</b>, giving the model awareness of 
        predictable demand fluctuations throughout the year.
        </p>

        <ul style="color:#E5E7EB; margin-left:20px;">
            <li>The <b>trend</b> shows long-term demand direction.</li>
            <li>The <b>yearly seasonality</b> shows recurring monthly patterns.</li>
            <li>This helps the model distinguish true price effects from seasonal demand peaks.</li>
        </ul>

        <p style="color:#34D399; font-weight:600;">
        💡 This page is global context — not a forecast.  
        It helps Alejandro understand the demand environment that the price model is operating in.
        </p>

        </div>
        """, unsafe_allow_html=True)



    elif page == 'Data Insights':
        st.header('Data Insights')
        st.write('Trend, anomaly detection, and categorical breakdowns.')
        st.subheader('Trend')
        date_candidates = [c for c in df.columns if 'date' in c.lower()]
        if date_candidates:
            date_col = st.selectbox('Choose date column', date_candidates)
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df_sorted = df.sort_values(date_col)
                metric = st.selectbox('Choose metric to trend', numeric_cols(df_sorted), index=0)
                fig = px.line(df_sorted, x=date_col, y=metric, title=f'{metric} over time')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning('Could not parse date column: ' + str(e))
        else:
            st.info('No date-like column detected. Showing index-based trend for a selected numeric column.')
            if num_cols:
                metric = st.selectbox('Choose metric to trend (index-based)', num_cols)
                fig = px.line(df.reset_index(), x='index', y=metric, title=f'{metric} over records')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader('Anomaly Detection (simple z-score)')
        if num_cols:
            metric = st.selectbox('Choose metric for anomaly detection', num_cols, index=0)
            series = df[metric].fillna(method='ffill').fillna(method='bfill')
            z = (series - series.mean()) / (series.std() + 1e-9)
            anomalies = z.abs() > st.slider('Z-score threshold', 2.0, 5.0, 3.0)
            out_df = pd.DataFrame({'value': series, 'zscore': z, 'anomaly': anomalies})
            fig = px.scatter(out_df.reset_index(), x='index', y='value', color='anomaly', title=f'Anomalies in {metric}')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader('Categorical breakdown (pie)')
        if cat_cols:
            cat = st.selectbox('Choose categorical column for pie', cat_cols)
            counts = df[cat].value_counts().reset_index()
            counts.columns = [cat, 'count']
            fig = px.pie(counts, values='count', names=cat, title=f'Distribution of {cat}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No categorical columns detected to show pie chart.')

    elif page == 'Price Optimization': 
        st.header("Price Optimization (XGBoost – Point & Quantile)")

        # ---- Required columns ----
        required_cols = ['sku', 'trade_partner', 'price_final', 'dcm', 'quantity']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns for price optimization: {missing}")
            return

        # ---- Select SKU ----
        sku_list = sorted(df['sku'].dropna().unique())
        selected_sku = st.selectbox("Select SKU", sku_list)
        df_sku = df[df['sku'] == selected_sku].copy()

        # ---- Optional trade partner filter ----
        tp_list = ['All'] + sorted(df_sku['trade_partner'].dropna().unique())
        selected_tp = st.selectbox("Select Trade Partner (optional)", tp_list)
        if selected_tp != 'All':
            df_sku = df_sku[df_sku['trade_partner'] == selected_tp]

        # Need enough data
        if df_sku.shape[0] < 10:
            st.warning("Not enough observations for this (SKU, trade partner) selection.")
            return

        # ---- Prepare base_row for price grid ----
        base_row = df_sku.iloc[[0]].copy()

        # approximate variable cost
        if 'mean_total_variable_cost' in df_sku.columns:
            cost = float(base_row['mean_total_variable_cost'].iloc[0])
        else:
            tmp = df_sku[df_sku['quantity'] > 0].copy()
            if not tmp.empty:
                tmp['approx_cost'] = tmp['price_final'] - tmp['dcm'] / tmp['quantity']
                cost = float(tmp['approx_cost'].mean())
            else:
                cost = float(base_row['price_final'].iloc[0]) * 0.7

        actual_price = float(base_row['price_final'].iloc[0])
        p_min = max(actual_price * 0.5, cost)
        p_max = actual_price * 1.5
        price_grid = np.linspace(p_min, p_max, 150)

        # ---- Feature Set ----
        candidate_features = [
            "price_final",
            "mean_price_final",
            "mean_price_list",
            "forecasted_demand",
            "price_final_var_coeff",
            "quantity_var_coeff",
            "real_inventory"
        ]
        features = [c for c in candidate_features if c in df_sku.columns]
        if "price_final" not in features:
            features.append("price_final")

        # Clean df
        df_model = df_sku.dropna(subset=features + ['quantity']).copy()
        if df_model.shape[0] < 10:
            st.warning("Not enough clean rows with all required features for modeling.")
            return

        X = df_model[features]
        y = df_model['quantity']
        has_inv = 'real_inventory' in df_model.columns


        # ================================================================
        # 1. HISTORICAL PERFORMANCE (TIME SERIES)
        # ================================================================
        st.subheader("1. Historical Performance Over Time")

        # date handling
        if "year" in df_sku.columns and "iso_week" in df_sku.columns:
            df_sku["date"] = pd.to_datetime(
                df_sku["year"].astype(str) + df_sku["iso_week"].astype(str) + "1",
                format="%G%V%u",
                errors="coerce"
            )
        elif "date" not in df_sku.columns:
            st.error("No usable date column found (need date, or year + iso_week).")
            return

        df_sku = df_sku.sort_values("date")

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig_hist = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.33, 0.33, 0.33],
        )

        # Top: DCM
        fig_hist.add_trace(
            go.Scatter(
                x=df_sku["date"],
                y=df_sku["dcm"],
                mode="lines+markers",
                name="DCM",
                line=dict(color="lightgreen", width=2),
                marker=dict(color="lightgreen", size=5),
                hovertemplate="Date: %{x}<br>DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )
        fig_hist.update_yaxes(title_text="DCM (MXN)", row=1, col=1)

        # Middle: Quantity
        fig_hist.add_trace(
            go.Scatter(
                x=df_sku["date"],
                y=df_sku["quantity"],
                mode="lines+markers",
                name="Quantity",
                line=dict(color="lightblue", width=2),
                marker=dict(color="lightblue", size=5),
                hovertemplate="Date: %{x}<br>Quantity: %{y}<extra></extra>"
            ),
            row=2, col=1
        )
        fig_hist.update_yaxes(title_text="Quantity", row=2, col=1)

        # Bottom: Price
        fig_hist.add_trace(
            go.Scatter(
                x=df_sku["date"],
                y=df_sku["price_final"],
                mode="lines+markers",
                name="Price",
                line=dict(color="grey", width=2),
                marker=dict(color="lightgrey", size=5),
                hovertemplate="Date: %{x}<br>Price: %{y:.2f} MXN<extra></extra>"
            ),
            row=3, col=1
        )
        fig_hist.update_yaxes(title_text="Price (MXN)", row=3, col=1)
        fig_hist.update_xaxes(title_text="Date", row=3, col=1)

        fig_hist.update_layout(
            height=800,
            title="Historical Time Series (DCM → Quantity → Price)",
            template="simple_white",
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)


        # ================================================================
        # 2. POINT ESTIMATE MODEL (XGBOOST)
        # ================================================================
        st.subheader("2. Point Estimate Model (XGBoost)")

        import xgboost as xgb

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
            inv = df_grid_point['real_inventory'].values
            q_pred_point = np.round(np.minimum(np.maximum(q_pred_point, 0), inv))
        else:
            q_pred_point = np.round(np.maximum(q_pred_point, 0))

        dcm_point = (df_grid_point["price_final"].values - cost) * q_pred_point
        best_idx_point = int(np.argmax(dcm_point))
        best_price_point = float(df_grid_point["price_final"].iloc[best_idx_point])
        best_dcm_point = float(dcm_point[best_idx_point])

        # ---- two stacked plots: DCM (top), quantity (bottom) ----
        fig_point = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45]
        )

        # --- TOP: DCM w/ historical points ---
        fig_point.add_trace(
            go.Scatter(
                x=df_sku['price_final'],
                y=df_sku['dcm'],
                mode="markers",
                name="Historical",
                marker=dict(color="white", size=6, opacity=0.8),
                hovertemplate="Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        fig_point.add_trace(
            go.Scatter(
                x=price_grid,
                y=dcm_point,
                mode="lines",
                name="Predicted DCM (Point Model)",
                line=dict(color="blue", width=3),
                hovertemplate="Price: %{x:.2f} MXN<br>Pred DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        # optimal
        fig_point.add_trace(
            go.Scatter(
                x=[best_price_point],
                y=[best_dcm_point],
                mode="markers",
                name="Optimal Price",
                marker=dict(color="green", size=11),
                hovertemplate="Optimal Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        # --- BOTTOM: Quantity w/ historical ---
        fig_point.add_trace(
            go.Scatter(
                x=df_sku['price_final'],
                y=df_sku['quantity'],
                mode="markers",
                name="Historical",
                showlegend=False,
                marker=dict(color="white", size=6, opacity=0.8),
                hovertemplate="Price: %{x:.2f} MXN<br>Quantity: %{y}<extra></extra>"
            ),
            row=2, col=1
        )

        fig_point.add_trace(
            go.Scatter(
                x=price_grid,
                y=q_pred_point,
                mode="lines",
                name="Predicted Quantity",
                line=dict(color="grey", width=3, dash="dash"),
                hovertemplate="Price: %{x:.2f} MXN<br>Pred Qty: %{y}<extra></extra>"
            ),
            row=2, col=1
        )

        fig_point.update_xaxes(title_text="Price (MXN)", row=2, col=1)
        fig_point.update_yaxes(title_text="DCM (MXN)", row=1, col=1)
        fig_point.update_yaxes(title_text="Quantity", row=2, col=1)

        fig_point.update_layout(
            height=650,
            title_text="Point Estimate Model – Price Grid",
            template="simple_white"
        )

        st.metric("Optimal Price (Point Estimate)", f"{best_price_point:,.2f} MXN")
        st.plotly_chart(fig_point, use_container_width=True)


        # ================================================================
        # 3. QUANTILE ESTIMATE MODEL (XGBOOST)
        # ================================================================
        st.subheader("3. Quantile Estimate Model (XGBoost)")

        quantiles = {"lower": 0.05, "median": 0.5, "upper": 0.95}
        q_models = {}

        params_base = {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }

        # Train 3 quantile models
        for name_q, alpha in quantiles.items():
            params_q = params_base.copy()
            params_q.update({
                "objective": "reg:quantileerror",
                "quantile_alpha": alpha,
            })
            m_q = xgb.XGBRegressor(**params_q)
            m_q.fit(X, y)
            q_models[name_q] = m_q

        qL = q_models["lower"].predict(X_grid_point)
        qM = q_models["median"].predict(X_grid_point)
        qU = q_models["upper"].predict(X_grid_point)

        if has_inv:
            inv = df_grid_point['real_inventory'].values
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

        best_idx_quant = int(np.argmax(dcm_M))
        best_price_quant = float(price_grid[best_idx_quant])
        best_dcm_quant = float(dcm_M[best_idx_quant])

        fig_quant = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45]
        )

        # --- TOP: DCM band + median + HISTORICAL ---
        fig_quant.add_trace(
            go.Scatter(
                x=df_sku['price_final'],
                y=df_sku['dcm'],
                mode="markers",
                name="Historical",
                marker=dict(color="white", size=6, opacity=0.8),
                hovertemplate="Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        # lower band
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=dcm_L,
                mode="lines",
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )

        # upper band
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=dcm_U,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0,120,255,0.18)",
                line=dict(width=0),
                name="Uncertainty Band"
            ),
            row=1, col=1
        )

        # median
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=dcm_M,
                mode="lines",
                name="Median DCM",
                line=dict(color="blue", width=3),
                hovertemplate="Price: %{x:.2f} MXN<br>Median DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        # optimal price
        fig_quant.add_trace(
            go.Scatter(
                x=[best_price_quant],
                y=[best_dcm_quant],
                mode="markers",
                name="Optimal Price (Median)",
                marker=dict(color="green", size=11),
                hovertemplate="Optimal Price: %{x:.2f} MXN<br>DCM: %{y:.2f} MXN<extra></extra>"
            ),
            row=1, col=1
        )

        # --- BOTTOM: Quantity band + median + HISTORICAL ---
        fig_quant.add_trace(
            go.Scatter(
                x=df_sku['price_final'],
                y=df_sku['quantity'],
                mode="markers",
                name="Historical",
                showlegend=False,
                marker=dict(color="white", size=6, opacity=0.8),
                hovertemplate="Price: %{x:.2f} MXN<br>Qty: %{y}<extra></extra>"
            ),
            row=2, col=1
        )

        # lower band
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=qL,
                mode="lines",
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=1
        )
        # upper band
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=qU,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0,120,255,0.18)",
                line=dict(width=0),
                showlegend=False
            ),
            row=2, col=1
        )

        # median qty
        fig_quant.add_trace(
            go.Scatter(
                x=price_grid,
                y=qM,
                mode="lines",
                name="Median Quantity",
                line=dict(color="grey", width=3, dash="dash"),
                hovertemplate="Price: %{x:.2f} MXN<br>Median Qty: %{y}<extra></extra>"
            ),
            row=2, col=1
        )

        fig_quant.update_xaxes(title_text="Price (MXN)", row=2, col=1)
        fig_quant.update_yaxes(title_text="DCM (MXN)", row=1, col=1)
        fig_quant.update_yaxes(title_text="Quantity", row=2, col=1)

        fig_quant.update_layout(
            height=650,
            title_text="Quantile Estimate Model – Price Grid (DCM & Quantity)",
            template="simple_white"
        )

        st.metric("Optimal Price (Quantile Median)", f"{best_price_quant:,.2f} MXN")
        st.plotly_chart(fig_quant, use_container_width=True)

        st.markdown("""
<div style="padding:18px; border-radius:10px; background-color:#111827; border:1px solid #1f2937;">
<h3 style="color:white; margin-top:0;">🔍 How the Model Arrives at Its Price Recommendation</h3>

<p style="color:#E5E7EB;">
<b style="color:#93C5FD;">1. It reviews the historical price–volume behaviour</b><br>
The model starts by analysing how this SKU has actually behaved in the past — how quantity reacts to price changes, how margins move, and how stable demand usually is.
</p>

<p style="color:#E5E7EB;">
<b style="color:#93C5FD;">2. It identifies the underlying pattern</b><br>
It picks up patterns such as:<br>
• typical quantity levels<br>
• how sensitive the SKU is to price<br>
• how volatile or stable the demand normally is<br>
These patterns form the baseline for estimating how the product is likely to respond today.
</p>

<p style="color:#E5E7EB;">
<b style="color:#93C5FD;">3. It runs price scenarios across a full range</b><br>
For each possible price level, the model forecasts:<br>
• expected units sold<br>
• the resulting DCM (margin contribution)<br>
This produces the full price–DCM curve used for decision-making.
</p>

<p style="color:#E5E7EB;">
<b style="color:#93C5FD;">4. It selects the price with the strongest margin outlook</b><br>
• The <i>Point Model</i> takes the average expected outcome.<br>
• The <i>Quantile Model</i> adds an uncertainty band, showing best-case and worst-case ranges.<br>
Both approaches highlight where DCM is expected to peak.
</p>

<p style="color:#E5E7EB;">
<b style="color:#93C5FD;">5. It validates the curve against real behaviour</b><br>
Historical observations (white dots) are plotted on the same graph so you can visually check whether the model’s predicted shape aligns with actual outcomes for this SKU.
</p>

<p style="color:#34D399; font-weight:600;">
💡 In short: the model recommends the price where expected margin is highest, based on real price–volume history and current conditions.
</p>

</div>
""", unsafe_allow_html=True)


    # (Artifact Diagnostics removed to simplify UI — dashboard will train models locally by default)

    # Footer: show sample of data if requested
    if st.sidebar.checkbox('Show data sample'):
        st.subheader('Data sample')
        st.dataframe(df.head(200))


if __name__ == '__main__':
    main()