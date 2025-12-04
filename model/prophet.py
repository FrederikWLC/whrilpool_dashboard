from prophet import Prophet
import pandas as pd

def do_forecast(data, cutoff=None, name="General Trend"):

    if cutoff is None:
        cutoff = data["ds"].max() - pd.Timedelta(weeks=15)
    else:
        cutoff = pd.to_datetime(cutoff)
    
    train = data[data["ds"] < cutoff]
    test = data[data["ds"] >= cutoff]

    # Fit Prophet (same config as before)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.75
    )

    model.fit(train)

    # do not include training data
    if not test.empty:
        future = test[["ds"]].copy()  
    else:
        future = pd.DataFrame({
            "ds": pd.date_range(
                start=data["ds"].max() + pd.Timedelta(weeks=1),
                end=cutoff,
                freq="W-MON"
            )
        })

    forecast = model.predict(future)

    return model, train, test, forecast
