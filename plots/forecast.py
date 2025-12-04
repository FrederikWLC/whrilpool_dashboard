from matplotlib import pyplot as plt

def forecast_figure(forecast, train, test, name):
     # === Plot ===
    plt.figure(figsize=(15, 6))
    plt.plot(train["ds"], train["y"], label="Train", color="black")
    if not test.empty:
        plt.plot(test["ds"], test["y"], label="Actual (Test)", color="red", linestyle="dashed")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="blue")

    # Confidence interval
    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="blue",
        alpha=0.2,
        label="Confidence Interval"
    )

    plt.title(f"Demand Forecast – {name}")
    plt.xlabel("Date")
    plt.ylabel("Scaled Demand: minmax(Quantity × Price)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()