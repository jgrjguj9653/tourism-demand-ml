import pandas as pd
from prophet import Prophet


def forecast_tourism_prophet(data, target_column):

    forecast_results = []

    countries = data["Country"].unique()

    for country in countries:

        country_data = data[data["Country"] == country].copy()

        # Prepare data for Prophet
        df_prophet = country_data[["Year", target_column]].copy()
        df_prophet = df_prophet.rename(columns={
            "Year": "ds",
            target_column: "y"
        })

        # Convert year to datetime
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

        # Train Prophet model
        model = Prophet()
        model.fit(df_prophet)

        # Create future years
        future = model.make_future_dataframe(periods=5, freq="YE")

        forecast = model.predict(future)

        # Extract only future years
        future_forecast = forecast[forecast["ds"].dt.year >= 2022].head(5)

        for _, row in future_forecast.iterrows():

         forecast_results.append({
    "Country": country,
    "Year": row["ds"].year,
    "Predicted_Tourism": max(0, row["yhat"])
})

    forecast_df = pd.DataFrame(forecast_results)

    return forecast_df