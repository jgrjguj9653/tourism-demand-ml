import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def forecast_tourism(data, target_column):

    forecast_results = []

    countries = data["Country"].unique()

    for country in countries:

        country_data = data[data["Country"] == country].copy()

        # Features for forecasting
        X = country_data[["Year", "COVID", "Year_Index"]]
        y = country_data[target_column]

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        # Create future years
        future_years = pd.DataFrame({
            "Year": [2022, 2023, 2024, 2025, 2026],
            "COVID": [0, 0, 0, 0, 0]
        })

        # Create Year_Index for future data
        future_years["Year_Index"] = future_years["Year"] - data["Year"].min()

        predictions = model.predict(future_years)

        for year, pred in zip(future_years["Year"], predictions):

            forecast_results.append({
                "Country": country,
                "Year": year,
                "Predicted_Tourism": pred
            })

    forecast_df = pd.DataFrame(forecast_results)

    return forecast_df