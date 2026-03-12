import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecasts(data, forecast_df):

    os.makedirs("results/forecast_plots", exist_ok=True)

    countries = forecast_df["Country"].unique()

    target_column = "Inbound-Total Arrival(overnight stay)( In thousands)"

    # Individual country plots
    for country in countries:

        historical = data[data["Country"] == country]
        forecast = forecast_df[forecast_df["Country"] == country]

        plt.figure(figsize=(8,5))

        plt.plot(
            historical["Year"],
            historical[target_column],
            label="Historical",
            marker="o"
        )

        plt.plot(
            forecast["Year"],
            forecast["Predicted_Tourism"],
            label="Forecast",
            marker="o"
        )

        plt.title(f"Tourism Forecast - {country}")
        plt.xlabel("Year")
        plt.ylabel("Tourism Arrivals")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"results/forecast_plots/{country}_forecast.png")

        plt.close()


def plot_dashboard(data, forecast_df):

    os.makedirs("results", exist_ok=True)

    target_column = "Inbound-Total Arrival(overnight stay)( In thousands)"

    plt.figure(figsize=(12,7))

    countries = forecast_df["Country"].unique()

    # Plot multiple countries on one chart
    for country in countries[:6]:

        historical = data[data["Country"] == country]
        forecast = forecast_df[forecast_df["Country"] == country]

        combined_years = list(historical["Year"]) + list(forecast["Year"])
        combined_values = list(historical[target_column]) + list(forecast["Predicted_Tourism"])

        plt.plot(combined_years, combined_values, marker="o", label=country)

    plt.title("Tourism Forecast Dashboard")
    plt.xlabel("Year")
    plt.ylabel("Tourism Arrivals")

    plt.axvline(x=2022, color="black", linestyle="--", label="Forecast Start")

    plt.legend()
    plt.grid(True)

    plt.savefig("results/tourism_forecast_dashboard.png")

    plt.close()


def plot_model_comparison(results_df):

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8,5))

    plt.bar(results_df["Model"], results_df["RMSE"])

    plt.title("Model Performance Comparison (RMSE)")
    plt.xlabel("Model")
    plt.ylabel("RMSE")

    plt.grid(axis="y")

    plt.savefig("results/model_comparison.png")

    plt.close()