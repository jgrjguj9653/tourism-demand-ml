import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_data, handle_missing_values, remove_duplicates, remove_outliers_iqr
from src.feature_engineering import create_features
from src.models import get_models
from src.train_models import train_and_evaluate, train_neural_network
from src.experiment_tracker import save_results, select_best_model
from src.neural_network import TourismNet

from src.forecasting import forecast_tourism
from src.forecasting_prophet import forecast_tourism_prophet
from src.visualization import plot_forecasts, plot_dashboard, plot_model_comparison


# Load dataset
data = load_data("data/tourism_dataset.csv")


# Data preprocessing
data = handle_missing_values(data)
data = remove_duplicates(data)


# Feature engineering
data = create_features(data)


# Define target variable
target = "Inbound-Total Arrival(overnight stay)( In thousands)"


# Define feature columns
features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
features.remove(target)


X = data[features]
y = data[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Load sklearn models
models = get_models()

results = []


# Train and evaluate sklearn models
for model_name, model in models.items():

    rmse, mae, r2 = train_and_evaluate(
        model, X_train, y_train, X_test, y_test
    )

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


# Train Neural Network
input_size = X_train.shape[1]

nn_model = TourismNet(input_size)

rmse, mae, r2 = train_neural_network(
    nn_model, X_train, y_train, X_test, y_test
)

results.append({
    "Model": "Neural_Network",
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
})


# Save experiment results
results_df = save_results(results)
plot_model_comparison(results_df)

print("Model comparison chart generated.")


# Select best model
best_model = select_best_model(results_df)

print("Best Model:", best_model)


# --------------------------------
# Random Forest Forecast
# --------------------------------
target_column = "Inbound-Total Arrival(overnight stay)( In thousands)"

rf_forecast_df = forecast_tourism(data, target_column)

rf_forecast_df.to_csv("results/tourism_forecast_rf.csv", index=False)

print("Random Forest forecasting completed.")


# --------------------------------
# Prophet Forecast
# --------------------------------
prophet_forecast_df = forecast_tourism_prophet(data, target_column)

prophet_forecast_df.to_csv("results/tourism_forecast_prophet.csv", index=False)

print("Prophet forecasting completed.")


# Generate forecast plots
plot_forecasts(data, prophet_forecast_df)

# Generate dashboard visualization
plot_dashboard(data, prophet_forecast_df)

print("Forecast plots and dashboard generated.")