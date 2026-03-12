# Machine Learning Pipeline for Tourism Demand Prediction

## Project Overview

This project implements an end-to-end machine learning and forecasting pipeline to analyze and predict tourism demand across multiple countries using tourism and economic indicators.

The system performs data preprocessing, feature engineering, machine learning model comparison, neural network training, experiment tracking, forecasting, and visualization within a structured and reproducible pipeline.

The goal of the project is to evaluate multiple machine learning models for tourism demand prediction, automatically identify the best-performing model, generate tourism forecasts for future years, and provide visual insights into tourism trends and recovery patterns.

This project was originally developed as part of academic research in tourism demand analysis and later extended into a production-style machine learning pipeline.

---

## Machine Learning Models Used

The pipeline compares several regression models to identify the best model for predicting tourism demand.

Models included:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Neural Network (PyTorch)

Each model is trained and evaluated using the same dataset to ensure a fair comparison.

The pipeline automatically selects the best-performing model based on evaluation metrics.

---

## Forecasting Approach

This project includes two complementary forecasting approaches.

### Random Forest Forecasting

Random Forest forecasting uses machine learning with engineered features such as tourism indicators and economic variables.

This approach is useful because it captures complex nonlinear relationships, uses multiple explanatory variables, and provides feature-driven predictions.

Random Forest is retained in the pipeline because it was identified as the best-performing predictive model during experimentation.

---

### Prophet Time-Series Forecasting

In addition to machine learning prediction, the project also implements time-series forecasting using Facebook Prophet.

Prophet models long-term tourism trends, structural shocks such as COVID-19, and recovery patterns in tourism demand.

Prophet is particularly suitable for time-series tourism data where trends, shocks, and recovery periods play a major role.

Using both approaches allows the project to demonstrate both machine learning prediction and statistical time-series forecasting.

---

## Features of the Project

The pipeline includes the following components.

### Data Processing

* Missing value handling
* Duplicate removal
* Outlier detection using IQR

### Feature Engineering

New features are created to improve model performance, including:

* Tourism growth rate
* Expenditure per tourist

### Model Training

The system trains multiple machine learning models including a neural network implemented using PyTorch.

### Experiment Tracking

All model results are automatically saved and compared.

### Model Evaluation

Models are evaluated using standard regression metrics.

### Automatic Model Selection

The pipeline automatically identifies the best-performing model.

### Forecasting

Future tourism demand is predicted using:

* Random Forest forecasting
* Prophet time-series forecasting

### Visualization

The system automatically generates:

* Country-level forecast plots
* Tourism forecast dashboard
* Model comparison chart

---

## Project Structure

tourism-demand-ml

data
└── tourism_dataset.csv

notebooks
└── exploratory_analysis.ipynb

results
├── model_results.csv
├── model_comparison.png
├── tourism_forecast_rf.csv
├── tourism_forecast_prophet.csv
├── tourism_forecast_dashboard.png
└── forecast_plots

src
├── preprocessing.py
├── feature_engineering.py
├── models.py
├── neural_network.py
├── train_models.py
├── experiment_tracker.py
├── forecasting.py
├── forecasting_prophet.py
└── visualization.py

main.py
requirements.txt
README.md

---

## Evaluation Metrics

Model performance is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

These metrics help compare models and determine which one performs best for tourism demand prediction.

---

## Dataset

The dataset used in this project was compiled from multiple tourism statistics sources during academic research.

Due to data licensing and source restrictions, the original research dataset cannot be distributed publicly.

Instead, this repository includes a sample dataset that replicates the structure and format used in the original project.

This sample dataset allows users to run the full pipeline, test the models, and reproduce the forecasting workflow.

The pipeline expects the dataset in the following location:

data/tourism_dataset.csv

---

## How to Run the Project

Install the required dependencies:

pip install -r requirements.txt

Run the main pipeline:

python main.py

---

## Output

The pipeline will automatically:

Train all models:

* Linear Regression
* Random Forest
* Gradient Boosting
* Neural Network

Evaluate their performance and save results to:

results/model_results.csv

Generate a model comparison chart:

results/model_comparison.png

Generate tourism forecasts using:

Random Forest forecasting
Prophet time-series forecasting

Forecast outputs will be saved as:

results/tourism_forecast_rf.csv
results/tourism_forecast_prophet.csv

The pipeline will also generate visualizations including:

* Tourism forecast dashboard
* Country-level forecast plots

---

## Author

Suhas Ramesh
