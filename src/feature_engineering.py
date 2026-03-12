import pandas as pd


# Create new features to improve model performance
def create_features(df):

    # Tourism growth rate per country
    df["tourism_growth_rate"] = df.groupby("Country")[
        "Inbound-Total Arrival(overnight stay)( In thousands)"
    ].pct_change()

    # Expenditure per tourist
    df["expenditure_per_tourist"] = (
        df["Inbound- Expenditure(Travel)(US$ Millions)"]
        / df["Inbound-Total Arrival(overnight stay)( In thousands)"]
    )

    # COVID indicator variable
    df["COVID"] = df["Year"].apply(lambda x: 1 if 2020 <= x <= 2021 else 0)

    # Year index (helps model learn time trend for forecasting)
    df["Year_Index"] = df["Year"] - df["Year"].min()

    # Replace infinite values
    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # Fill missing values created during feature engineering
    df = df.fillna(0)

    return df