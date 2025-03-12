import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def categorize_year(year):
    if 70 <= year <= 73:
        return 0
    elif 74 <= year <= 79:
        return 1
    elif 80 <= year <= 82:
        return 2
    else:
        return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df["speed_category"] = pd.cut(df["acceleration"], 5, labels=range(5))
    df["model_time_period"] = df["model_year"].apply(categorize_year)
    df["origin_us"] = df["origin"].apply(lambda x: 1 if x == 1 else 0)
    return df


def remove_features(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(
        columns=[
            "cylinders",
            "displacement",
            "origin",
            "model_year",
            "car_name",
            "speed_category",
        ],
        inplace=True,
    )
    return df


def fill_null(df: pd.DataFrame) -> pd.DataFrame:

    mean_hp = df.groupby("speed_category")["horsepower"].mean().round(1)
    df["horsepower"] = df.apply(
        lambda row: (
            mean_hp[row["speed_category"]]
            if pd.isna(row["horsepower"])
            else row["horsepower"]
        ),
        axis=1,
    )

    return df
