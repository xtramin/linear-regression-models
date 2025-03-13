import os
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:

    column_names = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]

    df = pd.read_csv(filepath, names=column_names, sep=r"\s+", na_values="?")

    return df
