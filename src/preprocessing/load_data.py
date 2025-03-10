import os
import pandas as pd


DATA_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/auto-mpg.data")
    )

def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:

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

    df["speed_category"] = pd.cut(
        df["acceleration"], 5, labels=range(5)
    )

    return df