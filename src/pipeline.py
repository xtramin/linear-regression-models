import os
from src.preprocessing.load_data import load_dataset
from src.preprocessing.feature_engineering import (
    add_features,
    fill_null,
    remove_features,
)
from src.preprocessing.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split


def run_pipeline(data_path: str, train_size: float = 0.8):
    # Step 1
    df = load_dataset(data_path)

    # Step 2
    df = add_features(df)
    df = fill_null(df)
    df = remove_features(df)

    # Step 3
    df_train, df_test = train_test_split(df, random_state=42, train_size=train_size)

    preprocessor = Preprocessor()

    df_train = preprocessor.fit(
        df_train,
        log_cols=["horsepower", "weight", "mpg"],
        scale_cols=["horsepower", "weight", "acceleration", "mpg"],
    )

    df_test = preprocessor.transform(df_test)

    return df_train, df_test