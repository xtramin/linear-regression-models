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
    # Step 1: Load the data
    df = load_dataset(data_path)

    # Step 2: Feature engineering
    df = add_features(df)
    df = fill_null(df)
    df = remove_features(df)

    # Step 3: Split the data and transform
    df_train, df_test = train_test_split(df, random_state=42, train_size=train_size)

    preprocessor = Preprocessor()

    df_train = preprocessor.fit(
        df_train,
        log_cols=["horsepower", "weight", "mpg"],
        scale_cols=["horsepower", "weight", "acceleration", "mpg"],
    )

    df_test = preprocessor.transform(df_test)

    # Step 4: Convert to Numpy
    y_train = df_train.pop("mpg").to_numpy()
    y_test = df_test.pop("mpg").to_numpy()
    X_train = df_train.to_numpy()
    X_test = df_test.to_numpy()

    return X_train, y_train, X_test, y_test