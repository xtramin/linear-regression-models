import os
from src.preprocessing.load_data import load_dataset
from src.preprocessing.feature_engineering import (
    add_features,
    fill_null,
    remove_features,
)
from src.preprocessing.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = load_dataset()
    df = add_features(df)
    df = fill_null(df)
    df = remove_features(df)

    df_train, df_test = train_test_split(df, random_state=42, train_size=0.8)

    prep = Preprocessor()
    df_train = prep.fit(
        df_train,
        log_cols=["horsepower", "weight", "mpg"],
        scale_cols=["horsepower", "weight", "acceleration", "mpg"],
    )

    df_test = prep.transform(df_test)

    print(df_test.head())