import os
from src.pipeline import run_pipeline

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./data/auto-mpg.data")
)

if __name__ == "__main__":

    df_train, df_test = run_pipeline(DATA_PATH)

    print(df_train.head(), df_test.head())