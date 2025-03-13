import os
from src.pipeline import run_pipeline
from src.linear_models import OrdinaryLeastSquares

import numpy as np
from sklearn.metrics import r2_score

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data/auto-mpg.data")
)

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = run_pipeline(DATA_PATH)

    ols = OrdinaryLeastSquares()

    ols.fit(X_train,  y_train)

    ypred = ols.predict(X_test)

    print(r2_score(y_test, ypred))