import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self):
        self.scalers = {}

    def fit(self, df, log_cols, scale_cols):
        """
        Fits transformations (log + standard scaling).
        """
        fit_df = df.copy()
        self.log_cols = log_cols
        self.scale_cols = scale_cols

        for col in self.scale_cols:
            if col in self.log_cols:
                fit_df[col] = np.log1p(fit_df[col])
            self.scalers[col] = (np.mean(fit_df[col]), np.std(fit_df[col]))

        return self.transform(df)

    def transform(self, df):
        """
        Applies log transformation & scaling using stored parameters.
        """

        df[self.log_cols] = np.log1p(df[self.log_cols])
  
        for col in self.scale_cols:
            mean, std = self.scalers[col]
            df[col] = (df[col] - mean) / std

        return df

    def inverse_transform_y(self, y_pred):
        """
        Reverts transformations on predicted y values.
        """
        return np.expm1(y_pred)
