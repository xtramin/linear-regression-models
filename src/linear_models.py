import numpy as np


class OrdinaryLeastSquares:
    def __init__(self):
        self.X = None
        self.y = None
        self.observations = None
        self.features = None
        self.beta = None

    def fit(self, X, y):

        self.X = X
        self.y = y

        self.observations, self.features = self.X.shape

        if self.observations != self.y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

        dot_product = self.X.T @ self.X

        if np.linalg.matrix_rank(dot_product) < self.features:
            raise ValueError("X transpose X is not invertible.")
        
        self.beta = np.linalg.pinv(dot_product) @ self.X.T @ self.y

        return self.beta
    
    def predict(self, X):
        y_pred = X @ self.beta
        return y_pred