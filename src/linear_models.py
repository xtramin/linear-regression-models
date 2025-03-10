import numpy as np


class OrinaryLeastSquares:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.observations, self.features = self.X.shape
        self.beta = None

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

    def calculate_beta(self):
        dot_product = self.X.T @ self.X

        if np.linalg.matrix_rank(dot_product) < self.observations:
            raise ValueError("X transpose X is not invertible.")
        
        self.beta = np.linalg.pinv(dot_product) @ self.X.T @ self.y

        return self.beta