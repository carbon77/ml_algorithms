import random

import numpy as np
import pandas as pd


class MySVM:
    def __init__(
            self,
            n_iter: int = 10,
            learning_rate=0.001,
            weights=None,
            b=None,
            C=1,
            sgd_sample=None,
            random_state=42,
    ):
        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = weights
        self.bias = b
        self.loss = None
        self.c_coef = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        self.w_grad = None
        self.b_grad = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        y[y == 0] = -1
        y = y.to_numpy()
        X = X.to_numpy()
        self.weights = np.ones(shape=(X.shape[1],))
        self.bias = 1.0

        if verbose:
            self._compute_loss(X, y)
            print(f'start | loss: {self.loss}')

        for step in range(1, self.n_iter + 1):
            y_batch = y
            X_batch = X
            if self.sgd_sample:
                n_samples = self.sgd_sample if isinstance(self.sgd_sample, int) else int(
                    X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), n_samples)
                y_batch = y[sample_rows_idx]
                X_batch = X[sample_rows_idx]

            for i in range(len(X_batch)):
                self._compute_grad(i, X_batch, y_batch)
                self.weights -= self.lr * self.w_grad
                self.bias -= self.lr * self.b_grad

            if verbose and step % verbose == 0:
                self._compute_loss(X, y)
                print(f'start | loss: {self.loss}')

    def get_coef(self):
        return self.weights, self.bias

    def predict(self, X: pd.DataFrame):
        y = np.sign(X @ self.weights + self.bias)
        y[y == -1] = 0
        return y.astype(int)

    def _compute_grad(self, idx: int, X: np.ndarray, y: np.ndarray):
        if y[idx] * (self.weights @ X[idx] + self.bias) >= 1:
            self.w_grad = 2 * self.weights
            self.b_grad = 0
        else:
            self.w_grad = 2 * self.weights - self.c_coef * y[idx] * X[idx]
            self.b_grad = -self.c_coef * y[idx]

    def _compute_loss(self, X: np.ndarray, y: np.ndarray):
        preds = X @ self.weights + self.bias
        hinge_loss = self.c_coef * np.mean(np.maximum(0, 1 - y * preds))
        self.loss = np.linalg.norm(self.weights) ** 2 + hinge_loss

    def __str__(self):
        print_params = [
            ('n_iter', self.n_iter),
            ('learning_rate', self.lr),
        ]
        params_str = ', '.join(f'{name}={value}' for name, value in print_params)
        return f'MySVM class: {params_str}'


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    model = MySVM(n_iter=10, learning_rate=0.05)
    model.fit(X, y, verbose=10)

    print(model.get_coef())
