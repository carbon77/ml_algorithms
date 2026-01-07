import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class MyLineReg:
    def __init__(
            self,
            weights: np.ndarray = None,
            n_iter: int = 100,
            learning_rate: float = 0.1,
            metric: str = None,
            reg: str = None,
            l1_coef=0.0,
            l2_coef=0.0,
            sgd_sample = None,
            random_state: int = 42,
    ):
        self.weights = weights
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.metric_score = None
        self.loss = 0
        self.grad = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        print_params = [
            ('n_iter', self.n_iter),
            ('learning_rate', self.learning_rate),
        ]
        if self.metric:
            print_params.append(('metric', self.metric))
        params_str = ', '.join(f'{name}={value}' for name, value in print_params)
        return f'MyLineReg class: {params_str}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        X.insert(loc=0, column='ones', value=1)
        n_features = X.shape[1]
        self.weights = np.ones(shape=(n_features,))

        if verbose:
            y_pred = np.dot(X, self.weights)
            self._compute_loss(y, y_pred)
            self._compute_metric_score(y, y_pred)
            self._log_step('start')

        for step in range(1, self.n_iter + 1):
            y_pred = np.dot(X, self.weights)
            self._compute_metric_score(y, y_pred)

            self._compute_grad(X.to_numpy(), y.to_numpy(), y_pred)
            lr = self.learning_rate if isinstance(self.learning_rate, float) else self.learning_rate(step)
            self.weights -= lr * self.grad

            if verbose and step % verbose == 0:
                self._log_step(step)
        self._compute_metric_score(y, X @ self.weights)

    def _log_step(self, step):
        step_params = [
            ('loss', self.loss),
        ]
        if self.metric:
            step_params.append(('metric', self.metric_score))
        print(f'{step} |', ' | '.join([f'{name}: {value}' for name, value in step_params]))

    def get_best_score(self):
        return self.metric_score

    def _compute_loss(self, y_true, y_pred):
        self.loss = np.mean((y_pred - y_true) ** 2)

        if self.reg == 'elasticnet' or self.reg == 'l1':
            l1 = self.l1_coef * np.sum(np.abs(self.weights))
            self.loss += l1

        if self.reg == 'elasticnet' or self.reg == 'l2':
            l2 = self.l2_coef * np.sum(self.weights ** 2)
            self.loss += l2

    def _compute_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        true_sample, pred_sample = y_true, y_pred
        X_sample = X
        if self.sgd_sample:
            n_samples = self.sgd_sample if isinstance(self.sgd_sample, int) else int(X.shape[0] * self.sgd_sample)
            sample_rows_idx = random.sample(range(X.shape[0]), n_samples)
            true_sample = y_true[sample_rows_idx]
            pred_sample = y_pred[sample_rows_idx]
            X_sample = X[sample_rows_idx]
        self.grad = (2 / X_sample.shape[0]) * np.dot(pred_sample - true_sample, X_sample)

        if self.reg == 'elasticnet' or self.reg == 'l1':
            l1 = self.l1_coef * np.sign(self.weights)
            self.grad += l1

        if self.reg == 'elasticnet' or self.reg == 'l2':
            l2 = self.l2_coef * 2 * self.weights
            self.grad += l2

    def _compute_metric_score(self, y_true, y_pred):
        if not self.metric:
            return
        metrics = {
            'mae': lambda t, p: np.mean(np.abs(t - p)),
            'mse': lambda t, p: np.mean((t - p) ** 2),
            'rmse': lambda t, p: np.sqrt(np.mean((t - p) ** 2)),
            'mape': lambda t, p: 100 * np.mean(np.abs((t - p) / t)),
            'r2': lambda t, p: 1 - np.sum((t - p) ** 2) / np.sum((t - np.mean(t)) ** 2)
        }
        self.metric_score = metrics[self.metric](y_true, y_pred)

    def predict(self, X: pd.DataFrame):
        X.insert(loc=0, column='ones', value=1)
        preds = np.dot(X, self.weights)
        return preds

    def get_coef(self):
        return self.weights[1:]


if __name__ == '__main__':
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    model = MyLineReg(n_iter=50, learning_rate=0.5, metric='mae', sgd_sample=0.1)
    model.fit(X, y, verbose=10)
    print(np.mean(model.get_coef()))
    print(model.get_best_score())
