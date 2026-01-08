import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    true_c = (y_true >= threshold).astype(int)
    pred_c = (y_pred >= threshold).astype(int)

    t = np.sum(true_c == pred_c)
    f = np.sum(true_c != pred_c)
    return t / (t + f)


def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    Precision = TP / (TP + FP)
    """
    true_c = (y_true >= threshold).astype(int)
    pred_c = (y_pred >= threshold).astype(int)

    tp = np.sum((pred_c == 1) & (true_c == 1))
    fp = np.sum((pred_c == 1) & (true_c == 0))
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    recall = TP / (TP + FN)
    """
    true_c = (y_true >= threshold).astype(int)
    pred_c = (y_pred >= threshold).astype(int)

    tp = np.sum((pred_c == 1) & (true_c == 1))
    fn = np.sum((pred_c == 0) & (true_c == 1))
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    f1_score = 2 * (precision * recall) / (precision + recall)
    """
    p = precision(y_true, y_pred, threshold)
    r = recall(y_true, y_pred, threshold)
    return 2 * p * r / (p + r)


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    true_c = (y_true >= threshold).astype(int)

    sorted_indices = np.argsort(y_pred)[::-1]
    true_c = true_c[sorted_indices]
    y_pred = y_pred[sorted_indices]

    total_sum = 0.0
    eps = 1e-10
    for i in range(len(true_c)):
        if true_c[i] == 1:
            continue

        count_higher = np.sum(true_c[:i] == 1)

        same_score_mask = np.abs(y_pred - y_pred[i]) <= eps
        count_same = np.sum(true_c[same_score_mask] == 1)
        total_sum += count_higher + count_same / 2

    P = np.sum(true_c == 1)
    N = np.sum(true_c == 0)
    return total_sum / (P * N)


class MyLogReg:
    def __init__(
            self,
            weights: np.ndarray = None,
            n_iter=10,
            learning_rate=0.1,
            metric: str = None,
            reg: str = None,
            l1_coef=0.0,
            l2_coef=0.0,
            sgd_sample=None,
            random_state=42,
    ):
        self.weights = weights
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.metric_score = 0.0
        self.loss = 0.0
        self.grad = 0.0
        self.eps = 1e-15
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
        params_str = ', '.join(f'{name}={value}' for name, value in print_params)
        return f'MyLogReg class: {params_str}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        X.insert(loc=0, column='ones', value=1)
        X = X.to_numpy()
        y_true = y.to_numpy()
        n_features = X.shape[1]
        self.weights = np.ones(shape=(n_features,))

        if verbose:
            y_pred = self._sigmoid(X @ self.weights)
            self._compute_loss(X, y_true, y_pred)
            self._compute_metric(y_true, y_pred)
            self.log('start')

        for step in range(1, self.n_iter + 1):
            y_pred = self._sigmoid(X @ self.weights)
            self._compute_grad(X, y_true, y_pred)

            lr = self.learning_rate
            if not isinstance(self.learning_rate, float):
                lr = self.learning_rate(step)
            self.weights -= lr * self.grad

            if verbose and step % verbose == 0:
                self._compute_loss(X, y_true, y_pred)
                self._compute_metric(y_true, y_pred)
                self.log(step)

        y_pred = self._sigmoid(X @ self.weights)
        self._compute_metric(y_true, y_pred)

    def get_best_score(self):
        return self.metric_score

    def log(self, step):
        params = [
            ('loss', self.loss),
        ]
        if self.metric:
            params.append((self.metric, self.metric_score))
        params_str = ' | '.join([f'{name}: {value}' for name, value in params])
        print(f'{step} | {params_str}')

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        X_copy = X.copy()
        X_copy.insert(loc=0, column='ones', value=1)
        preds = self._sigmoid(X_copy @ self.weights)
        return preds > 0.5

    def predict_proba(self, X: pd.DataFrame):
        X_copy = X.copy()
        X_copy.insert(loc=0, column='ones', value=1)
        preds = self._sigmoid(X_copy @ self.weights)
        return preds

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        if self.metric == 'accuracy':
            self.metric_score = accuracy(y_true, y_pred)
        elif self.metric == 'precision':
            self.metric_score = precision(y_true, y_pred)
        elif self.metric == 'recall':
            self.metric_score = recall(y_true, y_pred)
        elif self.metric == 'f1':
            self.metric_score = f1_score(y_true, y_pred)
        elif self.metric == 'roc_auc':
            self.metric_score = roc_auc(y_true, y_pred)

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        self.loss = -np.mean(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log(1 - y_pred + self.eps))

        if self.reg == 'l1' or self.reg == 'elasticnet':
            self.loss += self.l1_coef * np.sum(np.abs(self.weights))

        if self.reg == 'l2' or self.reg == 'elasticnet':
            self.loss += self.l2_coef * np.sum(self.weights ** 2)

    def _compute_grad(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        true_batch, pred_batch = y_true, y_pred
        X_batch = X
        if self.sgd_sample:
            n_samples = self.sgd_sample if isinstance(self.sgd_sample, int) else int(X.shape[0] * self.sgd_sample)
            sample_rows_idx = random.sample(range(X.shape[0]), n_samples)
            true_batch = y_true[sample_rows_idx]
            pred_batch = y_pred[sample_rows_idx]
            X_batch = X[sample_rows_idx]

        self.grad = ((pred_batch - true_batch) @ X_batch) / X_batch.shape[0]
        if self.reg == 'l1' or self.reg == 'elasticnet':
            self.grad += self.l1_coef * np.sign(self.weights)

        if self.reg == 'l2' or self.reg == 'elasticnet':
            self.grad += self.l2_coef * 2 * self.weights


if __name__ == '__main__':
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    model = MyLogReg(n_iter=50, learning_rate=0.1, metric='roc_auc')
    model.fit(X, y, verbose=10)
    print(np.mean(model.get_coef()))
    print(model.get_best_score())
