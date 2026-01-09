import numpy as np
import pandas as pd

from functional import euclidean, manhattan, chebyshev, cosine


class MyKNNReg:
    def __init__(
            self,
            k: int = 3,
            metric='euclidean',
            weight='uniform',
    ):
        self.k = k
        self.train_size = 0
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.train_size = X.shape

    def _find_nearest_k(self, row: np.ndarray):
        d = self._find_metric(row)
        sorted_indices = np.argsort(d)
        nearest_k = self.y[sorted_indices][:self.k]
        nearest_distances = d[sorted_indices][:self.k]
        return nearest_k, nearest_distances

    def _find_metric(self, row: np.ndarray):
        metric_fns = {
            'euclidean': euclidean,
            'manhattan': manhattan,
            'chebyshev': chebyshev,
            'cosine': cosine,
        }
        metric_fn = metric_fns[self.metric]
        return np.apply_along_axis(lambda X_row: metric_fn(X_row, row), arr=self.X, axis=1)

    def _find_weights(self, nearest_k: np.ndarray, distances: np.ndarray):
        weight_metric = distances
        if self.weight == 'rank':
            weight_metric = np.arange(1, len(nearest_k) + 1)

        weights = []
        total_sum = np.sum(1 / weight_metric)
        for i in range(len(nearest_k)):
            w = (1 / weight_metric[i]) / total_sum
            weights.append(w)
        return np.array(weights)

    def _predict(self, row: np.ndarray):
        nearest_k, distances = self._find_nearest_k(row)

        if self.weight == 'uniform':
            return np.mean(nearest_k)
        weights = self._find_weights(nearest_k, distances)
        return weights @ nearest_k

    def predict(self, X: pd.DataFrame):
        preds = [self._predict(row) for row in X.to_numpy()]
        return np.array(preds)
