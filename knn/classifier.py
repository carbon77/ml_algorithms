import numpy as np
import pandas as pd


def euclidean(x: np.ndarray, y: np.ndarray):
    return np.sum((x - y) ** 2) ** 0.5


def manhattan(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y))


def chebyshev(x: np.ndarray, y: np.ndarray):
    return np.max(np.abs(x - y))


def cosine(x: np.ndarray, y: np.ndarray):
    return 1 - (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))


class MyKNNClf:
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

    def _find_weighted_classes(self, nearest_k: np.ndarray, distances: np.ndarray):
        q_metric = distances
        if self.weight == 'rank':
            q_metric = np.arange(1, len(nearest_k) + 1)

        qs = []
        total_sum = np.sum(1 / q_metric)
        for c in [0, 1]:
            c_indices = np.where(nearest_k == c)
            q = np.sum(1 / q_metric[c_indices]) / total_sum
            qs.append(q)
        return np.array(qs)

    def _predict(self, row: np.ndarray, prob=False):
        nearest_k, distances = self._find_nearest_k(row)

        if self.weight == 'uniform':
            ones_count = np.sum(nearest_k)
            if prob:
                return ones_count / self.k
            return ones_count >= np.ceil(self.k / 2)

        qs = self._find_weighted_classes(nearest_k, distances)
        if prob:
            return qs[1]
        return np.argmax(qs)

    def predict(self, X: pd.DataFrame):
        preds = [self._predict(row) for row in X.to_numpy()]
        return np.array(preds).astype(int)

    def predict_proba(self, X: pd.DataFrame):
        preds = [self._predict(row, True) for row in X.to_numpy()]
        return np.array(preds)


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    model = MyKNNClf(5, weight='distance')
    model.fit(X, y)
    X_test, y_test = make_classification(n_samples=10, n_features=14, n_informative=10, random_state=42)
    y_pred = model.predict_proba(pd.DataFrame(X_test))
    print(y_test)
    print(y_pred)
