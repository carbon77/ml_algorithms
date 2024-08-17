import numpy as np
import pandas as pd
import random

# Linear Regression
class MyLineReg:
    def __init__(self, n_iter, learning_rate, weights=[], metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.loss = 0
        self.score = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        params = {
            'n_iter': self.n_iter,
            'learning_rate': self.learning_rate,
            'metric': self.metric,
            'reg': self.reg,
        }

        return 'MyLineReg class: ' + ', '.join([f'{key}={value}' for key, value in params.items()])

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        n = X.shape[0]
        X.insert(0, 'ones', [1] * X.shape[0])
        X.columns = range(X.shape[1])
        self.weights = np.ones(X.shape[1])

        if verbose:
          y_pred = np.dot(X, self.weights)
          self.find_metric(y, y_pred)
          self.loss = self.find_loss(y, y_pred)
          self.log()

        for i in range(1, self.n_iter + 1):
          y_pred = np.dot(X, self.weights)
          self.find_metric(y, y_pred)

          if verbose and i % verbose == 0:
            self.loss = self.find_loss(y, y_pred)
            self.log(i)

          grad = self.find_gradient(X, y, y_pred)
          learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
          self.weights = self.weights - learning_rate * grad

        self.find_metric(y, np.dot(X, self.weights))

    def log(self, step='start'):
      fields = {
          'loss': self.loss,
      }

      if self.metric is not None:
        fields[self.metric] = self.score
      print(f'{step} |', ' | '.join([f'{name}: {value}' for name, value in fields.items()]))

    def find_loss(self, y_true, y_pred):
      l1 = 0
      l2 = 0

      if self.reg in ('l1', 'elasticnet'):
        l1 = self.l1_coef * np.sum(np.abs(self.weights))

      if self.reg in ('l2', 'elasticnet'):
        l2 = self.l2_coef * np.sum(self.weights ** 2)

      return np.mean((y - y_pred) ** 2) + l1 + l2

    def find_gradient(self, X, y_true, y_pred):
      l1 = 0
      l2 = 0

      X_batch = X.to_numpy()
      y_true_batch = y_true.to_numpy()
      y_pred_batch = y_pred

      if self.sgd_sample:
        sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample if isinstance(self.sgd_sample, int) else int(X.shape[0] * self.sgd_sample))
        X_batch = X_batch[sample_rows_idx]
        y_true_batch = y_true_batch[sample_rows_idx]
        y_pred_batch = y_pred_batch[sample_rows_idx]


      if self.reg in ('l1', 'elasticnet'):
        l1 = self.l1_coef * np.sign(self.weights)

      if self.reg in ('l2', 'elasticnet'):
        l2 = 2 * self.l2_coef * self.weights

      grad = 2 * np.dot((y_pred_batch - y_true_batch), X_batch) / X_batch.shape[0]
      return grad + l1 + l2

    def get_coef(self):
      return self.weights[1:]

    def predict(self, X: pd.DataFrame):
      X.insert(0, 'ones', [1] * X.shape[0])
      return np.sum(np.dot(X, self.weights))

    def find_metric(self, y_true, y_pred):
      if self.metric == 'mae':
        self.score = np.mean(np.abs(y_true - y_pred))
      elif self.metric == 'mse':
        self.score = np.mean((y_true - y_pred) ** 2)
      elif self.metric == 'rmse':
        self.score = np.sqrt(np.mean((y_true - y_pred) ** 2))
      elif self.metric == 'mape':
        self.score = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
      else:
        self.score = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    def get_best_score(self):
      return self.score

if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]
    model = MyLineReg(50, lambda iter: 0.5 * (0.85 ** iter), metric='mae', reg='l1', l1_coef=0.7, sgd_sample=0.5)
    print(model)
    model.fit(X, y, verbose=10)
    print(model.get_best_score())