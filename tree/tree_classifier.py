import numpy as np
import pandas as pd


def get_best_split(X: pd.DataFrame, y: pd.Series):
    best_split = None
    for col_name in X.columns:
        splits = np.sort(X[col_name].unique())
        if len(splits) < 2:
            continue

        splits = (splits[:-1] + splits[1:]) / 2
        for split in splits:
            ig = information_gain(X[col_name], y, split)
            if best_split is None or ig > best_split[2]:
                best_split = (col_name, split, ig)
    return best_split


def information_gain(x: np.ndarray, y: np.ndarray, split_value):
    result = entropy(y)
    condition = x <= split_value
    left_arr = y[condition]
    right_arr = y[~condition]

    for q in (left_arr, right_arr):
        result -= (q.shape[0] / y.shape[0]) * entropy(q)
    return result


def entropy(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    preds = counts / x.shape[0]
    return -np.sum(preds * np.log2(preds))


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree = {}
        self._id = 0

    def __str__(self):
        params = [
            ('max_depth', self.max_depth),
            ('min_samples_split', self.min_samples_split),
            ('max_leafs', self.max_leafs)
        ]
        params_str = [f'{name}={value}' for name, value in params]
        return 'MyTreeClf class: ' + ', '.join(params_str)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree = self.build_node(X, y)

    def build_node(self, X: pd.DataFrame, y: pd.Series, depth=1) -> dict:
        self._id += 1
        node = {"id": self._id, 'depth': depth}
        if self.is_leaf(X, y, depth):
            self.leafs_cnt += 1
            node['prob'] = y.sum() / len(y)
            return node
        split = get_best_split(X, y)
        node["col"] = split[0]
        node["split_value"] = split[1]
        node["information_gain"] = split[2]

        condition = X[node["col"]] <= node["split_value"]
        left_X, left_y = X[condition], y[condition]
        right_X, right_y = X[~condition], y[~condition]
        node["left"] = self.build_node(left_X, left_y, depth + 1)
        node["right"] = self.build_node(right_X, right_y, depth + 1)
        return node

    def is_leaf(self, X: pd.DataFrame, y: pd.Series, depth):
        return depth != 1 and (
                len(y.unique()) < 2 or
                len(X) < self.min_samples_split or
                depth - 1 >= self.max_depth or
                self.leafs_cnt + 1 >= self.max_leafs
        )

    def predict_proba(self, X: pd.DataFrame):
        result = []
        for i, x in X.iterrows():
            result.append(self.predict_row(x))
        return np.array(result)

    def predict(self, X: pd.DataFrame):
        result = []
        for index, x in X.iterrows():
            result.append(1 if self.predict_row(x) > 0.5 else 0)
        return np.array(result)

    def predict_row(self, x):
        stack = [self.tree]
        while stack:
            node = stack.pop()
            if 'prob' in node:
                return node['prob']

            if x[node['col']] <= node['split_value']:
                stack.append(node['left'])
            else:
                stack.append(node['right'])
        return 0.0

    def print_tree(self):
        visited = set()
        stack = [self.tree]
        while stack:
            node = stack.pop()
            if node["id"] not in visited:
                visited.add(node["id"])
                print((node["depth"] - 1) * '\t', end='')
                if 'prob' in node:
                    print(f'prob = {node["prob"]}')
                    return
                stack.append(node["right"])
                stack.append(node["left"])
                print(f'{node["col"]} < {node["split_value"]}')


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:, :4], df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MyTreeClf(max_depth=5, min_samples_split=10, max_leafs=10)
    model.fit(X_train, y_train)
    print(f'Leaves count: {model.leafs_cnt}')

    pred = model.predict(X_test)
    print(pred)
