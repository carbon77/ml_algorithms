import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray):
    return np.sum((x - y) ** 2) ** 0.5


def manhattan(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y))


def chebyshev(x: np.ndarray, y: np.ndarray):
    return np.max(np.abs(x - y))


def cosine(x: np.ndarray, y: np.ndarray):
    return 1 - (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))


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
