import pytest
import numpy as np

from rf_counterfactuals.single_cf_costs_functions import *
from rf_counterfactuals.multi_cf_costs_functions import *

def test_k_nearest_neighborhood():
    X = np.array([[-1, -1], [2, -1], [3, -2], [1, 1], [2, 1], [3, 2], [5, -1], [1, -3]])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    label = 1

    X_primes = [
        [1, 4],
        [5, -3],
        [1, -2],
        [0, -2]
    ]
    true_scores = [1/3, 2/3, 1/3, 0/3]

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    nbrs.fit(X)

    for no in range(len(X_primes)):
        X_prime = np.array(X_primes[no]).reshape(1, -1)
        score = k_nearest_neighborhood(X_prime, label, nbrs, y)
        assert score == pytest.approx(true_scores[no]), f"X_prime={X_prime}"

def test_diversity():
    categorical_features = []
    non_categorical_features = [0, 1, 2]
    feature_range = np.array([2, 1, 1])
    assert diversity(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), feature_range, categorical_features,
                     non_categorical_features) == 0
    assert diversity(np.array([[2, 1, 1], [1, 1, 1], [1, 1, 1]]), feature_range, categorical_features,
                     non_categorical_features) == pytest.approx(((1/2)*2)/3)
    assert diversity(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), feature_range, categorical_features,
                     non_categorical_features) == pytest.approx((math.sqrt(2)+math.sqrt(5/4)+math.sqrt(5/4))/3)
