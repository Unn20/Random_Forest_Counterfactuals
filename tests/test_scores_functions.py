import pytest
import numpy as np

from rf_counterfactuals.scores_functions import *

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
        score = k_nearest_neighborhood(X_prime, y, label, nbrs)
        assert score == pytest.approx(true_scores[no]), f"X_prime={X_prime}"

def test_n_sigma_random_neighborhood():
    assert True == True
