import math

import numpy as np
import pandas as pd
from scipy import spatial, stats
from sklearn.neighbors import NearestNeighbors


def k_nearest_neighborhood(X_prime: np.array, y: np.array, label, nbrs: NearestNeighbors):
    indices = np.squeeze(nbrs.kneighbors(X_prime, return_distance=False))
    score = np.mean(y[indices] == label)
    return score


def n_sigma_random_neighborhood(X_prime: np.array, trained_model, sigma, n):
    score = 0
    return score