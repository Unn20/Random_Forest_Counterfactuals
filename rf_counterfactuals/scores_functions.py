import math

import numpy as np
import pandas as pd
from scipy import spatial, stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from costs_functions import heterogeneous_euclidean_overlap_metric


def k_nearest_neighborhood(X_prime: np.array, label, nbrs: NearestNeighbors, y_train: np.array):
    indices = np.squeeze(nbrs.kneighbors(X_prime.reshape(1, -1), return_distance=False))
    score = np.mean(y_train[indices] == label)
    return score


def n_sigma_random_neighborhood(X_prime: np.array, label, sigma, n, trained_model, feature_means, feature_std):
    predictions = []
    for i in range(n):
        X_new = X_prime[0, :] + (np.random.normal(0, sigma, size=X_prime.shape[1]) * feature_std)
        prediction = trained_model.predict(pd.DataFrame(X_new.values.reshape(1, -1), columns=X_new.index))
        predictions.append(prediction)
    p = np.squeeze(np.array(predictions))
    score = np.sum(p == label) / n
    return score


def diversity(X_primes: np.array, feature_range: np.array, cat_features: list, non_cat_features: list):
    distance_matrix = pairwise_distances(X_primes, metric=heterogeneous_euclidean_overlap_metric, n_jobs=1,
                                         feature_range=feature_range, cat_features=cat_features, non_cat_features=non_cat_features)
    diversity = np.sum(np.triu(distance_matrix, k=1)) # Sum all elements of upper triangle of matrix
    diversity = diversity / ((distance_matrix.shape[0] * (distance_matrix.shape[0] - 1)) / 2)
    return diversity
