
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from single_cf_costs_functions import heterogeneous_euclidean_overlap_metric

def implausibility(X_primes: np.array, nbrs: NearestNeighbors):
    distances, indices = np.squeeze(nbrs.kneighbors(X_primes, n_neighbors=1, return_distance=True))
    score = np.mean(distances)
    return score


def diversity(X_primes: np.array, feature_range: np.array, cat_features: list, non_cat_features: list):
    distance_matrix = pairwise_distances(X_primes, metric=heterogeneous_euclidean_overlap_metric, n_jobs=1,
                                         feature_range=feature_range, cat_features=cat_features, non_cat_features=non_cat_features)
    diversity = np.sum(np.triu(distance_matrix, k=1)) # Sum all elements of upper triangle of matrix
    diversity = diversity / ((distance_matrix.shape[0] * (distance_matrix.shape[0] - 1)) / 2)
    return diversity
