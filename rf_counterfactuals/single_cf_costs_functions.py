import math

import numpy as np
import pandas as pd
from scipy import spatial, stats
from sklearn.neighbors import NearestNeighbors


def unmatched_components(X: np.array, X_prime: np.array, eps:float):
    result = 0
    for feature_id in range(len(X)):
        if abs(X[feature_id] - X_prime[feature_id]) > eps:
            result += 1
    return result


def unmatched_components_distance(X: np.array, X_prime: np.array, eps=1e-4):
    return unmatched_components(X, X_prime, eps) / len(X) if len(X) > 0 else 0.0


def heterogeneous_euclidean_overlap_metric(X: np.array, X_prime: np.array, feature_range: np.array, cat_features: list, non_cat_features: list):
    """ https://axon.cs.byu.edu/~randy/jair/wilson2.html """
    result = 0.0
    result += unmatched_components(X[cat_features], X_prime[cat_features], eps=1e-4)
    result += np.sum(np.power(np.clip(np.divide(
        np.absolute(X[non_cat_features] - X_prime[non_cat_features]), feature_range[non_cat_features],
        out=np.ones_like(X[non_cat_features], dtype=np.float), where=feature_range[non_cat_features] != 0.0), 0, 1), 2))
    return np.sqrt(result)


def euclidean_distance(X: np.array, X_prime: np.array):
    return np.linalg.norm(X - X_prime)


def euclidean_categorical_distance(X: np.array, X_prime: np.array, cat_features: list, non_cat_features: list):
    result = euclidean_distance(X[non_cat_features], X_prime[non_cat_features])\
             + unmatched_components(X[cat_features], X_prime[cat_features], eps=1e-4)
    return result


def cosine_distance(X: np.array, X_prime: np.array):
    return spatial.distance.cosine(X, X_prime)


def jaccard_distance(X: np.array, X_prime: np.array):
    """ Calculate a jaccard distance as a fraction of sets intersection and union lengths"""
    s_u = set(X)
    s_v = set(X_prime)
    s_u_and_v = s_u.intersection(s_v)
    s_u_or_v = s_u.union(s_v)
    Js = len(s_u_and_v) / float(len(s_u_or_v))
    Jd = 1 - Js
    return Jd


def pearson_correlation_distance(X: np.array, X_prime: np.array):
    rho = stats.pearsonr(X, X_prime)[0] # Returns only Pearson's correlation coefficient
    rho_d = 1 - rho
    return rho_d if not math.isnan(rho) else 1.0


def implausibility_single(X_prime: np.array, nbrs: NearestNeighbors, k=3):
    distances, indices = np.squeeze(nbrs.kneighbors(X_prime.reshape(1, -1), n_neighbors=k, return_distance=True))
    score = np.mean(distances)
    return score

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

