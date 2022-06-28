import math

import numpy as np
import pandas as pd
from scipy import spatial, stats


def unmatched_components_rate(X: np.array, X_prime: np.array, eps=1e-4):
    unmatched_components = 0
    for feature_id in range(len(X)):
        if abs(X[feature_id] - X_prime[feature_id]) > eps:
            unmatched_components += 1
    return unmatched_components / len(X) if len(X) > 0 else 0.0


def euclidean_distance(X: np.array, X_prime: np.array):
    return np.linalg.norm(X - X_prime)


def cosine_distance(X: np.array, X_prime: np.array):
    return spatial.distance.cosine(X, X_prime)


def jaccard_distance(X: np.array, X_prime: np.array):
    """ Calculate a jaccard distance as an fraction of sets intersection and union lengths"""
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



