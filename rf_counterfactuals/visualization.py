import numpy as np
import pandas as pd
from scipy.stats import rankdata

from collections import defaultdict

from rf_explainer import RandomForestExplainer
from costs_functions import *
from scores_functions import *

def visualize(rfe: RandomForestExplainer, X: pd.Series, X_prime: pd.Series, label_encoder_dict=None):
    """ Visualize a difference between original instance and its counterfactual """
    difference = X_prime - X
    # difference_rank = pd.Series(rankdata(np.abs(difference.values), method='dense'), index=X.index)
    constraints = pd.Series([_map_constraint(i, rfe) for i in range(X.shape[0])], index=X.index)
    X = X.round(3)
    X_prime = X_prime.round(3)
    difference = difference.round(3)
    if label_encoder_dict:
        X[rfe.categorical_features] = X[rfe.categorical_features].to_frame(0).apply(lambda x: label_encoder_dict[x.name].inverse_transform(x)[0], axis=1)
        X_prime[rfe.categorical_features] = X_prime[rfe.categorical_features].to_frame(0).apply(lambda x: label_encoder_dict[x.name].inverse_transform(x.astype(int))[0], axis=1)
        difference[rfe.categorical_features] = difference[rfe.categorical_features].apply(lambda x: 1.0 if x != 0.0 else 0.0)
    result = pd.concat([X.T, X_prime.T, difference.T, constraints.T], axis=1, ignore_index=True)
    result.columns = ["X", "X'", "difference", "constraints"]
    return result


def evaluate_counterfactual(rfe: RandomForestExplainer, X: pd.Series, X_prime: pd.Series, k=5):
    metrics = {}
    normalized_X = (X - rfe.X_train_stats['mean']) / rfe.X_train_stats['std']

    normalized_X_prime = (X_prime - rfe.X_train_stats['mean']) / rfe.X_train_stats['std']
    metrics['unmatched_components_rate'] = unmatched_components_distance(normalized_X, normalized_X_prime)
    metrics['euclidean_distance'] = euclidean_distance(normalized_X, normalized_X_prime)
    metrics['euclidean_categorical_distance'] = euclidean_categorical_distance(normalized_X, normalized_X_prime,
                                                                               rfe.categorical_features,
                                                                               rfe.non_categorical_features)
    metrics['cosine_distance'] = cosine_distance(normalized_X, normalized_X_prime)
    metrics['jaccard_distance'] = jaccard_distance(normalized_X, normalized_X_prime)
    metrics['pearson_correlation_distance'] = pearson_correlation_distance(normalized_X, normalized_X_prime)

    metrics['sparsity'] = metrics['unmatched_components_rate']
    metrics['proximity'] = heterogeneous_euclidean_overlap_metric(X.values, X_prime.values, rfe.X_train_stats['range'],
                                                                  rfe.categorical_features,
                                                                  rfe.non_categorical_features)

    label = rfe.rf_model.predict(pd.DataFrame(X_prime.values.reshape(1, -1), columns=X_prime.index))[0]
    nbrs_same_label = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
                                       metric=heterogeneous_euclidean_overlap_metric,
                                       metric_params={'feature_range': rfe.X_train_stats['range'],
                                                      'cat_features': rfe.categorical_features,
                                                      'non_cat_features': rfe.non_categorical_features})
    X_train_same_labels = rfe.X_train[rfe.y_train == label]
    nbrs_same_label.fit(X_train_same_labels.values)
    metrics['implausibility'] = implausibility_single(X_prime.values, nbrs_same_label)
    return metrics


def evaluate_counterfactual_set(rfe: RandomForestExplainer, X: pd.Series, X_primes: pd.DataFrame, k=5):
    metrics = {}

    metrics['mean_hoem'] = np.mean([heterogeneous_euclidean_overlap_metric(X.values, X_prime.values, rfe.X_train_stats['range'],
                                                                  rfe.categorical_features,
                                                                  rfe.non_categorical_features) for _, X_prime in X_primes.iterrows()])

    metrics['diversity'] = diversity(X_primes, rfe.X_train_stats['range'], rfe.categorical_features, rfe.non_categorical_features)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
                                    metric=heterogeneous_euclidean_overlap_metric,
                                    metric_params={'feature_range': rfe.X_train_stats['range'],
                                                   'cat_features': rfe.categorical_features,
                                                   'non_cat_features': rfe.non_categorical_features})
    nbrs.fit(rfe.X_train)
    metrics['implausibility'] = implausibility(X_primes, nbrs)
    return metrics


def _map_constraint(feature_id, rfe):
    constraints = []
    if feature_id in rfe.left_frozen_features and feature_id in rfe.right_frozen_features:
        constraints.append("frozen")
    elif feature_id in rfe.left_frozen_features and feature_id not in rfe.right_frozen_features:
        constraints.append("left_frozen")
    elif feature_id not in rfe.left_frozen_features and feature_id in rfe.right_frozen_features:
        constraints.append("right_frozen")
    if feature_id in rfe.categorical_features:
        constraints.append("categorical")
    return ", ".join(constraints)
