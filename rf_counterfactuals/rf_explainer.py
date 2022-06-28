from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import warnings
from functools import partial
from joblib import Parallel, delayed
import itertools

from .costs_functions import *
from .scores_functions import *

class RandomForestExplainer:
    available_loss_functions = {
        'euclidean': lambda x, x_prime: euclidean_distance(x, x_prime),
        'cosine': lambda x, x_prime: cosine_distance(x, x_prime),
        'jaccard': lambda x, x_prime: jaccard_distance(x, x_prime),
        'pearson_correlation': lambda x, x_prime: pearson_correlation_distance(x, x_prime),
        'unmatched_components': lambda x, x_prime: unmatched_components_rate(x, x_prime),
    }
    available_score_functions = {
        'k_nearest_neighborhood': lambda x_prime, y, label, nbrs:
        k_nearest_neighborhood(x_prime, y, label, nbrs),
        'n_sigma_random_neighborhood': lambda x_prime, trained_model, sigma, n:
        n_sigma_random_neighborhood(x_prime, trained_model, sigma, n),
    }

    def __init__(self, model: RandomForestClassifier):
        self.rf_model = model
        self.decision_trees = model.estimators_
        self.classes = list(model.classes_)
        self.get_class_for_dt = lambda label: self.classes.index(label)
        self.n_features = model.n_features_in_
        self.positive_paths = dict()

    def get_positive_paths_for_label(self, label):
        if label not in self.classes:
            raise Exception("This class label doesn't belong to the dataset, on which Random Forest was trained.")
        if label not in self.positive_paths:
            self.positive_paths[label] = _get_positive_paths(self.rf_model, self.get_class_for_dt(label))
        return self.positive_paths[label]

    def explain(self, X, label: int, eps=0.1, metrics=('euclidean',), k=5, sigma=0.1, n=100, n_jobs=-1):
        if eps <= 0.0:
            raise Exception(f"Eps parameter={eps} (must be greater than 0.0).")
        if X.ndim == 1:
            warnings.warn(f"X has ndim={X.ndim}. Reshaping: X = X.values.reshape(1, -1)", UserWarning)
            X = X.values.reshape(1, -1)
        elif X.ndim == 2:
            pass
        else:
            raise Exception(f"Incorrect dimensions of X={X.ndim} (should be 2).")
        params = {'k': k, 'sigma': sigma, 'n': n}
        # Precalculations
        positive_paths = self.get_positive_paths_for_label(label)
        feature_eps_mul_std = eps * np.std(X, axis=0)
        feature_means = np.mean(X, axis=0)

        y_hat_ensemble = self.rf_model.predict(X)
        wrong_label = np.where(y_hat_ensemble != label)[0]
        X_wrong_label = X.iloc[wrong_label, :]

        results_per_tree = Parallel(n_jobs=n_jobs, verbose=10, prefer='processes')(
            delayed(partial(_tweak_features, X_wrong_label, label, self.classes, self.rf_model, feature_means,
                            feature_eps_mul_std))(dt, pp) for dt, pp in zip(self.decision_trees, positive_paths))

        # results_per_tree have shape (trees_no, X.shape[0], CFs) so we need to concat CFs along 0 dimension
        rpt_iterator = iter(np.array(results_per_tree, dtype=object).T)
        counterfactuals = [pd.DataFrame(itertools.chain.from_iterable(next(rpt_iterator))) for no in range(X.shape[0]) if no in wrong_label]

        loss_functions = {metric: func for metric, func in RandomForestExplainer.available_loss_functions.items()
                          if metric in metrics}
        costs_per_row = []
        if len(loss_functions) > 0:
            costs_per_row = Parallel(n_jobs=n_jobs, verbose=10, prefer='processes')(
                delayed(partial(_calculate_distance, loss_functions))
                (x, x_primes) for x, x_primes in zip(X_wrong_label.values, counterfactuals))

        score_functions = {metric: func for metric, func in RandomForestExplainer.available_score_functions.items()
                           if metric in metrics}
        if 'k_nearest_neighborhood' in score_functions:
            nbrs = NearestNeighbors(n_neighbors=params['k'], algorithm='ball_tree')
            nbrs.fit(X.values)
            params['nbrs'] = nbrs
        else:
            params['nbrs'] = None

        scores_per_row = []
        if len(score_functions) > 0:
            scores_per_row = Parallel(n_jobs=n_jobs, verbose=10, prefer='processes')(
                delayed(partial(_calculate_score, score_functions, params, self.rf_model, X, y_hat_ensemble, label))
                (x_primes) for x_primes in counterfactuals)

        # Put whole calculation results into one DataFrame
        result = []
        for no, cfs in enumerate(counterfactuals):
            cfs["X_index"] = cfs.index
            if len(costs_per_row) > 0:
                for metric, value in costs_per_row[no].items():
                    cfs[metric] = value
            if len(scores_per_row) > 0:
                for metric, value in scores_per_row[no].items():
                    cfs[metric] = value
            result.append(cfs)
        result = pd.concat(result, ignore_index=True)

        return result


def _calculate_distance(distance_functions: dict, X_row: np.array, X_primes: pd.DataFrame):
    result = {metric: [] for metric in distance_functions.keys()}
    for ind, X_prime in X_primes.iterrows():
        for metric, func in distance_functions.items():
            result[metric].append(func(X_row, X_prime.values))
    return result


def _calculate_score(score_functions: dict, parameters: dict, trained_model: RandomForestClassifier,
                     X: pd.DataFrame, y: pd.Series, label, X_primes: pd.DataFrame):
    result = {metric: [] for metric in score_functions.keys()}
    for ind, X_prime in X_primes.iterrows():
        if 'k_nearest_neighborhood' in score_functions:
            k = parameters['k']
            nbrs = parameters['nbrs']
            func = score_functions['k_nearest_neighborhood']
            result[f"k_nearest_neighborhood"].append(func(X_prime.values.reshape(1, -1), y, label, nbrs))
        if 'n_random_neighborhood' in score_functions:
            sigma = parameters['sigma']
            n = parameters['n']
            func = score_functions['n_random_neighborhood']
            result[f'n_sigma_random_neighborhood'].append(func(X_prime.values.reshape(1, -1), trained_model, sigma, n))

    return result


def _tweak_features(X, label, classes, rf, feature_means, feature_eps_mul_std, decision_tree, positive_paths):
    predicted_classes_by_tree = np.array(decision_tree.predict(X.values), dtype=int)
    y_hat_tree = np.take(classes, predicted_classes_by_tree)

    to_tweak = np.where(y_hat_tree != label)[0]
    X_to_tweak = X.iloc[to_tweak, :]

    X_tweaked_candidates = [[] for _ in range(X.shape[0])]

    for positive_path in positive_paths:
        X_prime = X_to_tweak.copy()
        for feature_id, sign, threshold in positive_path:
            X_prime.iloc[:, feature_id] = threshold - feature_means[feature_id] + (sign * feature_eps_mul_std[feature_id])

        y_hat_prime = rf.predict(X_prime)
        candidates_indices = np.where(y_hat_prime == label)[0]
        for i in candidates_indices:
            X_tweaked_candidates[to_tweak[i]].append(X_prime.iloc[i, :])

    return X_tweaked_candidates


def _get_positive_paths(model: RandomForestClassifier, label: int):
    positive_paths = []
    for decision_tree_classifier in model.estimators_:
        tree_paths = _get_positive_paths_from_decision_tree(decision_tree_classifier, label)
        positive_paths.append(tree_paths)
    return positive_paths


def _get_positive_paths_from_decision_tree(decision_tree_classifier: DecisionTreeClassifier, label: int):
    tree = decision_tree_classifier.tree_
    nodes_values = tree.value
    # If root note doesn't contain desired label in children nodes, return empty list
    if nodes_values[0, :, label] == 0:
        return []

    tree_paths = []
    _recursively_search_tree(0, label, [], tree, tree_paths)

    return tree_paths


def _recursively_search_tree(node_id: int, leaf_label: int, path_history: list, tree, tree_paths: list):
    children_left_node_id = tree.children_left[node_id]
    children_right_node_id = tree.children_right[node_id]
    nodes_values = tree.value

    if tree.children_left[children_left_node_id] != tree.children_right[children_left_node_id]:
        if nodes_values[children_left_node_id, :, leaf_label] > 0:
            new_path_history = path_history.copy() + [[tree.feature[node_id], -1, tree.threshold[node_id]]]
            _recursively_search_tree(children_left_node_id, leaf_label, new_path_history, tree, tree_paths)
    else:
        if np.argmax(nodes_values[children_left_node_id, 0, :]) == leaf_label:
            new_path_history = path_history.copy() + [[tree.feature[node_id], -1, tree.threshold[node_id]]]
            tree_paths.append(new_path_history)

    if tree.children_left[children_right_node_id] != tree.children_right[children_right_node_id]:
        if nodes_values[children_right_node_id, :, leaf_label] > 0:
            new_path_history = path_history.copy() + [[tree.feature[node_id], 1, tree.threshold[node_id]]]
            _recursively_search_tree(children_right_node_id, leaf_label, new_path_history, tree, tree_paths)
    else:
        if np.argmax(nodes_values[children_right_node_id, 0, :]) == leaf_label:
            new_path_history = path_history.copy() + [[tree.feature[node_id], 1, tree.threshold[node_id]]]
            tree_paths.append(new_path_history)

    return False
