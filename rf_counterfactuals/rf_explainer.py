from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import warnings
from functools import partial
from joblib import Parallel, delayed
import itertools
import sys

from single_cf_costs_functions import *
from multi_cf_costs_functions import *

class RandomForestExplainer:
    available_loss_functions = {
        'euclidean': lambda x, x_prime: euclidean_distance(x, x_prime),
        'euclidean_categorical': lambda x, x_prime, cat_features, non_cat_features: euclidean_categorical_distance(x, x_prime, cat_features, non_cat_features),
        'hoem': lambda x, x_prime, feature_range, cat_features, non_cat_features: heterogeneous_euclidean_overlap_metric(x, x_prime, feature_range, cat_features, non_cat_features),
        'cosine': lambda x, x_prime: cosine_distance(x, x_prime),
        'jaccard': lambda x, x_prime: jaccard_distance(x, x_prime),
        'pearson_correlation': lambda x, x_prime: pearson_correlation_distance(x, x_prime),
        'unmatched_components': lambda x, x_prime: unmatched_components_distance(x, x_prime),
        'k_nearest_neighborhood': lambda x_prime, label, nbrs, y_train: k_nearest_neighborhood(x_prime, label, nbrs, y_train),
        'implausibility_single': lambda x_prime, nbrs, k: implausibility_single(x_prime, nbrs, k),
    }

    def __init__(self, model: RandomForestClassifier, X_train, y_train, categorical_features=None, frozen_features=None, left_frozen_features=None, right_frozen_features=None):
        self.rf_model = model
        self.decision_trees = model.estimators_
        self.classes = list(model.classes_)
        self.get_class_for_dt = lambda label: self.classes.index(label)
        self.n_features = model.n_features_in_
        self.positive_paths = dict()

        self.X_train = X_train
        self.y_train = y_train

        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_max = np.max(X_train, axis=0)
        X_train_min = np.min(X_train, axis=0)
        X_train_range = X_train_max - X_train_min
        self.X_train_stats = {'min': X_train_min, 'max': X_train_max, 'range': X_train_range,
                              'mean': X_train_mean, 'std': X_train_std, 'labels': y_train}

        self.nbrs = {}
        self.nbrs_same_label = {}

        categorical_features = [] if categorical_features is None else categorical_features
        frozen_features = [] if frozen_features is None else frozen_features
        left_frozen_features = [] if left_frozen_features is None else left_frozen_features
        right_frozen_features = [] if right_frozen_features is None else right_frozen_features

        for name, list_parameter in {'categorical_features': categorical_features, 'frozen_features': frozen_features,
                                     'left_frozen_features': left_frozen_features, 'right_frozen_features': right_frozen_features}.items():
            if type(list_parameter) != list:
                raise Exception(f"'{name}' parameter must be a list")
            for feature_index in list_parameter:
                if type(feature_index) != int or (feature_index < 0 or feature_index > self.n_features-1):
                    raise Exception(f"Feature index '{feature_index}' in parameter '{name}' must be integer in range [0, {self.n_features-1}]")

        self.categorical_features = sorted(list(set(categorical_features)))
        self.non_categorical_features = [i for i in range(self.X_train.shape[1]) if i not in self.categorical_features]
        # We treat frozen_features as they are left_ and right_ frozen, so it's simpler
        for feature_index in frozen_features:
            left_frozen_features.append(feature_index)
            right_frozen_features.append(feature_index)
        self.left_frozen_features = sorted(list(set(left_frozen_features)))
        self.right_frozen_features = sorted(list(set(right_frozen_features)))

    def explain_with_single_metric(self, X, label, limit=1, eps=0.1, metric='euclidean', k=5, to_single_df=False, n_jobs=-1):
        """ Explain given instances with single metric """
        if eps <= 0.0:
            raise Exception(f"Eps parameter={eps} (must be greater than 0.0).")
        if X.ndim == 1:
            warnings.warn(f"X has ndim={X.ndim}. Reshaping: X = X.values.reshape(1, -1)", UserWarning)
            X = X.values.reshape(1, -1)
        elif X.ndim == 2:
            pass
        else:
            raise Exception(f"Incorrect dimensions of X={X.ndim} (should be 2).")

        if type(limit) != int and limit is not None:
            raise Exception(f"Incorrect 'limit' parameter (limit={limit}). It must be integer type or None.")

        y_hat_ensemble = self.rf_model.predict(X)
        wrong_label = np.where(y_hat_ensemble != label)[0]
        X_wrong_label = X.iloc[wrong_label, :]

        counterfactuals, costs_per_row = self._get_artificial_counterfactuals(X, wrong_label, y_hat_ensemble, label, 0.1, (metric,), k, n_jobs=n_jobs)

        # Put whole calculation results into one DataFrame
        cfs_index = np.zeros(X.shape[0])
        cfs_index[wrong_label] = 1
        result = self._select_counterfactuals(X_wrong_label, counterfactuals, costs_per_row, limit, cfs_index)

        if to_single_df:
            result = pd.concat(result, ignore_index=False)
        return result

    def explain_with_multiple_metrics(self, X, label, limit=1, eps=0.1, metrics=('unmatched_components', 'euclidean'), k=5, to_single_df=False, n_jobs=-1):
        """ Explain given instances with multiple metrics """
        if eps <= 0.0:
            raise Exception(f"Eps parameter={eps} (must be greater than 0.0).")
        if X.ndim == 1:
            warnings.warn(f"X has ndim={X.ndim}. Reshaping: X = X.values.reshape(1, -1)", UserWarning)
            X = X.values.reshape(1, -1)
        elif X.ndim == 2:
            pass
        else:
            raise Exception(f"Incorrect dimensions of X={X.ndim} (should be 2).")

        y_hat_ensemble = self.rf_model.predict(X)
        wrong_label = np.where(y_hat_ensemble != label)[0]
        X_wrong_label = X.iloc[wrong_label, :]

        counterfactuals, costs_per_row = self._get_artificial_counterfactuals(X, wrong_label, y_hat_ensemble, label, 0.1, metrics, k, n_jobs=n_jobs)

        cfs_index = np.zeros(X.shape[0])
        cfs_index[wrong_label] = 1
        result = self._select_counterfactuals_pareto_front(X_wrong_label, counterfactuals, costs_per_row, cfs_index)
        if to_single_df:
            result = pd.concat(result, ignore_index=False)
        return result


    def _get_positive_paths_for_label(self, label):
        """ Get positive path for given label and save paths to cache memory """
        if label not in self.classes:
            raise Exception(f"This label: '{label}' doesn't belong to the dataset classes: '{self.classes}'")
        if label not in self.positive_paths:
            self.positive_paths[label] = _get_positive_paths(self.rf_model, self.get_class_for_dt(label))
        return self.positive_paths[label]

    def _get_artificial_counterfactuals(self, X, wrong_label, y_hat_ensemble, label, eps, metrics, k, n_jobs=-1):
        """ Get counterfactual candidates """
        # Precalculations
        X_wrong_label = X.iloc[wrong_label, :]

        # TODO: It can be calculated concurrently
        sys.stdout.write(
            f"[1/3] Extracting positive paths.\n")
        positive_paths = self._get_positive_paths_for_label(label)
        feature_eps_mul_std = eps * self.X_train_stats['std']

        sys.stdout.write(
            f"[2/3] Generating counterfactual examples for each tree. Total number of tasks: {len(self.decision_trees)}\n")
        results_per_tree = Parallel(n_jobs=n_jobs, verbose=10, prefer='processes')(
            delayed(partial(_tweak_features, X_wrong_label, label, self.classes, self.rf_model, feature_eps_mul_std,
                            self.categorical_features))(dt, pp) for dt, pp in zip(self.decision_trees, positive_paths))

        # results_per_tree have shape (trees_no, X.shape[0], CFs) so we need to concat CFs along 0 dimension
        rpt_iterator = iter(np.array(results_per_tree, dtype=object).T)
        counterfactuals = [pd.DataFrame(itertools.chain.from_iterable(next(rpt_iterator))).drop_duplicates() for no in range(X.shape[0]) if
                           no in wrong_label]

        params = {'k': k, 'y_hat_ensemble': y_hat_ensemble, 'label': label, 'y_train': self.y_train.values}

        if 'k_nearest_neighborhood' in metrics:
            if k not in self.nbrs:
                self.nbrs[k] = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
                                                metric=heterogeneous_euclidean_overlap_metric,
                                                metric_params={'feature_range': self.X_train_stats['range'],
                                               'cat_features': self.categorical_features,
                                               'non_cat_features': self.non_categorical_features})
                self.nbrs[k].fit(self.X_train.values)
            params['nbrs'] = self.nbrs[k]

        if 'implausibility_single' in metrics:
            if (label, k) not in self.nbrs_same_label:
                self.nbrs_same_label[(label, k)] = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
                                                metric=heterogeneous_euclidean_overlap_metric,
                                                metric_params={'feature_range': self.X_train_stats['range'],
                                               'cat_features': self.categorical_features,
                                               'non_cat_features': self.non_categorical_features})
                X_train_same_labels = self.X_train[self.y_train == label]
                self.nbrs_same_label[(label, k)].fit(X_train_same_labels.values)
            params['nbrs_same_label'] = self.nbrs_same_label[(label, k)]

        params['loss_functions_names'] = metrics
        params['loss_functions'] = {k: v for k, v in RandomForestExplainer.available_loss_functions.items() if
                                    k in metrics}

        sys.stdout.write(f"[3/3] Calculating loss function. Total number of tasks: {len(counterfactuals)}\n")
        costs_per_row = Parallel(n_jobs=n_jobs if len(counterfactuals) * 4 > n_jobs else 1, verbose=10, prefer='processes')(
            delayed(partial(_calculate_loss, params, self.X_train_stats['range'], self.X_train_stats['mean'],
                            self.X_train_stats['std'], self.categorical_features, self.non_categorical_features))
            (x[1], x_primes) for x, x_primes in zip(X_wrong_label.iterrows(), counterfactuals))

        return counterfactuals, costs_per_row


    def _select_counterfactuals(self, original_rows, counterfactuals, costs, limit, cfs_index):
        """ Select first 'limit' counterfactual examples that meet certain constraints """
        result = []
        cfs_counter = 0
        for no, is_csf in enumerate(cfs_index):
            if is_csf and len(counterfactuals[cfs_counter]) > 0:
                cfs = counterfactuals[cfs_counter]
                cfs['_loss'] = costs[cfs_counter]
                cfs['_loss'] = cfs['_loss'].apply(lambda x: x[0] if type(x) == list else x)

                # Check actionability of counterfactual
                cfs = cfs[cfs.apply(lambda row: _check_row_frozen_validity(original_rows.iloc[cfs_counter, :].values, row.values,
                                                                          self.left_frozen_features,
                                                                          self.right_frozen_features), axis=1)]
                if len(cfs) == 0:
                    # If there are no candidates, continue the loop
                    result.append(pd.DataFrame({}))
                    continue

                sorted_cfs = cfs.sort_values('_loss')
                sorted_cfs = sorted_cfs.drop('_loss', axis=1)
                if limit is None:
                    result.append(sorted_cfs.iloc[:, :])
                else:
                    result.append(sorted_cfs.iloc[:limit, :])
                cfs_counter += 1
            else:
                result.append(pd.DataFrame({}))

        return result

    def _select_counterfactuals_pareto_front(self, original_rows, counterfactuals, costs, cfs_index):
        """ Select counterfactual examples from first pareto front that meet certain constraints """
        result = []
        cfs_counter = 0
        for no, is_csf in enumerate(cfs_index):
            if is_csf:
                cfs = counterfactuals[cfs_counter]

                to_process = np.ones(cfs.shape[0], dtype=bool)

                # Check actionability of counterfactual
                actionable = cfs.apply(
                    lambda row: _check_row_frozen_validity(original_rows.iloc[cfs_counter, :].values, row.values,
                                                           self.left_frozen_features,
                                                           self.right_frozen_features), axis=1)

                to_process[~actionable] = False

                cfs = cfs.iloc[to_process, :]

                cfs_costs = np.array(costs[cfs_counter])
                pareto_mask = _is_pareto_efficient(cfs_costs[to_process], return_mask=True)

                result.append(cfs.iloc[pareto_mask, :])
                cfs_counter += 1
            else:
                result.append(pd.DataFrame({}))

        return result


def _check_row_frozen_validity(x, cf, left_frozen, right_frozen):
    """ Check actionability of counterfactual """
    for left_frozen_feature in left_frozen:
        if cf[left_frozen_feature] < x[left_frozen_feature]:
            return False
    for right_frozen_feature in right_frozen:
        if cf[right_frozen_feature] > x[right_frozen_feature]:
            return False
    return True


def _calculate_loss(params: dict, feature_range, feature_means, feature_std, categorical_features: list, non_categorical_features: list, X_row: np.array, X_primes: pd.DataFrame):
    """ Calculate given loss metrics for counterfactual """
    result = []
    zscore_normalized_X_primes = (X_primes - feature_means) / feature_std
    zscore_normalized_X_row = (X_row.values - feature_means) / feature_std
    zscore_normalized_X_row[zscore_normalized_X_row.isna()] = 0

    for X_prime_no in range(zscore_normalized_X_primes.shape[0]):
        scores = []
        X_prime = X_primes.iloc[X_prime_no, :]
        normalized_X_prime = zscore_normalized_X_primes.iloc[X_prime_no, :]
        normalized_X_prime[normalized_X_prime.isna()] = 0
        for func_name, func in params['loss_functions'].items():
            if func_name == 'k_nearest_neighborhood':
                loss = 1 - func(X_prime.values, params['label'], params['nbrs'], params['y_train'])
            elif func_name == 'implausibility_single':
                loss = func(X_prime.values, params['nbrs_same_label'], params['k'])
            elif func_name == 'euclidean_categorical':
                loss = func(zscore_normalized_X_row.values, normalized_X_prime.values, categorical_features, non_categorical_features)
            elif func_name == 'hoem':
                loss = func(X_row.values, X_prime.values, feature_range, categorical_features,
                            non_categorical_features)
            else:
                loss = func(zscore_normalized_X_row.values, normalized_X_prime.values)
            scores.append(loss)
        result.append(scores)
    return result


def _is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.

    From: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def _tweak_features(X, label, classes, rf, feature_eps_mul_std, categorical_features, decision_tree, positive_paths):
    """ Tweak features to find counterfactual candidates for each tree """
    predicted_classes_by_tree = np.array(decision_tree.predict(X.values), dtype=int)
    y_hat_tree = np.take(classes, predicted_classes_by_tree)

    to_tweak = np.where(y_hat_tree != label)[0]
    X_to_tweak = X.iloc[to_tweak, :]

    X_tweaked_candidates = [[] for _ in range(X.shape[0])]

    for positive_path in positive_paths:
        X_prime = X_to_tweak.copy()
        for feature_id, sign, threshold in positive_path:
            feature_values = X_prime.iloc[:, feature_id]

            if sign == 1:
                rows_to_tweak = feature_values.loc[(feature_values < threshold)].index
                if feature_id in categorical_features:
                    X_prime.loc[rows_to_tweak, X_prime.columns[feature_id]] = np.ceil(threshold + (
                        feature_eps_mul_std[feature_id]))
                else:
                    X_prime.loc[rows_to_tweak, X_prime.columns[feature_id]] = threshold + (
                                feature_eps_mul_std[feature_id])
            else:
                rows_to_tweak = feature_values.loc[(feature_values >= threshold)].index
                if feature_id in categorical_features:
                    X_prime.loc[rows_to_tweak, X_prime.columns[feature_id]] = np.floor(threshold - (
                        feature_eps_mul_std[feature_id]))
                else:
                    X_prime.loc[rows_to_tweak, X_prime.columns[feature_id]] = threshold - (
                                feature_eps_mul_std[feature_id])

        y_hat_prime = rf.predict(X_prime)
        candidates_indices = np.where(y_hat_prime == label)[0]
        for i in candidates_indices:
            X_tweaked_candidates[to_tweak[i]].append(X_prime.iloc[i, :])

    return X_tweaked_candidates


def _get_positive_paths(model: RandomForestClassifier, label: int):
    """ Get all positive paths from Random Forest Classifier """
    positive_paths = []
    for decision_tree_classifier in model.estimators_:
        tree_paths = _get_positive_paths_from_decision_tree(decision_tree_classifier, label)
        positive_paths.append(tree_paths)
    return positive_paths


def _get_positive_paths_from_decision_tree(decision_tree_classifier: DecisionTreeClassifier, label: int):
    """ Get all positive paths from a single Decision Tree Classifier"""
    tree = decision_tree_classifier.tree_
    nodes_values = tree.value
    # If root note doesn't contain desired label in children nodes, return empty list
    if nodes_values[0, :, label] == 0:
        return []

    tree_paths = []
    _recursively_search_tree(0, label, [], tree, tree_paths)

    return tree_paths


def _recursively_search_tree(node_id: int, leaf_label: int, path_history: list, tree, tree_paths: list):
    """ Recursively go through tree to find positive paths """
    children_left_node_id = tree.children_left[node_id]
    children_right_node_id = tree.children_right[node_id]
    nodes_values = tree.value

    # Check if left children is leaf node
    if tree.children_left[children_left_node_id] != tree.children_right[children_left_node_id]:
        # If no, check if it can have still a path to target label, then process it recursively
        if nodes_values[children_left_node_id, :, leaf_label] > 0:
            new_path_history = path_history.copy() + [[tree.feature[node_id], -1, tree.threshold[node_id]]]
            _recursively_search_tree(children_left_node_id, leaf_label, new_path_history, tree, tree_paths)
    else:
        # If it is a leaf node, check if it contains target label
        if np.argmax(nodes_values[children_left_node_id, 0, :]) == leaf_label:
            new_path_history = path_history.copy() + [[tree.feature[node_id], -1, tree.threshold[node_id]]]
            tree_paths.append(new_path_history)

    # Check if right children is leaf node
    if tree.children_left[children_right_node_id] != tree.children_right[children_right_node_id]:
        # If no, check if it can have still a path to target label, then process it recursively
        if nodes_values[children_right_node_id, :, leaf_label] > 0:
            new_path_history = path_history.copy() + [[tree.feature[node_id], 1, tree.threshold[node_id]]]
            _recursively_search_tree(children_right_node_id, leaf_label, new_path_history, tree, tree_paths)
    else:
        # If it is a leaf node, check if it contains target label
        if np.argmax(nodes_values[children_right_node_id, 0, :]) == leaf_label:
            new_path_history = path_history.copy() + [[tree.feature[node_id], 1, tree.threshold[node_id]]]
            tree_paths.append(new_path_history)
