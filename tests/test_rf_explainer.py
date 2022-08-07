from rf_counterfactuals.rf_explainer import _get_positive_paths_from_decision_tree, RandomForestExplainer, _check_row_frozen_validity
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import pickle

import pytest


def test_features_parameter_validation():
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    with pytest.raises(Exception, match=r"'categorical_features' parameter must be a list"):
        rfe = RandomForestExplainer(clf, X, y, categorical_features=0)
    with pytest.raises(Exception, match=r"'left_frozen_features' parameter must be a list"):
        rfe = RandomForestExplainer(clf, X, y, left_frozen_features=0)
    with pytest.raises(Exception, match=r"'frozen_features' parameter must be a list"):
        rfe = RandomForestExplainer(clf, X, y, frozen_features='feature')

    with pytest.raises(Exception, match=r"Feature index *"):
        rfe = RandomForestExplainer(clf, X, y, frozen_features=['feature_0', 1, 2])
    with pytest.raises(Exception, match=r"Feature index *"):
        rfe = RandomForestExplainer(clf, X, y, left_frozen_features=['feature_0', 1, 2])
    with pytest.raises(Exception, match=r"Feature index *"):
        rfe = RandomForestExplainer(clf, X, y, left_frozen_features=[-1, 0, 1, 2])
    with pytest.raises(Exception, match=r"Feature index *"):
        rfe = RandomForestExplainer(clf, X, y, right_frozen_features=[0, 1, 10, 100, 1000, 10000, 100000, 9999999999])

    rfe = RandomForestExplainer(clf, X, y)
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == []
    assert rfe.right_frozen_features == []

    rfe = RandomForestExplainer(clf, X, y, frozen_features=[0, 1, 2])
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == [0, 1, 2]
    assert rfe.right_frozen_features == [0, 1, 2]

    rfe = RandomForestExplainer(clf, X, y, frozen_features=[2, 1, 0])
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == [0, 1, 2]
    assert rfe.right_frozen_features == [0, 1, 2]

    rfe = RandomForestExplainer(clf, X, y, frozen_features=[0], left_frozen_features=[1, 2], right_frozen_features=[2])
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == [0, 1, 2]
    assert rfe.right_frozen_features == [0, 2]

    rfe = RandomForestExplainer(clf, X, y, left_frozen_features=[1, 2, 0], right_frozen_features=[2])
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == [0, 1, 2]
    assert rfe.right_frozen_features == [2]

    rfe = RandomForestExplainer(clf, X, y, left_frozen_features=[0, 0, 0], right_frozen_features=[0, 0, 1, 1, 2, 2])
    assert rfe.categorical_features == []
    assert rfe.left_frozen_features == [0]
    assert rfe.right_frozen_features == [0, 1, 2]

    rfe = RandomForestExplainer(clf, X, y, categorical_features=[0, 1, 2])
    assert rfe.categorical_features == [0, 1, 2]
    assert rfe.left_frozen_features == []
    assert rfe.right_frozen_features == []

    rfe = RandomForestExplainer(clf, X, y, categorical_features=[2, 1, 0])
    assert rfe.categorical_features == [0, 1, 2]
    assert rfe.left_frozen_features == []
    assert rfe.right_frozen_features == []

    rfe = RandomForestExplainer(clf, X, y, categorical_features=[0, 0, 1, 2, 1, 2])
    assert rfe.categorical_features == [0, 1, 2]
    assert rfe.left_frozen_features == []
    assert rfe.right_frozen_features == []


def test_check_row_frozen_validity():
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([0, 1, 2]), [], []) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([0, 1, 2]), [1], [0]) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([0, 1, 2]), [0, 1, 2], [0, 1, 2]) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [], []) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [], [0]) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [0], []) == False
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [1], []) == True
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [], [1]) == False
    assert _check_row_frozen_validity(np.array([0, 1, 2]), np.array([-1, 2, 0]), [0, 2], [1]) == False


def test_get_positive_paths_from_tree():
    def match_paths(model_path, ground_truth_path):
        if len(model_path) != len(ground_truth_path):
            return False
        for no, step in enumerate(model_path):
            if model_path[no][0] != ground_truth_path[no][0]:
                return False
            if model_path[no][1] != ground_truth_path[no][1]:
                return False
        return True

    ground_truth = [[[3, 1], [3, -1], [2, -1], [3, 1]],
                    [[3, 1], [3, -1], [2, 1]],
                    [[3, 1], [3, 1]]
                    ]
    incorrect_path = list(range(len(ground_truth)))
    clf = pickle.load(open("pickled_trees/tree_for_tests.pkl", 'rb'))
    positive_paths = _get_positive_paths_from_decision_tree(clf, 2)
    for path in positive_paths:
        for no, target_path in enumerate(ground_truth):
            if no not in incorrect_path:
                continue
            if match_paths(path, target_path):
                incorrect_path.remove(no)
                break

    assert incorrect_path == []