from rf_counterfactuals.rf_explainer import _get_positive_paths_from_decision_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import pickle


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