import numpy as np
import pandas as pd
from scipy.stats import rankdata
from collections import defaultdict

from rf_explainer import RandomForestExplainer


def visualize(rfe: RandomForestExplainer, X: pd.Series, X_prime: pd.Series, label_encoder_dict=None):
    """ Visualize a difference between original instance and its counterfactual """
    difference = X_prime - X
    # difference_rank = pd.Series(rankdata(np.abs(difference.values), method='dense'), index=X.index)
    constraints = pd.Series([_map_constraint(i, rfe) for i in range(X.shape[0])], index=X.index)
    X = X.round(3)
    X_prime = X_prime.round(3)
    difference = difference.round(3)
    if label_encoder_dict and len(rfe.categorical_features) > 0:
        X.loc[rfe.categorical_features] = X.loc[rfe.categorical_features].to_frame(0).apply(lambda x: label_encoder_dict[x.name].inverse_transform(x)[0], axis=1)
        X_prime.loc[rfe.categorical_features] = X_prime.loc[rfe.categorical_features].to_frame(0).apply(lambda x: label_encoder_dict[x.name].inverse_transform(x.astype(int))[0], axis=1)
        difference.loc[rfe.categorical_features] = difference.loc[rfe.categorical_features].apply(lambda x: 1.0 if x != 0.0 else 0.0)
    result = pd.concat([X.T, X_prime.T, difference.T, constraints.T], axis=1, ignore_index=True)
    result.columns = ["X", "X'", "difference", "constraints"]
    return result



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
