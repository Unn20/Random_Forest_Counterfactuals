import numpy as np
import pandas as pd
import time
import pickle
from subprocess import call
import random

from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rf_counterfactuals import RandomForestExplainer


from sklearn.metrics import accuracy_score, f1_score

DATASET_PATH = r"C:\Users\macio\JupyterNotebook\Magisterka\datasets\adult"

if __name__ == '__main__':
    np.random.seed(420)
    miniadult_dataset = pd.read_csv(DATASET_PATH + "\\miniadult_modified.csv")
    miniadult_dataset['class'] = np.random.randint(-1, 4, size=miniadult_dataset.shape[0])

    feature_names = [c for c in miniadult_dataset.columns if c != "class"]
    class_names = list(miniadult_dataset["class"].unique())

    train, test = train_test_split(miniadult_dataset, train_size=0.67, random_state=420,
                                   stratify=miniadult_dataset["class"])
    X_train = train.loc[:, train.columns != "class"]
    y_train = train["class"]

    rf = RandomForestClassifier(n_estimators=10, random_state=420)
    rf.fit(X_train, y_train)

    rfe = RandomForestExplainer(rf)
    start_time = time.time()
    cfs = rfe.explain(X_train.iloc[:100, :], 1, metrics=('unmatched_components', 'k_nearest_neighborhood'), n_jobs=2)
    end_time = time.time()

    total_cfs = sum([len(c) for c in cfs])
    total_time = end_time - start_time

    print(f"Total counterfactuals found: {total_cfs}")

    print(f"Finished in {total_time: 1.4f}s")
