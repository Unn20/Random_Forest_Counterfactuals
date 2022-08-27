import numpy as np
import pandas as pd
import time
import os
import pickle

from rf_counterfactuals import RandomForestExplainer

RETRY_EXPERIMENT = 5
n_jobs = 1
DATASET_PATH = "./"
RESULTS_PATH = f"results_rfc_{n_jobs}/"
CHUNKS = [i*5 for i in range(1, 11)]
# CHUNKS = [i for i in range(1, 11)]
METRICS = ('euclidean', 'cosine', 'unmatched_components', 'jaccard', 'pearson_correlation')
EPSILON=0.1



if __name__ == '__main__':
    for chunk in CHUNKS:
        adult_dataset = pd.read_csv(os.path.join(DATASET_PATH, f"adult_part_{chunk}.csv"))
        X_train = adult_dataset.loc[:, adult_dataset.columns!="class"]
        y_train = adult_dataset["class"]

        rf = pickle.load(open(f"rf_{chunk}.pkl", 'rb'))
        print(f"Chunk: [{chunk}], dataset shape={X_train.shape}, trees_to_process={len(rf.estimators_)}")
        y_hat = rf.predict(X_train)

        true_negatives = y_train[y_train==-1]==y_hat[y_train==-1]
        true_negatives = true_negatives[true_negatives].index
        X_true_negatives = X_train.loc[true_negatives]

        # print(X_true_negatives.shape[0])

        for _ in range(RETRY_EXPERIMENT):
            start_time = time.time()
            rfe = RandomForestExplainer(rf)
            cfs = rfe.explain(X_true_negatives, 1, eps=EPSILON, metrics=METRICS, n_jobs=n_jobs)
            end_time = time.time()
            cfs = cfs.drop_duplicates()
            cf_no = len(cfs)
            total_time = end_time - start_time
            scores = np.array([X_train.shape[0], X_train.shape[1], len(rf.estimators_), cf_no, total_time])            
            np.savetxt(os.path.join(RESULTS_PATH, f"rfc_part_{chunk}_{_}.txt"), scores)
            print(f"Experiment: {_+1}/{RETRY_EXPERIMENT} result time: {total_time: 1.2f}s")

        try:
            predictions = rf.predict(cfs.iloc[:, 1:-5])
            cfs['predicted_class'] = predictions
            cfs.to_csv(os.path.join(RESULTS_PATH, f"cfs_{chunk}.csv"), index=False)
        except Exception as e:
            print(e)
