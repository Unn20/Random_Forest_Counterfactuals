import numpy as np
import pandas as pd
import time
import os
import pickle
import subprocess
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


DATASET_PATH = "./"
RESULTS_PATH = "results_tolomei/"
CHUNKS = [i*5 for i in range(1, 11)]
# CHUNKS = [i for i in range(1, 11)]
METRICS = ('euclidean', 'cosine', 'unmatched_components', 'jaccard', 'pearson_correlation')
EPSILON=0.1
PRINT_OUTPUT = False

RETRY_EXPERIMENT = 5
n_jobs = 4

if __name__ == '__main__':
    for chunk in CHUNKS:
        out_dir = f"tolomei_temp_{chunk}\\"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        dataset_path = os.path.join(DATASET_PATH, f"adult_part_{chunk}.csv")
        adult_dataset = pd.read_csv(dataset_path)
        X_train = adult_dataset.loc[:, adult_dataset.columns!="class"]
        y_train = adult_dataset["class"]
        rf = pickle.load(open(f"rf_{chunk}.pkl", 'rb'))
        print(f"Chunk: [{chunk}], dataset shape={X_train.shape}, trees_to_process={len(rf.estimators_)}")

        start_time = time.time()
        pipe = subprocess.Popen([".\\tolomei_venv\\Scripts\\python.exe", "..\\ml-feature-tweaking\\dump_paths.py", f"rf_{chunk}.pkl",
         os.path.join(out_dir, f"tolomei_paths_{chunk}.txt")], shell=True, stderr=subprocess.STDOUT if PRINT_OUTPUT else None)
        pipe.wait()
        path_dump_time = time.time() - start_time
        print(f"Positive paths for [{chunk}] extracted in {path_dump_time:1.2f}s")

        for _ in range(RETRY_EXPERIMENT):
            start_time = time.time()
            pipe = subprocess.Popen([".\\tolomei_venv\\Scripts\\python.exe", "..\\ml-feature-tweaking\\tweak_features.py", dataset_path, f"rf_{chunk}.pkl",
             os.path.join(out_dir, f"tolomei_paths_{chunk}.txt"), out_dir, f"--epsilon={EPSILON}"], shell=True, stderr=subprocess.STDOUT if PRINT_OUTPUT else None)
            pipe.wait()
            print("cfs have been found")
            pipe = subprocess.Popen([".\\tolomei_venv\\Scripts\\python.exe", "..\\ml-feature-tweaking\\compute_tweaking_costs.py", dataset_path,
             os.path.join(out_dir,f"transformations_{EPSILON}.tsv"), out_dir, "--costfuncs=unmatched_component_rate,euclidean_distance,cosine_distance,jaccard_distance,pearson_correlation_distance"],
              shell=True, stderr=subprocess.STDOUT if PRINT_OUTPUT else None)
            pipe.wait()
            end_time = time.time()
            total_time = (end_time - start_time) + (1 / RETRY_EXPERIMENT) * path_dump_time
            cfs = pd.read_csv(f"tolomei_temp_{chunk}\\transformations_0.1.tsv", sep='\t')
            scores = np.array([X_train.shape[0], X_train.shape[1], len(rf.estimators_), cfs.shape[0], total_time])
            np.savetxt(os.path.join(RESULTS_PATH, f"tolomei_part_{chunk}_{_}.txt"), scores)
            print(f"Experiment: {_+1}/{RETRY_EXPERIMENT} result time: {total_time: 1.2f}s")

        try:
            cfs = pd.read_csv(f"tolomei_temp_{chunk}\\transformations_0.1.tsv", sep='\t')
            predictions = rf.predict(cfs.iloc[:, 4:])
            cfs['predicted_class'] = predictions
            cfs.to_csv(os.path.join(RESULTS_PATH, f"cfs_{chunk}.csv"), index=False)
        except Exception as e:
            print(e)