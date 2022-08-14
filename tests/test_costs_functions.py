import pytest

from rf_counterfactuals.single_cf_costs_functions import *
from rf_counterfactuals.multi_cf_costs_functions import *

left_vector_case_0 = [
    [],
    [0],
    [0, 0, 0],
    [0, 1, 2],
    [5, 5, 5],
    [1.0, 10.0, 100.0],
    [1, 2, 3, 4, 5],
    [-1.0, 0.0, 1.0],
    [1.23, 4.56, 7.89],
    [1.23, 4.56, 7.89],
    [1.23, 4.56, 7.89],
]

right_vector_case_0 = [
    [],
    [1],
    [1, 1, 0],
    [0, 1, 2],
    [6, 6, 6],
    [1.0, 10.0, 10.0],
    [5, 2, 3, 4, 5],
    [-1.0, 0.0, 1.0],
    [1.23, 4.56, 7.89],
    [1.50, 4.20, 7.89],
    [1.22999999, 4.55999999, 7.89000001],
]

left_vector_case_1 = [
    [1, 0.0, 1, 0.0],
    [0, 2.0, 1, 2.0],
    [6, 6.0, 6, 3.0],
    [65, 10.0, 17, 15.0],
    [0, 0, 0, 0]
]

right_vector_case_1 = [
    [0, 0.0, 0, 0.0],
    [0, 2.0, 1, 4.0],
    [5, 5.5, 5, 3.0],
    [25, 12.5, 17, 12.5],
    [1, 15.0, 1, 20.0]
]

def test_unmatched_components():
    eps = 1e-4
    # Manual calculation
    ground_truth = np.array([0, 1, 2, 0, 3, 1, 1, 0, 0, 2, 0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = unmatched_components(left, right, eps=eps)
        assert pytest.approx(ground_truth[no], rel=1e-4) == result, f"left={left}; right={right}"


def test_unmatched_components_distance():
    eps = 1e-4
    # Manual calculation
    ground_truth = np.array([0.0, 1/1, 2/3, 0/3, 3/3, 1/3, 1/5, 0/3, 0/3, 2/3, 0/3])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = unmatched_components_distance(left, right, eps=eps)
        assert pytest.approx(ground_truth[no], rel=1e-4) == result, f"left={left}; right={right}"


def test_heterogeneous_euclidean_overlap_metric():
    feature_range = np.array([7, 10, 100, 0, 0])
    categorical_features = []
    non_categorical_features = [0, 1, 2, 3, 4]
    ground_truth = np.array([0.0, 1/7, math.sqrt((1/7)**2+(1/10)**2),
                             0.0, math.sqrt((1/7)**2+(1/10)**2+(1/100)**2),
                             90/100, math.sqrt((4/7)**2+1+1), 0.0, 0.0,
                             math.sqrt((0.27/7)**2+(0.36/10)**2), 0.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = heterogeneous_euclidean_overlap_metric(left, right, feature_range[:left.shape[0]], categorical_features[:left.shape[0]],
                                                non_categorical_features[:left.shape[0]])
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"

    feature_range = np.array([999, 12.5, 999, 15.0])
    categorical_features = [0, 2]
    non_categorical_features = [1, 3]
    ground_truth = np.array([math.sqrt(2), math.sqrt((2/15)**2), math.sqrt(1+(0.5/12.5)**2+1), math.sqrt(1+(2.5/12.5)**2+(2.5/15)**2), 2.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_1[no]), np.array(right_vector_case_1[no])
        result = heterogeneous_euclidean_overlap_metric(left, right, feature_range[:left.shape[0]], categorical_features, non_categorical_features)
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"


def test_euclidean_categorical_distance():
    # Calculated with Wolfram Alpha
    categorical_features = []
    non_categorical_features = [0, 1, 2, 3, 4]
    ground_truth = np.array([0.0, 1.0, 1.41421, 0.0, 1.73205, 90.0, 4.0, 0.0, 0.0, 0.45, 0.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = euclidean_categorical_distance(left, right, categorical_features[:left.shape[0]], non_categorical_features[:left.shape[0]])
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"

    categorical_features = [0, 2]
    non_categorical_features = [1, 3]
    ground_truth = np.array([2.0, 2.0, 2.5, 4.5355])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_1[no]), np.array(right_vector_case_1[no])
        result = euclidean_categorical_distance(left, right, categorical_features, non_categorical_features)
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"


def test_euclidean_distance():
    # Calculated with Wolfram Alpha
    ground_truth = np.array([0.0, 1.0, 1.41421, 0.0, 1.73205, 90.0, 4.0, 0.0, 0.0, 0.45, 0.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = euclidean_distance(left, right)
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"


def test_cosine_distance():
    # Calculated using this https://onlinemschool.com/math/assistance/vector/angl/
    ground_truth = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0-0.77269, 1.0-0.89507, 0.0, 0.0, 1.0-0.99889, 0.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = cosine_distance(left, right)
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"


def test_jaccard_distance():
    # Manual calculation
    ground_truth = np.array([None, 1-0.0, 1-1/2, 1-3/3, 1-0/2, 1-2/3, 1-4/5, 1-3/3, 1-3/3, 1-1/5, 1-0/3])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        try:
            result = jaccard_distance(left, right)
        except ZeroDivisionError:
            assert len(left) == 0
            continue
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-6) == result, f"left={left}; right={right}"


def test_pearson_correlation_distance():
    # Calculated using this https://www.statskingdom.com/correlation-calculator.html
    ground_truth = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0-0.5695, 1.0-0.2425, 0.0, 0.0, 1.0 - 0.99603, 0.0])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        try:
            result = pearson_correlation_distance(left, right)
        except ValueError:
            assert len(left) < 2
            continue
        assert pytest.approx(ground_truth[no], rel=1e-4, abs=1e-5) == result, f"left={left}; right={right}"

def test_implausibility_single():
    X = np.array([[-1, -1], [2, -1], [3, -2], [1, 1], [2, 1], [3, 2], [5, -1], [1, -3]])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 0])

    cat_features = []
    non_cat_features = [0, 1]
    feature_range = np.array([6.0, 7.0])

    label = 1
    k=3

    X_primes = [
        [1, 4],
        [5, -3],
        [1, -2],
        [0, -2]
    ]
    true_scores = [(math.sqrt((2/6)**2+(6/7)**2)+math.sqrt((0/6)**2+(3/7)**2)+math.sqrt((4/6)**2+(5/7)**2))/3,
                   (math.sqrt((2/6)**2+(1/7)**2)+math.sqrt((4/6)**2+(4/7)**2)+math.sqrt((0/6)**2+(2/7)**2))/3,
                   (math.sqrt((2/6)**2+(0/7)**2)+math.sqrt((0/6)**2+(3/7)**2)+math.sqrt((4/6)**2+(1/7)**2))/3,
                   (math.sqrt((3/6)**2+(0/7)**2)+math.sqrt((1/6)**2+(3/7)**2)+math.sqrt((5/6)**2+(1/7)**2))/3
                   ]

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=heterogeneous_euclidean_overlap_metric,
                            metric_params={'feature_range': feature_range, 'cat_features': cat_features,
                                           'non_cat_features': non_cat_features})
    nbrs.fit(X[y==label])

    for no in range(len(X_primes)):
        X_prime = np.array(X_primes[no]).reshape(1, -1)
        score = implausibility_single(X_prime, nbrs, k)
        assert score == pytest.approx(true_scores[no], rel=1e-4, abs=1e-5), f"X_prime={X_prime}"

def test_implausibility():
    X = np.array([[-1, -1], [2, -1], [3, -2], [1, 1], [2, 1], [3, 2], [5, -1], [1, -3]])

    cat_features = []
    non_cat_features = [0, 1]
    feature_range = np.array([6.0, 5.0])

    X_primes = [
        [1, 4],
        [5, -3],
        [1, -2],
        [0, -2]
    ]
    true_score = (math.sqrt((2/6)**2+(2/5)**2) + math.sqrt((2/6)**2+(1/5)**2)
                  + math.sqrt((0/6)**2+(1/5)**2) + math.sqrt((1/6)**2+(1/5)**2)) / 4

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric=heterogeneous_euclidean_overlap_metric,
                            metric_params={'feature_range': feature_range, 'cat_features': cat_features,
                                           'non_cat_features': non_cat_features})
    nbrs.fit(X)

    score = implausibility(X_primes, nbrs)
    assert score == pytest.approx(true_score, rel=1e-4, abs=1e-5), f"X_primes={X_primes}"
