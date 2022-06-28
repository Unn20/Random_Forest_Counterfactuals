import pytest

from rf_counterfactuals.costs_functions import *

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

def test_unmatched_components_rate():
    eps = 1e-4
    # Manual calculation
    ground_truth = np.array([0.0, 1/1, 2/3, 0/3, 3/3, 1/3, 1/5, 0/3, 0/3, 2/3, 0/3])
    for no in range(len(ground_truth)):
        left, right = np.array(left_vector_case_0[no]), np.array(right_vector_case_0[no])
        result = unmatched_components_rate(left, right, eps=eps)
        assert pytest.approx(ground_truth[no], rel=1e-4) == result, f"left={left}; right={right}"

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
