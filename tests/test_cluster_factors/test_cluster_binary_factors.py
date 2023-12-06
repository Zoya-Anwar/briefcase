import os
import pytest
import yaml
import pandas as pd
from cluster_factors.cluster_binary_factors import cluster_factors_corr, cluster_factors_voting

# Define a fixture to load test cases from the YAML file
@pytest.fixture
def test_cases():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'unclustered_casebases.yaml')
    with open(test_data_path, 'r') as file:
        return yaml.safe_load(file)


# Define the tests using the loaded test cases
@pytest.mark.parametrize(
    "test_case_name",
    [
        "CB1",
        "CB2"
    ],
)
def test_cluster_factors_corr(test_cases, test_case_name):
    data = test_cases[test_case_name]
    df = pd.DataFrame(data["cb"])
    answer = data["answer_corr"]
    assert cluster_factors_corr(df, "y") == answer

# Define the tests using the loaded test cases
@pytest.mark.parametrize(
    "test_case_name",
    [
        "CB1",
        "CB2"
    ],
)
def test_cluster_factors_voting(test_cases, test_case_name):
    data = test_cases[test_case_name]
    df = pd.DataFrame(data["cb"])
    answer = data["answer_voting"]
    assert cluster_factors_voting(df, "y") == answer
