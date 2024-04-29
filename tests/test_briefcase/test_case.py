from pathlib import Path
import pytest
import yaml

from briefcase.enums import decision_enum
from briefcase.factor import Factor
from briefcase.case import Case

# Define a fixture to load test cases from the YAML file
@pytest.fixture
def test_cases():
    test_data_path = Path(__file__).parent / 'test_data' / 'test_relevant_diffs.yaml'
    with open(test_data_path, 'r') as file:
        return yaml.safe_load(file)


@pytest.mark.parametrize(
    "test_case_name",
    [
        "small_same_outcome",
        "small_different_outcome",
        "big_same_outcome",
        "big_different_outcome"
    ],
)
def test_relevant_differences(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    case1 = Case.from_dict(cs["case1"])
    factors = {Factor(name, decision_enum[polarity]) for name, polarity in cs["diff"].items()}
    assert frozenset(case1.relevant_diff_from(Case.from_dict(cs["case2"]))) == frozenset(factors)


def test_case_repr():
    pi_factors = frozenset({Factor("test_factor1", decision_enum.pi)})
    delta_factors = frozenset({Factor("test_factor2", decision_enum.delta)})
    decision = decision_enum.pi
    reason = pi_factors
    case = Case(pi_factors, delta_factors, decision, reason)

    # Expected representation string
    expected_repr = (
        "Case(Pi Factors: [\"Factor('test_factor1', 'Decision.pi')\"], "
        "Delta Factors: [\"Factor('test_factor2', 'Decision.delta')\"], "
        "Decision.pi, Reasons: [\"Factor('test_factor1', 'Decision.pi')\"])"
    )


    # Check the representation returned by __repr__
    assert repr(case) == expected_repr

def test_case_eq():
    pi_factors1 = frozenset({Factor("test_factor1", decision_enum.pi)})
    delta_factors1 = frozenset({Factor("test_factor2", decision_enum.delta)})
    decision1 = decision_enum.pi
    reason1 = pi_factors1
    case1 = Case(pi_factors1, delta_factors1, decision1, reason1)

    pi_factors2 = frozenset({Factor("test_factor1", decision_enum.pi)})
    delta_factors2 = frozenset({Factor("test_factor2", decision_enum.delta)})
    decision2 = decision_enum.pi
    reason2 = pi_factors2
    case2 = Case(pi_factors2, delta_factors2, decision2, reason2)

    assert case1 == case2

    delta_factors2 = frozenset({Factor("test_factor3", decision_enum.delta)})
    case2_modified = Case(pi_factors2, delta_factors2, decision2, reason2)

    assert case1 != case2_modified
