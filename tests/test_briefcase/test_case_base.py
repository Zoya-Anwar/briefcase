from pathlib import Path
import pytest
import yaml

from briefcase.case_base import CaseBase
from briefcase.enums import decision_enum
from briefcase.factor import Factor
from briefcase.case import Case


# Define a fixture to load test cases from the YAML file
@pytest.fixture
def test_cases():
    test_data_path = Path(__file__).parent / 'test_data' / 'test_case_base.yaml'
    with open(test_data_path, 'r') as file:
        return yaml.safe_load(file)


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_add_cases",
    ],
)
def test_add_cases(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    new_cases = cs
    cases = [Case.from_dict(c) for c in new_cases]
    cb1 = CaseBase(cases)
    cb2 = (CaseBase())
    cb2.add_cases(cases)
    assert cb1.cases == cb2.cases


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_add_cases",
        "simple_small",
    ],
)
def test_add_cases(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    new_cases = cs
    cases = [Case.from_dict(c) for c in new_cases]
    cb1 = CaseBase(cases)
    assert cb1.metrics() == (len(cases), 2)
@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_add_cases",
        "simple_small",
    ],
)
def test_add_wrong_incons_enum(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    new_cases = cs
    cases = [Case.from_dict(c) for c in new_cases]
    cb1 = CaseBase()
    with pytest.raises(KeyError):
        cb1.add_cases(cases, "random-wrong")
