from pathlib import Path
import pytest
import yaml
from briefcase.case import Case
from briefcase.priority_order import PriorityOrder
from briefcase.case_base import CaseBase

from collections import Counter


# Define a fixture to load test cases from the YAML file
@pytest.fixture
def test_cases():
    test_data_path = Path(__file__).parent / 'test_data' / 'test_priority_order.yaml'
    with open(test_data_path, 'r') as file:
        return yaml.safe_load(file)


@pytest.mark.parametrize(
    "test_name, error_type",
    [("error_case_bad_decision", KeyError), ("error_case_bad_reason", ValueError)],
)
def test_errors_case_from_dict(test_cases, test_name, error_type):
    with pytest.raises(error_type):
        Case.from_dict(test_cases[test_name])


def test_base_case_consistency(test_cases):
    cb1 = CaseBase([])  # no cases
    assert cb1.is_cb_consistent()
    case = Case.from_dict(test_cases["one_case"])  # one case
    cb1.add_case(case)
    assert cb1.is_cb_consistent()


# Define the tests using the loaded test cases
@pytest.mark.parametrize(
    "test_case_name",
    [
        "simple_small",
        "simple_big",
        "distractor_small",
        "subset_small",
        "subset_big",
        "multi_defeated_small",
        "multi_defeated_big",
        "mega_case_10",
        "combined_factors"
    ],
)
def test_consistency(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    cases = [Case.from_dict(c) for c in cs[:-1]]  # don't include last case
    cb1 = CaseBase(cases)
    assert cb1.is_cb_consistent()

    inconsistent_case = Case.from_dict(cs[-1])

    assert not cb1.is_consistent_with(inconsistent_case)
    cb1.add_case(inconsistent_case)

    assert not cb1.is_cb_consistent()


# test add order with subsets references the same object to the order
@pytest.mark.parametrize(
    "test_case_name",
    [
        "simple_small",
        "simple_big"
    ],
)
def test_add_order_with_subsets_id(test_cases, test_case_name):
    cs = test_cases[test_case_name]
    case = Case.from_dict(cs[0])  # get first case
    order = PriorityOrder(CaseBase())  # blank priority order
    order.add_order_with_subsets(case.reason, case.defeated())  # add one element
    # check if items added, and that the id is the same id
    assert any(case.defeated() is obj for obj in order.order.keys())
    assert any(case.defeated() is obj for subset in order.defeated_factor_index.values() for obj in subset)


@pytest.mark.parametrize(
    "test_case_name",
    [
        "test_get_incons_pairs_with_case"
    ],
)
def test_get_incons_pairs_with_case(test_cases, test_case_name):
    cs = test_cases[test_case_name]['cases']
    cases = [Case.from_dict(c) for c in cs[:-1]]  # don't include last three case
    cb1 = CaseBase(cases)
    assert cb1.is_cb_consistent()

    inconsistent_case = Case.from_dict(cs[-1])

    answer = [(cases[v].reason, cases[v].defeated()) for v in test_cases[test_case_name]['answer']]
    expected = cb1.order.get_incons_pairs_with_case(inconsistent_case.reason, inconsistent_case.defeated())
    assert Counter(expected) == Counter(answer)

def test_priority_order_str_order():
    # Create a PriorityOrder instance
    priority_order = PriorityOrder(cb=None)

    # Add some order entries for testing
    priority_order.add_order_with_subsets(frozenset({"factor1"}), frozenset({"factor2"}))
    priority_order.add_order_with_subsets(frozenset({"factor3"}), frozenset({"factor4", "factor5"}))

    # Expected string representation

    expected_str = "\nPriority Order:\nReason: frozenset({'factor2'}), Defeated: {frozenset({'factor1'})}\nReason: frozenset({'factor5', 'factor4'}), Defeated: {frozenset({'factor3'})}"
    expected_str_2 = "\nPriority Order:\nReason: frozenset({'factor2'}), Defeated: {frozenset({'factor1'})}\nReason: frozenset({'factor4', 'factor5'}), Defeated: {frozenset({'factor3'})}"
    actual = priority_order.str_order()
    # Check the string representation returned by str_order
    assert actual == expected_str or actual == expected_str_2