from briefcase.factor import Factor
import pytest

def test_factor_eq():
    factor1 = Factor("test_factor", "pi")
    factor2 = 1
    assert factor1 != factor2

def test_factor_repr():
    factor = Factor("test_factor", "pi")
    expected_repr = "Factor('test_factor', 'pi')"
    assert repr(factor) == expected_repr
