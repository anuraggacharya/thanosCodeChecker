# test_calculator.py
from calculator import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-2, -3) == -5


def test_add_wrong_expected():
    assert add(2, 2) == 5
