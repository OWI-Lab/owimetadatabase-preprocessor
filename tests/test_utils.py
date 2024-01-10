import pytest

import numpy as np

from owimetadatabase_preprocessor.utils import (
    dict_generator, compare_if_simple_close, deepcompare, fix_nan
)


def test_dict_generator(dict_in, dict_out):
    assert dict_generator(dict_in, **dict_out["method_keys"]) == dict_out["data_out"]


def test_dict_generator_error(dict_in):
    with pytest.raises(ValueError):
        dict_generator(dict_in, method_="wrong_method", keys_=[])


@pytest.mark.parametrize(
    "a, b, expected", 
    [(1.0, 1.0, True),
    (1.0, 1.0000000000000001, True),
    (np.float64(1.0), float(1.0), True),
    ("test", 1.0, False), 
    ]
)
def test_compare_if_simple_close(a, b, expected):
    result = compare_if_simple_close(a, b, expected)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected", 
    [
        (1.0, 1.0, True),
        (1.0, 1.0000000000000001, True),
        (np.float64(1.0), float(1.0), True),
        ("test", 1.0, False),
        ({"key_1": 1.0, "key_2": "value_2"}, {"key_1": 1.0, "key_2": "value_2"}, True),
        ({"key_1": 1.0, "key_3": "value_2"}, {"key_1": 1.0, "key_2": "value_2"}, False),
        ({"key_1": 1.0, "key_2": "value_2"}, {"key_1": 1.0, "key_2": "value_3"}, False),
        ({"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}, {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}, True),
        ({"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}, {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 3}}, False),
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3], [1, 2, 4], False),
        ([1, 2, [3, 4]], [1, 2, [3, 4]], True),
        ([1, 2, [3, 4]], [1, 2, [3, 5]], False),
        ((1, 2, 3), (1, 2, 3), True),
        ((1, 2, 3), (1, 2, 4), False),
        ((1, 2, (3, 4)), (1, 2, (3, 4)), True),
        ((1, 2, (3, 4)), (1, 2, (3, 5)), False),
    ]
)
def test_deepcompare(a, b, expected):
    result = deepcompare(a, b, expected)
    assert result == expected


@pytest.mark.parametrize(
    "obj, expected", 
    [
        (np.nan, None),
        (1.0, 1.0),
        (1, 1),
        ("nan", None),
        ("NaN", None),
        ("NAN", None),
        ("number", "number"),
        ({"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}, {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}),
        ({"key_1": np.nan, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}, {"key_1": None, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}),
        ({"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": "nan", "key_32": 2}}, {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": None, "key_32": 2}}),
        ([1, 2, [3, 4]], [1, 2, [3, 4]]),
        ([1, "nan", [3, 4]], [1, None, [3, 4]]),
        ([1, 2, [3, np.nan]], [1, 2, [3, None]]),
    ]
    )
def test_fix_nan(obj, expected):
    result = fix_nan(obj)
    assert result == expected