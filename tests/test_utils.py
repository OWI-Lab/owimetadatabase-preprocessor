import pytest

import numpy as np

from owimetadatabase_preprocessor.utils import (
    dict_generator, compare_if_simple_close, deepcompare, fix_nan, fix_outline
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
        (np.nan, np.nan, True),
        (np.nan, 1.0, False),
        (None, None, True),
        (None, 1.0, False),
        ({"key_1": 1.0, "key_2": "value_2"}, {"key_1": 1.0, "key_2": "value_2"}, True),
        ({"key_1": 1.0, "key_3": "value_2"}, {"key_1": 1.0, "key_2": "value_2"}, False),
        ({"key_1": 1.0, "key_2": "value_2"}, {"key_1": 1.0, "key_2": "value_3"}, False),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            True
        ),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 3}},
            False
        ),
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
        (np.nan, np.nan),
        (1.0, 1.0),
        (1, 1),
        (None, None),
        ([1, "nan", np.nan, None], [1, np.nan, np.nan, None]),
        ("nan", np.nan),
        ("NaN", np.nan),
        ("NAN", np.nan),
        ("number", "number"),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}
        ),
        (
            {"key_1": np.nan, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": np.nan, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}}
        ),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": "nan", "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": np.nan, "key_32": 2}}
        ),
        ([1, 2, [3, 4]], [1, 2, [3, 4]]),
        ([1, "nan", [3, 4]], [1, np.nan, [3, 4]]),
        ([1, 2, [3, np.nan]], [1, 2, [3, np.nan]]),
    ]
    )
def test_fix_nan(obj, expected):
    result = fix_nan(obj)
    assert deepcompare(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        ([{"key_1": 1}], [{"key_1": 1}]),
        (
            [{"key_1": 1, "key_2": "value_2"}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "key_2": "value_2"}, {"key_1": 1, "key_2": "value_2"}]
        ),
        (
            [{"key_1": 1, "outline": None}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": None}, {"key_1": 1, "key_2": "value_2"}]
        ),
        (
            [{"key_1": 1, "outline": [1, 2, 3]}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}]
        ),
        (
            [{"key_1": 1, "outline": [[1, 2], 3]}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": ([1, 2], 3)}, {"key_1": 1, "key_2": "value_2"}]
        ),
        (
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}]
        ),
        (
            [{"key_1": 1, "outline": [[1, 2], 3]}, {"key_1": [1, 2, 3], "key_2": "value_2"}],
            [{"key_1": 1, "outline": ([1, 2], 3)}, {"key_1": (1, 2, 3), "key_2": "value_2"}]
        ),
        ({"key_1": 1, "key_2": "value_2"}, {"key_1": 1, "key_2": "value_2"}),
        ({"key_1": 1, "key_2": [[1, 2], 3]}, {"key_1": 1, "key_2": ([1, 2], 3)})
    ]
)
def test_fix_outline(data, expected):
    result = fix_outline(data)
    assert deepcompare(result, expected)


def test_fix_outline():
    with pytest.raises(ValueError):
        fix_outline("Will give error!")