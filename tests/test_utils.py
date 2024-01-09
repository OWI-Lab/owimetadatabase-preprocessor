import pytest

import numpy as np

from owimetadatabase_preprocessor.utils import dict_generator, compare_if_simple_close, deepcompare


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
    [(1.0, 1.0, True),
    (1.0, 1.0000000000000001, True),
    (np.float64(1.0), float(1.0), True),
    ("test", 1.0, False), 
    ]
)
def test_deepcompare(a, b, expected):
    result = deepcompare(a, b, expected)
    assert result == expected