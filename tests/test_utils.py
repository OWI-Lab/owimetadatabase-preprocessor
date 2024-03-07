import numpy as np
import pandas as pd
import pytest

from owimetadatabase_preprocessor.utils import (
    compare_if_simple_close,
    deepcompare,
    dict_generator,
    fix_nan,
    fix_outline,
)


def test_dict_generator(dict_gen_dict_in, dict_gen_dict_out):
    assert (
        dict_generator(dict_gen_dict_in, **dict_gen_dict_out["method_keys"])
        == dict_gen_dict_out["data_out"]
    )


def test_dict_generator_error(dict_gen_dict_in):
    with pytest.raises(ValueError):
        dict_generator(dict_gen_dict_in, method_="wrong_method", keys_=[])


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 1.0, (True, None)),
        (1.0, 1.0000000000000001, (True, None)),
        (np.float64(1.0), float(1.0), (True, None)),
        ("value_1", "value_2", (False, "Values of value_1 and value_2 are different.")),
        (1.0, 1.001, (False, "Values of 1.0 and 1.001 are different.")),
        ("value_1", "value_1", (True, None)),
    ],
)
def test_compare_if_simple_close(a, b, expected):
    result = compare_if_simple_close(a, b)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1.0, 1.0, (True, None)),
        (1.0, 1.0000000000000001, (True, None)),
        (np.float64(1.0), float(1.0), (True, None)),
        ("test", 1.0, (False, "Types of test and 1.0 are different: str and float.")),
        (np.nan, np.nan, (True, None)),
        (np.nan, 1.0, (False, "Values of nan and 1.0 are different.")),
        (
            np.nan,
            "test",
            (False, "Types of nan and test are different: float and str."),
        ),
        (None, None, (True, None)),
        (
            None,
            1.0,
            (False, "Types of None and 1.0 are different: NoneType and float."),
        ),
        (
            {"key_1": 1.0, "key_2": "value_2"},
            {"key_1": 1.0, "key_2": "value_2"},
            (True, None),
        ),
        (
            {"key_1": 1.0, "key_3": "value_2"},
            {"key_1": 1.0, "key_2": "value_2"},
            (
                False,
                "Dictionary keys dict_keys(['key_1', 'key_3']) and dict_keys(['key_1', 'key_2']) are different.",
            ),
        ),
        (
            {"key_1": 1.0, "key_2": "value_2"},
            {"key_1": 1.0, "key_2": "value_3"},
            (
                False,
                (
                    "Dictionary values are different for {'key_1': 1.0, 'key_2': 'value_2'} "
                    "and {'key_1': 1.0, 'key_2': 'value_3'},"
                    " for keys: ['key_2']."
                ),
            ),
        ),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            (True, None),
        ),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 3}},
            (
                False,
                (
                    "Dictionary values are different for "
                    "{'key_1': 1.0, 'key_2': 'value_2', 'key_3': {'key_31': 1, 'key_32': 2}} "
                    "and {'key_1': 1.0, 'key_2': 'value_2', 'key_3': {'key_31': 1, 'key_32': 3}}, "
                    "for keys: ['key_3']."
                ),
            ),
        ),
        ([1, 2, 3], [1, 2, 3], (True, None)),
        (
            [1, 2, 3],
            [1, 2, 3, 4],
            (
                False,
                "Lists/tuples [1, 2, 3] and [1, 2, 3, 4] are of different length: 3 and 4.",
            ),
        ),
        (
            [1, 2, 3],
            [1, 2, 4],
            (
                False,
                "Lists/tuples are different for [1, 2, 3] and [1, 2, 4], for indices: [2].",
            ),
        ),
        ([1, 2, [3, 4]], [1, 2, [3, 4]], (True, None)),
        (
            [1, 2, [3, 4]],
            [1, 2, [3, 5]],
            (
                False,
                "Lists/tuples are different for [1, 2, [3, 4]] and [1, 2, [3, 5]], for indices: [2].",
            ),
        ),
        ((1, 2, 3), (1, 2, 3), (True, None)),
        (
            (1, 2, 3),
            (1, 2, 3, 4),
            (
                False,
                "Lists/tuples (1, 2, 3) and (1, 2, 3, 4) are of different length: 3 and 4.",
            ),
        ),
        (
            (1, 2, 3),
            (1, 2, 4),
            (
                False,
                "Lists/tuples are different for (1, 2, 3) and (1, 2, 4), for indices: [2].",
            ),
        ),
        (
            (1, 2, 3),
            (1, 4, 5),
            (
                False,
                "Lists/tuples are different for (1, 2, 3) and (1, 4, 5), for indices: [1, 2].",
            ),
        ),
        ((1, 2, (3, 4)), (1, 2, (3, 4)), (True, None)),
        (
            (1, 2, (3, 4)),
            (1, 2, (3, 5)),
            (
                False,
                "Lists/tuples are different for (1, 2, (3, 4)) and (1, 2, (3, 5)), for indices: [2].",
            ),
        ),
        (
            pd.DataFrame({"col_1": [1.0, 2.0], "col_2": ["val_12", "val_22"]}),
            pd.DataFrame({"col_1": [1.0, 2.0], "col_2": ["val_12", "val_22"]}),
            (True, None),
        ),
        (
            pd.DataFrame({"col_1": [1.0, 2.0], "col_2": ["val_12", "val_22"]}),
            pd.DataFrame({"col_1": [1.0, 2.1], "col_2": ["val_12", "val_22"]}),
            (
                False,
                "Dataframes    col_1   col_2\n0    1.0  val_12\n1    2.0  val_22 "
                "and    col_1   col_2\n0    1.0  val_12\n1    2.1  val_22 are different for"
                "   col_1      \n   self other\n1   2.0   2.1.",
            ),
        ),
        (
            pd.DataFrame({"col_1": [1.0, 2.0], "col_2": ["val_12", "val_22"]}),
            pd.DataFrame({"col_1": [1.0, 2.0000001], "col_2": ["val_12", "val_22"]}),
            (True, None),
        ),
    ],
)
def test_deepcompare(a, b, expected):
    result = deepcompare(a, b)
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
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
        ),
        (
            {"key_1": np.nan, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
            {"key_1": np.nan, "key_2": "value_2", "key_3": {"key_31": 1, "key_32": 2}},
        ),
        (
            {"key_1": 1.0, "key_2": "value_2", "key_3": {"key_31": "nan", "key_32": 2}},
            {
                "key_1": 1.0,
                "key_2": "value_2",
                "key_3": {"key_31": np.nan, "key_32": 2},
            },
        ),
        ([1, 2, [3, 4]], [1, 2, [3, 4]]),
        ([1, "nan", [3, 4]], [1, np.nan, [3, 4]]),
        ([1, 2, [3, np.nan]], [1, 2, [3, np.nan]]),
    ],
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
            [{"key_1": 1, "key_2": "value_2"}, {"key_1": 1, "key_2": "value_2"}],
        ),
        (
            [{"key_1": 1, "outline": None}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": None}, {"key_1": 1, "key_2": "value_2"}],
        ),
        (
            [{"key_1": 1, "outline": [1, 2, 3]}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}],
        ),
        (
            [{"key_1": 1, "outline": [[1, 2], 3]}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": ([1, 2], 3)}, {"key_1": 1, "key_2": "value_2"}],
        ),
        (
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}],
            [{"key_1": 1, "outline": (1, 2, 3)}, {"key_1": 1, "key_2": "value_2"}],
        ),
        (
            [
                {"key_1": 1, "outline": [[1, 2], 3]},
                {"key_1": [1, 2, 3], "key_2": "value_2"},
            ],
            [
                {"key_1": 1, "outline": ([1, 2], 3)},
                {"key_1": (1, 2, 3), "key_2": "value_2"},
            ],
        ),
        ({"key_1": 1, "key_2": "value_2"}, {"key_1": 1, "key_2": "value_2"}),
        ({"key_1": 1, "key_2": [[1, 2], 3]}, {"key_1": 1, "key_2": ([1, 2], 3)}),
    ],
)
def test_fix_outline(data, expected):
    result = fix_outline(data)
    assert deepcompare(result, expected)


def test_fix_outline_exception():
    with pytest.raises(ValueError):
        fix_outline("Will give error!")
