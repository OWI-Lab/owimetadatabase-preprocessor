"""Utility functions for the owimetadatabase_preprocessor package."""

import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def custom_formatwarning(message, category, filename, lineno, line=None):
    """Custom format for warnings."""
    return f"{category.__name__}: {message}\n"


def dict_generator(
    dict_: Dict[str, Any], keys_: List[str] = [], method_: str = "exclude"
) -> Dict[str, Any]:
    """Generate a dictionary with the specified keys.

    :param dict_: Dictionary to be filtered.
    :param keys_: List of keys to be included or excluded.
    :param method_: Method to be used for filtering. Options are "exclude" and "include".
    :return: Filtered dictionary.
    """
    if method_ == "exclude":
        return {k: dict_[k] for k in dict_.keys() if k not in keys_}
    elif method_ == "include":
        return {k: dict_[k] for k in dict_.keys() if k in keys_}
    else:
        raise ValueError("Method not recognized!")


def compare_if_simple_close(
    a: Any, b: Any, tol: float = 1e-9
) -> Tuple[bool, Union[None, str]]:
    """Compare two values and return a boolean and a message.

    :param a: First value to be compared.
    :param b: Second value to be compared.
    :param tol: Tolerance for the comparison.
    :return: Tuple with a result of comparison and a message if different.
    """
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True, None
        assertion = math.isclose(a, b, rel_tol=tol)
        if assertion:
            messsage = None
        else:
            messsage = f"Values of {a} and {b} are different."
        return assertion, messsage
    assertion = a == b
    if assertion:
        messsage = None
    else:
        messsage = f"Values of {a} and {b} are different."
    return assertion, messsage


def check_df_eq(df1: pd.DataFrame, df2: pd.DataFrame, tol: float = 1e-9) -> bool:
    """Check if two dataframes are equal.

    :param df1: First dataframe to be compared.
    :param df2: Second dataframe to be compared.
    :param tol: Tolerance for the comparison.
    :return: Boolean indicating if the dataframes are equal.
    """
    if df1.empty and df2.empty:
        return True
    elif (df1.empty and not df2.empty) or (not df1.empty and df2.empty):
        return False
    if df1.shape != df2.shape:
        return False
    num_cols_eq = np.allclose(
        df1.select_dtypes(include=np.number),  # type: ignore
        df2.select_dtypes(include=np.number),  # type: ignore
        rtol=tol,
        atol=tol,
        equal_nan=True,
    )
    str_cols_eq = df1.select_dtypes(include=object).equals(  # type: ignore
        df2.select_dtypes(include=object)  # type: ignore
    )
    return num_cols_eq and str_cols_eq


def deepcompare(a: Any, b: Any, tol: float = 1e-5) -> Tuple[bool, Union[None, str]]:
    """Compare two complicated (potentailly nested) objects recursively and return a result and a message.

    :param a: First object to be compared.
    :param b: Second object to be compared.
    :param tol: Tolerance for the comparison.
    :return: Tuple with a result of comparison and a message if different.
    """
    if type(a) != type(b):  # noqa: E721
        if hasattr(a, "__dict__") and isinstance(b, dict):
            return deepcompare(a.__dict__, b, tol)
        elif hasattr(b, "__dict__") and isinstance(a, dict):
            return deepcompare(a, b.__dict__, tol)
        elif isinstance(a, (float, np.floating)) and isinstance(
            b, (float, np.floating)
        ):
            return deepcompare(np.float64(a), np.float64(b), tol)
        return (
            False,
            f"Types of {a} and {b} are different: {type(a).__name__} and {type(b).__name__}.",
        )
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return False, f"Dictionary keys {a.keys()} and {b.keys()} are different."
        compare = [deepcompare(a[key], b[key], tol)[0] for key in a]
        assertion = all(compare)
        if assertion:
            message = None
        else:
            keys = [key for key, val in zip(a.keys(), compare) if val is False]
            message = (
                f"Dictionary values are different for {a} and {b}, for keys: {keys}."
            )
        return assertion, message
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return (
                False,
                f"Lists/tuples {a} and {b} are of different length: {len(a)} and {len(b)}.",
            )
        compare = [deepcompare(i, j, tol)[0] for i, j in zip(a, b)]
        assertion = all(compare)
        if assertion:
            message = None
        else:
            inds = [
                ind for ind, val in zip(range(len(compare)), compare) if val is False
            ]
            message = (
                f"Lists/tuples are different for {a} and {b}, for indices: {inds}."
            )
        return assertion, message
    elif hasattr(a, "__dict__") and not isinstance(a, pd.DataFrame):
        return deepcompare(a.__dict__, b.__dict__, tol)
    elif isinstance(a, pd.DataFrame):
        assertion = check_df_eq(a, b, tol)
        if assertion:
            message = None
        else:
            message = f"Dataframes {a} and {b} are different for {a.compare(b)}."
        return assertion, message
    else:
        return compare_if_simple_close(a, b, tol)


def fix_nan(obj: Any) -> Any:
    """Replace "nan" strings with None.

    :param obj: Object to be fixed.
    :return: Fixed object.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = fix_nan(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = fix_nan(obj[i])
    elif isinstance(obj, str) and obj.lower() == "nan":
        # obj = None
        obj = np.nan
    return obj


def fix_outline(data: Any) -> Any:
    """Fix the outline attribute in the data.

    :param data: Data to be fixed.
    :return: Fixed data.
    """
    if isinstance(data, list):
        for i in range(len(data)):
            if "outline" in data[i].keys() and data[i]["outline"] is not None:
                data[i]["outline"] = tuple(data[i]["outline"])
    elif isinstance(data, dict):
        if "outline" in data.keys() and data["outline"] is not None:
            data["outline"] = tuple(data["outline"])
    else:
        raise ValueError("Not supported data type.")
    return data


def hex_to_dec(value):
    """Return [red, green, blue, alpha] for the color given as #rrggbbaa."""

    def _hex_to_dec(value):
        value = value.lstrip("#") if value.startswith("#") else value
        if len(value) != 6:
            if len(value) != 8:
                raise ValueError("len(value) != 6 or 8 (excluding #)")
        col = value[0:6]
        alpha = value[6:] / 100 if len(value) == 8 else 1
        lv = len(col)
        return list(
            int(col[i : i + lv // 3], 16) / 255  # noqa: E203
            for i in range(0, lv, lv // 3)
        ) + [alpha]

    if isinstance(value, str):
        value = [value]
    elif isinstance(value, list):
        return [_hex_to_dec(_) for _ in value]
