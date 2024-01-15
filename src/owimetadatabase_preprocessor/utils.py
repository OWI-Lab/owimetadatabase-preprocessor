import math

import numpy as np
import pandas as pd


def dict_generator(dict_, keys_=None, method_="exclude"):
    if method_ == "exclude":
        return {k: dict_[k] for k in dict_.keys() if k not in keys_}
    elif method_ == "include":
        return {k: dict_[k] for k in dict_.keys() if k in keys_}
    else:
        raise ValueError("Method not recognized!")


def compare_if_simple_close(a, b, tol=1e-9):
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


def check_df_eq(df1, df2, tol=1e-9):
    num_cols_eq = np.allclose(
        df1.select_dtypes(include=np.number),
        df2.select_dtypes(include=np.number),
        rtol=tol,
        atol=tol,
        equal_nan=True,
    )
    str_cols_eq = df1.select_dtypes(include=object).equals(
        df2.select_dtypes(include=object)
    )
    return num_cols_eq and str_cols_eq


def deepcompare(a, b, tol=1e-5):
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


def fix_nan(obj):
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


def fix_outline(data):
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
