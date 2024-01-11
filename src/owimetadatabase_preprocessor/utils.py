import math
import numpy as np


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
                return True
            return math.isclose(a, b, rel_tol=tol)
        return a == b


def deepcompare(a, b, tol=1e-5):
    if type(a) != type(b):
        if (hasattr(a, '__dict__') and type(b) == dict):
            return deepcompare(a.__dict__, b, tol)
        elif (hasattr(b, '__dict__') and type(a) == dict):
            return deepcompare(a, b.__dict__, tol)
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return deepcompare(np.float64(a), np.float64(b), tol)
        return False
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deepcompare(a[key], b[key], tol) for key in a)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deepcompare(i, j, tol) for i, j in zip(a, b))
    elif hasattr(a, '__dict__'):
        return deepcompare(a.__dict__, b.__dict__, tol)
    else:
        return compare_if_simple_close(a, b, tol)
    

def fix_nan(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = fix_nan(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = fix_nan(obj[i])
    elif (
        # (isinstance(obj, (float, np.floating)) and np.isnan(obj))
        # or 
        (isinstance(obj, str) and obj.lower() == "nan")
    ):
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