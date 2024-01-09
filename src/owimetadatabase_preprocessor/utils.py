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
            return math.isclose(a, b, rel_tol=tol)
        return a == b

def deepcompare(a, b, tol=1e-9):
    if type(a) != type(b):
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
