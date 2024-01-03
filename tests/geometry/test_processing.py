import json
from typing import Any, Callable, Dict, List, Union
from unittest import mock

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.geometry.structures import Material, Position, BuildingBlock, SubAssembly
from .test_utils import assert_attributes


class  TestOWT:

    def test_init(self, api_root, header, data_mat, sa, owt_init) -> None:
        api_test = GeometryAPI(api_root, header)
        owt = OWT(api_test, data_mat, sa)
        assert_attributes(owt, owt_init)