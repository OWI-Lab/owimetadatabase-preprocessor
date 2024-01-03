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

    def test_init(self, owt, owt_init) -> None:
        assert_attributes(owt, owt_init)

    def test_set_subassemblies_and_members(self, owt_api, data_mat_df, sab) -> None:
        owt = OWT(owt_api, data_mat_df, sab)
