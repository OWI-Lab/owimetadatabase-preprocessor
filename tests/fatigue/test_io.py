from typing import Any, Dict

import pytest

from owimetadatabase_preprocessor.fatigue.io import FatigueAPI


@pytest.fixture(scope="function")
def fatigue_init(header):
    return {
        "api_root": "https://owimetadatabase.azurewebsites.net/api/v1/fatigue/userroutes/",
        "header": header,
        "auth": None,
        "uname": None,
        "password": None,
    }


def test_init(fatigue_init: Dict[str, Any], header: Dict[str, str]) -> None:
    api_fatigue = FatigueAPI(header=header)
    assert fatigue_init["api_root"] == api_fatigue.api_root
    assert fatigue_init["header"] == api_fatigue.header
    assert fatigue_init["auth"] == api_fatigue.auth
    assert fatigue_init["uname"] == api_fatigue.uname
    assert fatigue_init["password"] == api_fatigue.password
