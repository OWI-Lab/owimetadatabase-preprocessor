from typing import Dict

import pytest


@pytest.fixture(scope="module")
def api_root() -> str:
    return "https://test.api/test"


@pytest.fixture(scope="module")
def header() -> Dict[str, str]:
    return {"Authorization": "Token 12345"}