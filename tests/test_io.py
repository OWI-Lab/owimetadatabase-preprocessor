import requests

from owimetadatabase_preprocessor.io import API


def test_API_header() -> None:
    """Test parent API class with header that it initializes everything correctly."""
    api_root = "https://test.api/test"
    header = {"Authorization": "Token 12345"}
    api_test = API(api_root, header=header)
    assert api_test.api_root == api_root
    assert api_test.header == header
    assert api_test.uname is None
    assert api_test.password is None
    assert api_test.auth is None


def test_API_user() -> None:
    """Test parent API class with user credentials that it initializes everything correctly."""
    api_root = "https://test.api/test"
    name = "test"
    pswd = "12345"
    api_test = API(api_root, uname=name, password=pswd)
    assert api_test.api_root == api_root
    assert api_test.header is None
    assert api_test.uname == name
    assert api_test.password == pswd
    assert api_test.auth == requests.auth.HTTPBasicAuth(name, pswd)
