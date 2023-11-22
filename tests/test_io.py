import pytest
import requests

from owimetadatabase_preprocessor.io import API


@pytest.fixture
def mock_requests_get(mocker):
    mock = mocker.patch("requests.get")
    mock.return_value = requests.Response()
    return mock


class TestAPIAuth:
    """Tests of authentication setup."""

    def test_API_header(self) -> None:
        """Test parent API class with header that it initializes everything correctly."""
        api_root = "https://test.api/test"
        header = {"Authorization": "Token 12345"}
        api_test = API(api_root, header=header)
        assert api_test.api_root == api_root
        assert api_test.header == header
        assert api_test.uname is None
        assert api_test.password is None
        assert api_test.auth is None

    def test_API_user(self) -> None:
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


def test_send_request(mock_requests_get) -> None:
    api_root = "https://test.api/test"
    header = {"Authorization": "Token 12345"}
    url_data_type = "/test/"
    url_params = {"test": "test"}
    api_test = API(api_root, header=header)
    response = api_test.send_request(url_data_type, url_params)
    assert isinstance(response, requests.models.Response)
