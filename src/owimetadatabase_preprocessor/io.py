import requests


class API(object):
    """Base API class handling user access information to the Database API."""

    def __init__(
        self,
        api_root: str,
        header: dict[str, str] | None = None,
        uname: str | None = None,
        password: str | None = None
    ) -> None:
        self.api_root = api_root
        self.header = header
        self.uname = uname
        self.password = password
        if self.uname is not None and self.password is not None:
            self.auth = requests.auth.HTTPBasicAuth(self.uname, self.password)
        else:
            self.auth = None
