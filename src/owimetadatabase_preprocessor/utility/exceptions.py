""" 
Custom exceptions for the API client. These exceptions encapsulate 
common errors that can occur during API calls and data post‐processing.
"""


class APIException(Exception): 
    """ Base exception for all errors raised by SoilAPI. """ 
    pass

class APIConnectionError(APIException): 
    """ 
    Exception raised when the API cannot be reached or returns a failure. 
    """ 
    def init(self, message="Failed to connect to API.", response=None): 
        self.response = response 
        super().init(message)

class DataNotFoundError(APIException): 
    """ 
    Exception raised when no data is found for the given query parameters. 
    """ 
    def init(self, message="No data found for the given search criteria."): 
        super().init(message)

class DataProcessingError(APIException): 
    """ Exception raised when there is a problem while processing the data 
    (e.g. merging, converting). 
    """ 
    def init(self, message="Error during data processing."): 
        super().init(message)

class InvalidParameterError(APIException): 
    """ 
    Exception raised when query parameters are invalid or missing. 
    """ 
    def init(self, message="Invalid or missing parameters for the request."): 
        super().init(message)