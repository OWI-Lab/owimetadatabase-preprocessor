"""Module to connect to the database API to retrieve and operate on locations data."""

import pandas as pd
import plotly.express as px  # type: ignore

from owimetadatabase_preprocessor.io import API


class LocationsAPI(API):
    """
    Class to connect to the location data API with methods to retrieve data.

    A number of methods are provided to query the database via the owimetadatabase API.
    In the majority of cases, the methods return a dataframe based on the URL parameters provided.
    The methods are written such that a number of mandatory URL parameters are required (see documentation of the methods).
    The URL parameters can be expanded with Django-style additional filtering arguments (e.g.
    ``location__title__icontains="BB"``) as optional keyword arguments. Knowledge of the Django models is required for this
    (see ``owimetadatabase`` code).
    """

    def get_projectsites(self, **kwargs):
        """Get all available projects.

        :return:  Dictionary with the following keys:
            - 'data': Pandas dataframe with the location data for each project
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {}
        url_params = {**url_params, **kwargs}
        url_data_type = "/locations/projectsites/"
        output_type = "list"
        resp = self.send_request(url_data_type, url_params)
        df = self.output_to_df(resp)
        df_add = self.postprocess_data(df, output_type)
        return {"data": df, "exists": df_add["existance"]}
