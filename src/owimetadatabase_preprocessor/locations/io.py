"""Module to connect to the database API to retrieve and operate on locations data."""

import numpy as np 
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

    def get_projectsites(self, **kwargs) -> dict[str, pd.DataFrame | bool]:
        """Get all available projects.

        :return:  Dictionary with the following keys:
            - 'data': Pandas dataframe with the location data for each project
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {}
        url_params = {**url_params, **kwargs}
        url_data_type = "/locations/projectsites/"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_projectsite_detail(self, projectsite, **kwargs) -> dict[str, pd.DataFrame | bool | np.int64]:
        """Get details for a specific projectsite.

        :param projectsite: Title of the projectsite
        :return:  Dictionary with the following keys:
            - 'id': id of the selected project site
            - 'data': Pandas dataframe with the location data for each projectsite
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {"projectsite": projectsite}
        url_params = {**url_params, **kwargs}
        url_data_type = "/locations/projectsites/"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}
