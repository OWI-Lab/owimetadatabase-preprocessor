"""Module to connect to the database API to retrieve and operate on results data."""

# mypy: ignore-errors

from typing import Union

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.io import API


class ResultsAPI(API):
    """Class to connect to the results data API with methods to retrieve data."""

    def __init__(
        self,
        api_subdir: str = "/results/routes/",
        **kwargs,
    ) -> None:
        """Create an instance of the ResultsAPI class with the required parameters.

        :param api_subdir: Optional: subdirectory of the API endpoint url for specific type of data.
        :param kwargs: Additional parameters to pass to the API (see the base class).
        :return: None
        """
        super().__init__(**kwargs)
        # self.loc_api = LocationsAPI(**kwargs)
        self.api_root = self.api_root + api_subdir

    def get_analyses(
        self,
        name: Union[str, None] = None,
    ) -> dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all relevant analyses.

        :param name: Optional: Name of the analysis.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the analyses
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}
        if name is not None:
            url_params["name"] = name
        url_data_type = "analysis"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_results(
        self,
        assetlocation: Union[str, None] = None,
        projectsite: Union[str, None] = None,
        analysis: Union[str, None] = None,
        short_description: Union[str, None] = None,
        **kwargs,
    ) -> dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get the results.

        :param assetlocation: Optional: Title of the asset location.
        :param projectsite: Optional: Title of the projectsite.
        :param analysis: Optional: Title of the analysis.
        :param short_description: Optional: Short description of the analysis.
        :return: Dictionary with the following keys:

            - "id": ID of the specified model definition
            - "multiple_modeldef": Boolean indicating whether there are multiple model definitions
                                   for the asset location in general
        """
        url_params = kwargs
        if projectsite is not None:
            url_params["site__title"] = projectsite
        if assetlocation is not None:
            url_params["location__title"] = assetlocation
        if analysis is not None:
            url_params["analysis__title"] = analysis
        if short_description is not None:
            url_params["short_description"] = short_description
        url_data_type = "result"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
