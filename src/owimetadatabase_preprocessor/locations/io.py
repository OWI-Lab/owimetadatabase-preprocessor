"""Module to connect to the database API to retrieve and operate on locations data."""

from typing import Dict, Union

import numpy as np
import pandas as pd
import plotly as plt  # type: ignore
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

    def __init__(
        self,
        api_subdir: str = "/locations/",
        **kwargs,
    ) -> None:
        """Create an instance of the LocationsAPI class with the required parameters.

        :param api_subdir: Optional: subdirectory of the API endpooint url for specific type of data.
        :param kwargs: Additional parameters to pass to the API (see the base class).
        :return: None
        """
        super().__init__(**kwargs)
        self.api_root = self.api_root + api_subdir

    def get_projectsites(
        self, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all available projects.

        :param: kwargs: Additional parameters to pass to the API.
        :return:  Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}  # type: Dict[str, str]
        url_params = {**url_params, **kwargs}
        url_data_type = "projectsites"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_projectsite_detail(
        self, projectsite: str, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get details for a specific projectsite.

        :param projectsite: Title of the projectsite.
        :param kwargs: Additional parameters to pass to the API.
        :return:  Dictionary with the following keys:

            - "id": id of the selected project site.
            - "data": Pandas dataframe with the location data for each projectsite.
            - "exists": Boolean indicating whether matching records are found.
        """
        url_params = {"projectsite": projectsite}
        url_params = {**url_params, **kwargs}
        url_data_type = "projectsites"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}

    def get_assetlocations(
        self, projectsite: Union[str, None] = None, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all available asset locations for all projectsites or a specific projectsite.

        :param projectsite: String with the projectsite title (e.g. "Nobelwind").
        :param assetlocation: String with the asset location title (e.g. "NW2A04").
        :param kwargs: Additional parameters to pass to the API.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each location in the projectsite.
            - "exists": Boolean indicating whether matching records are found.
        """
        url_params = {}  # type: Dict[str, str]
        url_params = {**url_params, **kwargs}
        if projectsite:
            url_params["projectsite__title"] = projectsite
        url_data_type = "assetlocations"
        if "assetlocations" in url_params.keys() and isinstance(
            url_params["assetlocations"], list
        ):
            df = []
            df_add = {"existance": []}
            for assetlocation in url_params["assetlocations"]:
                output_type = "single"
                url_params["assetlocation"] = assetlocation
                df_temp, df_add_temp = self.process_data(
                    url_data_type, url_params, output_type
                )
                df.append(df_temp)
                df_add["existance"].append(df_add_temp["existance"])
            df = pd.concat(df)
        else:
            output_type = "list"
            df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_assetlocation_detail(
        self, assetlocation: str, projectsite: Union[None, str] = None, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get a selected turbine.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind").
        :param assetlocation: Title of the asset location (e.g. "BBK05").
        :param kwargs: Additional parameters to pass to the API.
        :return: Dictionary with the following keys:

            - "id": id of the selected projectsite site.
            - "data": Pandas dataframe with the location data for the individual location.
            - "exists": Boolean indicating whether a matching location is found.
        """
        if projectsite is None:
            url_params = {"assetlocation": assetlocation}
        else:
            url_params = {"projectsite": projectsite, "assetlocation": assetlocation}
        url_params = {**url_params, **kwargs}
        url_data_type = "assetlocations"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}

    def plot_assetlocations(
        self, return_fig: bool = False, **kwargs
    ) -> Union[plt.graph_objects.Figure, None]:
        """Retrieve asset locations and generates a Plotly plot to show them.

        :param return_fig: Boolean indicating whether to return the figure.
        :param kwargs: Keyword arguments for the search (see ``get_assetlocations``).
        :return: Plotly figure object with selected asset locations plotted on OpenStreetMap tiles (if requested) or nothing.
        """
        assetlocations_data = self.get_assetlocations(**kwargs)
        if assetlocations_data["exists"]:
            assetlocations = assetlocations_data["data"]
        else:
            raise ValueError(
                f"No asset locations found for the given parameters: {kwargs}. \
                Please check for typos or if it is expected to exists."
            )
        fig = px.scatter_mapbox(
            assetlocations,
            lat="northing",
            lon="easting",
            hover_name="title",
            hover_data=["projectsite_name", "description"],
            zoom=9.6,
            height=500,
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        if return_fig:
            return fig
        else:
            fig.show()
