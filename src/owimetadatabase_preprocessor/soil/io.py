import json
import warnings
from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from groundhog.general.soilprofile import profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from pyproj import Transformer

from owimetadatabase_preprocessor.io import API


class SoilAPI(API):
    """
    Class to connect to the soil data API with methods to retrieve data.

    A number of methods are provided to query the database via the owimetadatabase API.
    In the majority of cases, the methods return a dataframe based on the URL parameters provided.
    The methods are written such that a number of mandatory URL parameters are required (see documentation of the methods).
    The URL parameters can be expanded with Django-style additional filtering arguments
    (e.g. ``location__title__icontains="BB"``) as optional keyword arguments.
    Knowledge of the Django models is required for this (see ``owimetadatabase`` code).
    """

    def __init__(
        self,
        api_root: str = "https://owimetadatabase.owilab.be/api/v1",
        api_subdir: str = "/soildata/",
        token: Union[str, None] = None,
        uname: Union[str, None] = None,
        password: Union[str, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(api_root, token, uname, password, **kwargs)
        self.api_root = self.api_root + api_subdir

    def get_proximity_entities_2d(
        self, api_url: str, latitude: float, longitude: float, radius: float, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Find the entities in a certain radius around a point in 2D (cylindrical search area).

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km
        :param kwargs: Optional keyword arguments for the search
        :return:  Dictionary with the following keys:

            - "data": Pandas dataframe with the data according to the specified search criteria
            - "exists": Boolean indicating whether matching records are found
        """
        geosearch_params = dict(latitude=latitude, longitude=longitude, offset=radius)
        url_params = {**geosearch_params, **kwargs}
        url_data_type = api_url
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def _search_any_entity(
        self,
        api_url: str,
        radius_init: int,
        url_params: Dict[str, str],
        radius_max: int = 500,
    ) -> pd.DataFrame:
        """Search for any entity in a certain radius around a point in 2D (cylindrical search area).

        :param api_url: End-point for the API
        :param radius_init: Initial search radius around the central point in km
        :param url_params: Dictionary with the URL parameters for the endpoint
        :param radius_max: Maximum search radius around the central point in km
        :return: Pandas dataframe with the data according to the specified search criteria
        """
        radius = radius_init
        while True:
            url_params["offset"] = radius
            url_data_type = api_url
            output_type = "list"
            df, df_add = self.process_data(url_data_type, url_params, output_type)
            if df_add["existance"]:
                break
            warnings.warn("Expanding search radius to")
            radius *= 2
            warnings.warn(f"Expanding search radius to {radius: .1f}km")
            if radius > radius_max:
                raise ValueError(
                    "No locations found within 500km radius. Check your input information."
                )
        return df

    def _transform_coord(
        self, df: pd.DataFrame, longitude: float, latitude: float, target_srid: str
    ) -> Tuple[pd.DataFrame, float, float]:
        """Transform the coordinates from decimal degrees to a specified target SRID.

        :param df: Pandas dataframe with the data according to the specified search criteria
        :param longitude: Longitude of the central point in decimal
        :param latitude: Latitude of the central point in decimal
        :param target_srid: SRID for the offset calculation in meters
        :return: Tuple with the following elements:

            - Pandas dataframe with the data according to the specified search criteria
            - Easting of the central point in meters
            - Northing of the central point in meters
        """
        transformer = Transformer.from_crs("epsg:4326", "epsg:" + target_srid)
        df["easting [m]"], df["northing [m]"] = transformer.transform(
            df["easting"], df["northing"]
        )
        point_east, point_north = transformer.transform(longitude, latitude)
        return df, point_east, point_north

    def _gather_data_entity(
        df: pd.DataFrame,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Gather the data for the closest entity to a certain point in 2D.

        :param df: Pandas dataframe with the data according to the specified search criteria
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        if df.__len__() == 1:
            loc_id = df["id"].iloc[0]
        else:
            df.sort_values("offset [m]", inplace=True)
            loc_id = df[df["offset [m]"] == df["offset [m]"].min()]["id"].iloc[0]
        return {
            "data": df,
            "id": loc_id,
            "title": df["title"].iloc[0],
            "offset [m]": df[df["offset [m]"] == df["offset [m]"].min()][
                "offset [m]"
            ].iloc[0],
        }

    def get_closest_entity_2d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        radius_init: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the entity closest to a certain point in 2D with optional query arguments (cylindrical search area).

        :param api_url: End-point for the API
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
        :return:  Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        geosearch_params = dict(latitude=latitude, longitude=longitude)
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = self._transform_coord(
            df, longitude, latitude, target_srid
        )
        df["offset [m]"] = np.sqrt(
            (df["easting [m]"] - point_east) ** 2
            + (df["northing [m]"] - point_north) ** 2
        )
        return self._gather_data_entity(df)

    def get_closest_entity_3d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        depth: float,
        radius_init: int = 1,
        target_srid: str = "25831",
        sampletest: bool = True,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the entity closest to a certain point in 3D (spherical search area) with optional query arguments.

        :param api_url: End-point for the API
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth of the central point in meters below seabed
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param sampletest: Boolean indicating whether a sample or sample test needs to be retrieved (default is True to search for sample tests)
        :param **kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        geosearch_params = dict(latitude=latitude, longitude=longitude)
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = self._transform_coord(
            df, longitude, latitude, target_srid
        )
        if not sampletest:
            df["depth"] = 0.5 * (df["top_depth"] + df["bottom_depth"])
        df["offset [m]"] = np.sqrt(
            (df["easting [m]"] - point_east) ** 2
            + (df["northing [m]"] - point_north) ** 2
            + (df["depth"] - depth) ** 2
        )
        return self._gather_data_entity(df)

    def get_surveycampaigns(self, projectsite=None, **kwargs):
        """Get all available survey campaigns, specify a projectsite to filter by projectsite.

        :param projectsite: String with the projectsite title (e.g. "Nobelwind")
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the location data for each location in the projectsite
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {"projectsite": projectsite}
        url_params = {**url_params, **kwargs}
        url_data_type = "surveycampaign"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
