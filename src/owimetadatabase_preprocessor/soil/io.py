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
    
    :param api_root: Root URL for the API
    :param api_subdir: Subdirectory for the API
    :param token: Token for the API
    :param uname: Username for the API
    :param password: Password for the API
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
        """Constructor for the SoilAPI class."""
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

    def get_surveycampaigns(
        self,
        projectsite: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
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

    def get_surveycampaign_detail(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, int, None]]:
        """Get details for a specific survey campaign.
        
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param campaign: Title of the survey campaign (e.g. "Borehole campaign")
        :return: Dictionary with the following keys:

            - 'id': id of the selected projectsite site
            - 'data': Pandas dataframe with the location data for the individual location
            - 'exists': Boolean indicating whether a matching location is found
        """
        url_params = {"projectsite": projectsite, "campaign": campaign}
        url_params = {**url_params, **kwargs}
        url_data_type = "surveycampaign"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}
    
    def get_proximity_testlocations(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get all soil test locations in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="testlocationproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs
        )
    
    def get_closest_testlocation(
        self,
        latitude: float,
        longitude: float,
        radius_init: float = 1,
        target_srid: str = "25831",
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the soil test location closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_2d(
            api_url="testlocationproximity",
            latitude=latitude,
            longitude=longitude,
            initialradius=radius_init,
            target_srid=target_srid,
            **kwargs
        )
    
    def get_testlocations(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get the geotechnical test locations corresponding to the given search criteria.
        
        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:
            
            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {"projectsite": projectsite, "campaign": campaign, "location": location}
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
    
    def get_testlocation_detail(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, bool, None]]:
        """Get the detailed information for a geotechnical test location.
        
        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:

            - 'id': id of the selected test location
            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {"projectsite": projectsite, "campaign": campaign, "location": location}
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}
    
    def testlocation_exists(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the test location answering to the search criteria exists.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Returns the id if test location exists, False otherwise
        """    
        url_params = {"projectsite": projectsite, "campaign": campaign, "location": location}
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
        
    def testlocation_location_exists(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the location answering to the search criteria for the test location exists.
        If the test location id is required, run the method ``testlocation_exists`` instead.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Returns the id if test location exists, False otherwise
        """
        url_params = {"projectsite": projectsite, "campaign": campaign, "location": location}
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return df["location"].iloc[0] if df_add["existance"] else False

    def plot_testlocations(self, return_fig: bool = False, **kwargs) -> None:
        """Retrieves soil test locations and generates a Plotly plot to show them.

        :param return_fig: Boolean indicating whether the Plotly figure object needs to be returned (default is False which simply shows the plot)
        :param kwargs: Keyword arguments for the search (see ``get_testlocations``)
        :return: Plotly figure object with selected asset locations plotted on OpenStreetMap tiles (if requested)
        """
        testlocations = self.get_testlocations(**kwargs)["data"]
        fig = px.scatter_mapbox(
            testlocations,
            lat="northing",
            lon="easting",
            hover_name="title",
            hover_data=["projectsite_name", "description"],
            zoom=10,
            height=500,
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        if return_fig:
            return fig
        else:
            fig.show()
    
    def get_insitutest_types(self, **kwargs):
        """Retrieves the types of in-situ tests available in the database.

        :param kwargs: Keywords arguments for the GET request
        :return: Dataframe with the available InSituTestType records
        """
        url_data_type = "insitutesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, {}, output_type)
        return {"data": df, "exists": df_add["existance"]}
        
    def insitutest_type_exists(self, test_type: Union[str, None] = None, **kwargs) -> Union[int, bool]:
        """Checks if the in-situ test type answering to the search criteria exists and returns the id.

        :param test_type: Title of the in-situ test type (e.g. "Downhole PCPT")
        :return: Returns the id if the in-situ test type exists, False otherwise
        """
        url_params = {"testtype": test_type}
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_insitutests(
        self,
        projectsite: str = None,
        location: str = None,
        test_type: str = None,
        insitutest: str = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get the detailed information (measurement data) for an in-situ test of given type.
        
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test
        :return: Dictionary with the following keys:

            - 'data': Metadata of the insitu tests
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": test_type,
            "insitutest": insitutest
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_insitutests(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get all in-situ tests in a certain radius surrounding a point with given lat/lon.
        
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the in-situ test summary data for each in-situ test in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="insitutestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs
        )
    
    def get_closest_insitutest(
        self,
        latitude: float,
        longitude: float,
        radius_init: float = 1,
        target_srid: str = "25831",
        **kwargs
    ):
        """Get the in-situ test closest to a certain point with the name containing a certain string.
        
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the in-situ test data for each in-situ test in the specified search area
            - 'id': ID of the closest in-situ test
            - 'title': Title of the closest in-situ test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_2d(
            api_url="insitutestproximity",
            latitude=latitude,
            longitude=longitude,
            initialradius=radius_init,
            target_srid=target_srid,
            **kwargs
        )   

    # def _process_insitutest_dfs(df):
    #     try:
    #         df_raw = pd.DataFrame(df_resp_detail["rawdata"].iloc[0]).reset_index(
    #             drop=True
    #         )
    #     except Exception as err:
    #         df_raw = pd.DataFrame()

    # def get_insitutest_detail(
    #     self,
    #     projectsite: Union[str, None] = None,
    #     location: Union[str, None] = None,
    #     test_type: Union[str, None] = None,
    #     insitutest: Union[str, None] = None,
    #     combine: bool = False,
    #     **kwargs
    # ):
    #     """Get the detailed information (measurement data) for an in-situ test of give type.
        
    #     :param projectsite: Name of the projectsite (e.g. "Nobelwind")
    #     :param location: Name of the test location (e.g. "CPT-7C")
    #     :param testtype: Name of the test type (e.g. "PCPT")
    #     :param insitutest: Name of the in-situ test
    #     :param combine: Boolean indicating whether raw and processed data needs to be combined (default=False). If true, processed data columns are appended to the rawdata dataframe
    #     :param kwargs: Optional keyword arguments for further queryset filtering based on model attributes.
    #     :return: Dictionary with the following keys:

    #         - 'id': id of the selected test
    #         - 'insitutestsummary': Metadata of the insitu tests
    #         - 'rawdata': Raw data
    #         - 'processed': Processed data
    #         - 'conditions': Test conditions
    #         - 'response': Response text
    #         - 'exists': Boolean indicating whether a matching in-situ test is found
    #     """
    #     url_params = {
    #         "projectsite": projectsite,
    #         "location": location,
    #         "testtype": test_type,
    #         "insitutest": insitutest
    #     }
    #     url_params = {**url_params, **kwargs}
    #     url_data_type = "insitutestsummary"
    #     output_type = "single"
    #     df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
    #     url_data_type = "insitutestdetail"
    #     df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
    #     self._process_insitutest_dfs(df_detail)
    #     df_add_sum["existance"], df_add_detail["id"]