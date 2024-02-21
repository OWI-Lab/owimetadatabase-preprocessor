import json
import warnings
from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from groundhog.general.soilprofile import profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import \
    PCPTProcessing
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

    def process_data(
        self, url_data_type: str, url_params: Dict[str, str], output_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Union[bool, int, requests.Response, None]]]:
        """Process output data according to specified request parameters.

        :param url_data_type: Type of the data we want to request (according to database model).
        :param url_params: Parameters to send with the request to the database.
        :param output_type: Expected type (amount) of the data extracted.
        :return: A tuple of dataframe with the requested data and additional data from postprocessing.
        """
        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)
        df = self.output_to_df(resp)
        df_add = self.postprocess_data(df, output_type)
        df_add["response"] = resp
        return df, df_add

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
        self, projectsite: Union[str, None] = None, **kwargs
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
        **kwargs,
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
        self, latitude: float, longitude: float, radius: float, **kwargs
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
            **kwargs,
        )

    def get_closest_testlocation(
        self,
        latitude: float,
        longitude: float,
        radius_init: float = 1,
        target_srid: str = "25831",
        **kwargs,
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
            **kwargs,
        )

    def get_testlocations(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get the geotechnical test locations corresponding to the given search criteria.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
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
        **kwargs,
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
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
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
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the test location answering to the search criteria exists.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Returns the id if test location exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
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
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the location answering to the search criteria for the test location exists.
        If the test location id is required, run the method ``testlocation_exists`` instead.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Returns the id if test location exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

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

    def insitutest_type_exists(
        self, testtype: Union[str, None] = None, **kwargs
    ) -> Union[int, bool]:
        """Checks if the in-situ test type answering to the search criteria exists and returns the id.

        :param testtype: Title of the in-situ test type (e.g. "Downhole PCPT")
        :return: Returns the id if the in-situ test type exists, False otherwise
        """
        url_params = {"testtype": testtype}
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_insitutests(
        self,
        projectsite: str = None,
        location: str = None,
        testtype: str = None,
        insitutest: str = None,
        **kwargs,
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
            "testtype": testtype,
            "insitutest": insitutest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_insitutests(
        self, latitude: float, longitude: float, radius: float, **kwargs
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
            **kwargs,
        )

    def get_closest_insitutest(
        self,
        latitude: float,
        longitude: float,
        radius_init: float = 1,
        target_srid: str = "25831",
        **kwargs,
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
            **kwargs,
        )

    def _process_insitutest_dfs(self, df, cols):
        dfs = {k: None for k in cols}
        for col in cols:
            try:
                df_ = pd.DataFrame(df[col].iloc[0]).reset_index(
                    drop=True
                )
            except:
                df_ = pd.DataFrame()
            dfs[col] = df_
        for df_ in dfs.values():
            try:
                df_ = df_.apply(lambda x: pd.to_numeric(x, errors="ignore"))
            except Exception as err:
                warnings.warn(str(err))
        return dfs
    
    def _combine_dfs(self, dfs):
        try:
            df = pd.merge(dfs["rawdata"], dfs["processeddata"], on="z [m]", how="inner", suffixes=("", "_processed"))
            return df
        except Exception as err:
            warnings.warn(
                f"ERROR: Combining raw and processed data failed - {err}"
            )
            return dfs["rawdata"]

    def _process_cpt(self, df_sum, df_raw, **kwargs):
        try:
            cpt = PCPTProcessing(title=df_sum["title"].iloc[0])
            if "Push" in df_raw.keys():
                push_key = "Push"
            else:
                push_key = None
            cpt.load_pandas(
                df_raw, push_key=push_key, **kwargs
            )
            return cpt
        except Exception as err:
            warnings.warn(
                f"ERROR: PCPTProcessing object not created - {err}"
            )
            return None
 
    def get_insitutest_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        insitutest: Union[str, None] = None,
        combine: bool = False,
        **kwargs
    ):
        """Get the detailed information (measurement data) for an in-situ test of give type.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test
        :param combine: Boolean indicating whether raw and processed data needs to be combined (default=False). If true, processed data columns are appended to the rawdata dataframe
        :param kwargs: Optional keyword arguments for further queryset filtering based on model attributes.
        :return: Dictionary with the following keys:

            - 'id': id of the selected test
            - 'insitutestsummary': Metadata of the insitu tests
            - 'rawdata': Raw data
            - 'processed': Processed data
            - 'conditions': Test conditions
            - 'response': Response text
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = self._combine_dfs(dfs)
        return {
            "id": df_add_detail["id"],
            "insitutestsummary": df_sum,
            "rawdata": df_raw,
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }

    def get_cpttest_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        insitutest: Union[str, None] = None,
        combine: bool = False,
        cpt: bool = True,
        **kwargs
    ):
        """Get the detailed information (measurement data) for an in-situ test of CPT type (seabed or downhole CPT)

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test
        :param combine: Boolean indicating whether raw and processed data needs to be combined (default=False). If true, processed data columns are appended to the rawdata dataframe
        :param cpt: Boolean determining whether the in-situ test is a CPT or not. If True (default), a PCPTProcessing object is returned.
        :param kwargs: Optional keyword arguments for the cpt data loading. Note that further queryset filtering based on model attributes is not possible with this method. The in-situ test needs to be fully defined by the required arguments.
        :return: Dictionary with the following keys:

            - 'id': id of the selected test
            - 'insitutestsummary': Metadata of the insitu tests
            - 'rawdata': Raw data
            - 'processed': Processed data
            - 'conditions': Test conditions
            - 'response': Response text
            - 'cpt': PCPTProcessing object (only if the CPT data is successfully loaded)
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest
        }
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = self._combine_dfs(dfs)
        dict_ = {
            "id": df_add_detail["id"],
            "insitutestsummary": df_sum,
            "rawdata": df_raw,
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }
        if cpt:
            cpt_ = self._process_cpt(df_sum, df_raw, **kwargs)
            if cpt_:
                dict_["cpt"] = cpt_
            return dict_
        return dict_

    def insitutest_exists(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        insitutest: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the in-situ test answering to the search criteria exists.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test
        :return: Returns the id if the in-situ test exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_soilprofiles(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves soil profiles corresponding to the search criteria.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :return: Dictionary with the following keys:

            - 'data': Metadata for the soil profiles
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_soilprofiles(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get all soil profiles in a certain radius surrounding a point with given lat/lon.
        
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the soil profile summary data for each soil profile in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="soilprofileproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs
        )
    
    def get_closest_soilprofile(
        self,
        latitude: float,
        longitude: float,
        initialradius: float = 1.0,
        target_srid: str = "25831",
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the soil profile closest to a certain point with additional conditions as optional keyword arguments.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``location__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the soil profile data for each soil profile in the specified search area
            - 'id': ID of the closest in-situ test
            - 'title': Title of the closest in-situ test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_2d(
            api_url="soilprofileproximity",
            latitude=latitude,
            longitude=longitude,
            initialradius=initialradius,
            target_srid=target_srid,
            **kwargs
        )

    def _convert_to_profile(self, df_sum, df_detail, profile_title, drop_info_cols):
        try:
            soilprofile_df = (
                pd.DataFrame(df_detail["soillayer_set"].iloc[0])
                .sort_values("start_depth")
                .reset_index(drop=True)
            )
            soilprofile_df.rename(
                columns={
                    "start_depth": "Depth from [m]",
                    "end_depth": "Depth to [m]",
                    "soiltype_name": "Soil type",
                    "totalunitweight": "Total unit weight [kN/m3]",
                },
                inplace=True,
            )
            for i, row in soilprofile_df.iterrows():
                try:
                    for key, value in row["soilparameters"].items():
                        soilprofile_df.loc[i, key] = value
                except:
                    pass
            if profile_title is None:
                profile_title = f"{df_sum["location_name"].iloc[0]} - {df_sum["title"].iloc[0]}"            
            if drop_info_cols:
                soilprofile_df.drop(
                    [
                        "id",
                        "profile",
                        "soilparameters",
                        "soilprofile_name",
                        "soilunit",
                        "description",
                        "soilunit_name",
                    ],
                    axis=1,
                    inplace=True,
                )
            dsp = profile_from_dataframe(soilprofile_df, title=profile_title)
            return dsp
        except Exception as err:
            warnings.warn(
                f"Error during loading of soil layers and parameters for {df_sum["title"].iloc[0]} - {err}"
            )
            return None

    def get_soilprofile_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        convert_to_profile: bool = True,
        profile_title: Union[str, None] = None,
        drop_info_cols: bool = True,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, str, bool, None]]:
        """Retrieves a soil profile from the owimetadatabase and converts it to a groundhog SoilProfile object.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :param convert_to_profile: Boolean determining whether the soil profile needs to be converted to a groundhog SoilProfile object
        :param drop_info_cols: Boolean determining whether or not to drop the columns with additional info (e.g. soil description, ...)
        :return: Dictionary with the following keys:

            - 'id': id for the selected soil profile
            - 'soilprofilesummary': Metadata for the soil profile
            - 'response': Response text
            - 'soilprofile': Groundhog SoilProfile object (only if successfully processed)
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        # TODO: Ensure that an option for retrieving soilprofiles in mLAT is also available
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "soilprofiledetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        dict_ =  {
            "id": df_add_detail["id"],
            "soilprofilesummary": df_sum,
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }
        if convert_to_profile:
            dsp = self._convert_to_profile(df_sum, df_detail, profile_title, drop_info_cols)
            if dsp:
                dict_["soilprofile"] = dsp
            return dict_
        return dict_
    
    @staticmethod
    def soilprofile_pisa(
        soil_profile: pd.DataFrame,
        pw: float = 1.025,
        sbl: Union[float, None] = None
    ) -> pd.DataFrame:
        """Converts dataframes with soil profile data to a format suitable for PISA analysis.

        :param soil_profile: Groundhog SoilProfile object obtained through the get_soilprofile_detail method
        :param pw: Sea water density (default=1.025 t/m3)
        :param sbl: Sea bed level in mLAT coordinates
        :return: Dataframe containing soil model to carry out FE analysis through ```owi_monopylat``` of monopile
        following PISA guidance.
        """
        required_keys = [
            "Depth from [m]",
            "Depth to [m]",
            "Soil type",
            "Gmax from [kPa]",
            "Gmax to [kPa]",
            "Su from [kPa]",
            "Su to [kPa]",
            "Dr from [-]",
            "Dr to [-]",
            "Total unit weight [kN/m3]",
        ]
        for req_key in required_keys:
            if req_key not in soil_profile.columns:
                raise ValueError(f"Column key {req_key} not in dataframe")
        pisa_profile = deepcopy(soil_profile[required_keys])
        if sbl:
            pisa_profile["Depth from [mLAT]"] = sbl - pisa_profile["Depth from [m]"]
            pisa_profile["Depth to [mLAT]"] = sbl - pisa_profile["Depth to [m]"]
        else:
            raise ValueError(
                "You need to set a value for the mudline depth in mLAT coordinates."
            )
        g = 9.81  # gravity acceleration (m/s2)
        pisa_profile["Submerged unit weight from [kN/m3]"] = (
            pisa_profile["Total unit weight [kN/m3]"] - pw * g
        )
        pisa_profile["Submerged unit weight to [kN/m3]"] = (
            pisa_profile["Total unit weight [kN/m3]"] - pw * g
        )
        pisa_profile.rename(
            columns={
                "Su from [kPa]": "Undrained shear strength from [kPa]",
                "Su to [kPa]": "Undrained shear strength to [kPa]",
                "Dr from [-]": "Relative density from [-]",
                "Dr to [-]": "Relative density to [-]",
            },
            inplace=True,
        )
        return pisa_profile

    def soilprofile_exists(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the in-situ test answering to the search criteria exists.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :return: Returns the id if soil profile exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofiledetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
    
    def soiltype_exists(
        self,
        soiltype: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if a soiltype with a given name exists.

        :param soiltype: Name of the soil type
        :return: id of the soil type if it exists, False otherwise
        """
        url_params = {"soiltype": soiltype}
        url_params = {**url_params, **kwargs}
        url_data_type = "soiltype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
    
    def soilunit_exists(
        self,
        projectsite: Union[str, None] = None,
        soiltype: Union[str, None] = None,
        soilunit: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if a certain soil unit exists.

        :param projectsite: Name of the project site
        :param soiltype: Name of the soil type
        :param soilunit: Name of the soil unit
        :return: id of the soil unit if it exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "soiltype": soiltype,
            "soilunit": soilunit,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilunit"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
    
    def get_soilunits(
        self,
        projectsite: Union[str, None] = None,
        soiltype: Union[str, None] = None,
        soilunit: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Finds all soil units corresponding to the search parameters.

        :param projectsite: Name of the projectsite (e.g. ``"HKN"``)
        :param soiltype: Name of the soil type (e.g. ``"SAND"``)
        :param soilunit: Name of the soil unit (e.g. ``"Asse sand-clay"``)
        :return: Dictionary with the following keys:

            - 'data': Dataframe with the soil units returned from the query
            - 'exists': Boolean containing whether data is in the returned query
        """
        url_params = {
            "projectsite": projectsite,
            "soiltype": soiltype,
            "soilunit": soilunit,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilunit"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
    
    def get_batchlabtest_types(self, **kwargs) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves the types of batch lab tests available in the database.

        :param kwargs: Keywords arguments for the GET request
        :return: Dataframe with the available InSituTestType records
        """
        url_data_type = "batchlabtesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, kwargs, output_type)
        return {"data": df, "exists": df_add["existance"]}
    
    def batchlabtesttype_exists(
        self,
        testtype: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the in-situ test type answering to the search criteria exists and returns the id.

        :param testtype: Title of the in-situ test type (e.g. "Downhole PCPT")
        :return: Returns the id if the in-situ test type exists, False otherwise
        """
        url_params = {"testtype": testtype}
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_batchlabtests(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        batchlabtest: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves a summary of batch lab tests corresponding to the specified search criteria.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def batchlabtesttype_exists(
        self,
        batchlabtesttype: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the geotechnical sample type answering to the search criteria exists.

        :param batchlabtesttype: Title of the batch lab test type
        :return: Returns the id if the sample type exists, False otherwise
        """
        url_params = {"testtype": batchlabtesttype}
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_proximity_batchlabtests(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Gets all batch lab tests in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the batch lab test summary data for each batch lab test in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="batchlabtestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs
        )
    
    def get_closest_batchlabtest(
        self,
        latitude: float,
        longitude: float,
        initialradius: float = 1.0,
        target_srid: str = "25831",
        **kwargs
    ):
        """Gets the batch lab test closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the batch lab test data for each batch lab test in the specified search area
            - 'id': ID of the closest batch lab test
            - 'title': Title of the closest batch lab test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_2d(
            api_url="batchlabtestproximity",
            latitude=latitude,
            longitude=longitude,
            initialradius=initialradius,
            target_srid=target_srid,
            **kwargs
        )

    def get_batchlabtest_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        batchlabtest: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves detailed data for a specific batch lab test.

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test
        :return: Dictionary with the following keys:

            - 'id': id for the selected soil profile
            - 'summary': Metadata for the batch lab test
            - 'response': Response text
            - 'rawdata': Dataframe with the raw data
            - 'processeddata': Dataframe with the raw data
            - 'conditions': Dataframe with test conditions
            - 'exists': Boolean indicating whether a matching record is found
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "batchlabtestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        return {
            "id": df_add_detail["id"],
            "summary": df_sum,
            "response": df_add_detail["response"],
            "rawdata": dfs["rawdata"],
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }

    def batchlabtest_exists(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        batchlabtest: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the batch lab test answering to the search criteria exists.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test
        :return: Returns the id if batch lab test exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
    
    def geotechnicalsampletype_exists(
        self,
        sampletype: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the geotechnical sample type answering to the search criteria exists.

        :param sampletype: Title of the sample type
        :return: Returns the id if the sample type exists, False otherwise
        """
        url_params = {"sampletype": sampletype}
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsampletype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False
    
    def get_geotechnicalsamples(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        sample: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves geotechnical samples corresponding to the specified search criteria.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the sample
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_geotechnicalsamples(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Gets all geotechnical samples in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for each geotechnical sample in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs
        )

    def get_closest_geotechnicalsample(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        initialradius: float = 1.0,
        target_srid: str = "25831",
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Gets the geotechnical sample closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth: Depth of the central point in meters below seabed
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for each geotechnical sample in the specified search area
            - 'id': ID of the closest batch lab test
            - 'title': Title of the closest batch lab test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_3d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            initialradius=initialradius,
            target_srid=target_srid,
            sampletest=False,
            **kwargs
        )

    def get_geotechnicalsample_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        sample: Union[str, None] = None,
        **kwargs
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """Retrieves detailed data for a specific sample.

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample
        :return: Dictionary with the following keys:

            - 'id': id for the selected soil profile
            - 'data': Metadata for the batch lab test
            - 'response': Response text
            - 'exists': Boolean indicating whether a matching record is found
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {
            "id": df_add["id"],
            "data": df,
            "response": df_add["response"],
            "exists": df_add["existance"]
        }

    def geotechnicalsample_exists(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        sample: Union[str, None] = None,
        **kwargs
    ) -> Union[int, bool]:
        """Checks if the geotechnical sample answering to the search criteria exists.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample
        :return: Returns the id if the geotechnical sample exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False