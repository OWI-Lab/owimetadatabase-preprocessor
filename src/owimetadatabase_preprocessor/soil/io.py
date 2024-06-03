import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from groundhog.general.soilprofile import plot_fence_diagram, profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import (
    PCPTProcessing,
    plot_combined_longitudinal_profile,
    plot_longitudinal_profile,
)
from pyproj import Transformer

from owimetadatabase_preprocessor.io import API
from owimetadatabase_preprocessor.utils import deepcompare


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
        radius_init: float,
        url_params: Dict[str, str],
        radius_max: float = 500.0,
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
            radius *= 2
            warnings.warn(f"Expanding search radius to {radius: .1f}km")
            if radius > radius_max:
                raise ValueError(
                    f"No locations found within {radius_max}km radius. Check your input information."
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
        transformer = Transformer.from_crs(
            "epsg:4326", "epsg:" + target_srid, always_xy=True
        )
        df["easting [m]"], df["northing [m]"] = transformer.transform(
            df["easting"], df["northing"]
        )
        point_east, point_north = transformer.transform(longitude, latitude)
        return df, point_east, point_north

    def _gather_data_entity(
        self,
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
        :param radius_init: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
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
        radius_init: float = 1.0,
        target_srid: str = "25831",
        sampletest: bool = True,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the entity closest to a certain point in 3D (spherical search area) with optional query arguments.

        :param api_url: End-point for the API
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth: of the central point in meters below seabed
        :param radius_init: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param sampletest: Boolean indicating whether a sample or sample test needs to be retrieved
            (default is True to search for sample tests)
        :param kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
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
        projectsite: str,
        campaign: str,
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
        radius: float = 1.0,
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
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_testlocations(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Get the geotechnical test locations corresponding to the given search criteria.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind", optional, default is None)
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_testlocation_detail(
        self,
        location: str,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, None]]:
        """Get the detailed information for a geotechnical test location.

        :param location: Name of a specific location (e.g. "CPT-888")
        :param projectsite: Optional, name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Optional, name of the survey campaign (e.g. "Borehole campaign")
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
        location: str,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the test location answering to the search criteria exists.

        :param location: Name of a specific location (e.g. "CPT-888")
        :param projectsite: Optional, name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Optional, name of the survey campaign (e.g. "Borehole campaign")
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

        :param return_fig: Boolean indicating whether the Plotly figure object needs to be returned
            (default is False which simply shows the plot)
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

    def insitutest_type_exists(self, testtype: str, **kwargs) -> Union[int, bool]:
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
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        insitutest: Union[str, None] = None,
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
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ):
        """Get the in-situ test closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
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
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def _process_insitutest_dfs(self, df, cols):
        dfs = {k: None for k in cols}
        for col in cols:
            try:
                df_ = pd.DataFrame(df[col].iloc[0]).reset_index(drop=True)
            except KeyError:
                warnings.warn(
                    """
                    Something is wrong with the output dataframe:
                    check that the database gave a non-empty output.

                    Check that you entered correct parameters in your request
                    or contact database administrators.
                    """
                )
                df_ = pd.DataFrame()
            except Exception:
                df_ = pd.DataFrame()
            dfs[col] = df_
        for k, df_ in dfs.items():
            try:
                dfs[k] = df_.apply(lambda x: pd.to_numeric(x, errors="ignore"))
            except Exception as err:
                warnings.warn(str(err))
        return dfs

    def _combine_dfs(self, dfs):
        try:
            df = pd.merge(
                dfs["rawdata"],
                dfs["processeddata"],
                on="z [m]",
                how="inner",
                suffixes=("", "_processed"),
            )
            return df
        except Exception as err:
            warnings.warn(f"ERROR: Combining raw and processed data failed - {err}")
            return dfs["rawdata"]

    def _process_cpt(self, df_sum, df_raw, **kwargs):
        try:
            cpt = PCPTProcessing(title=df_sum["title"].iloc[0])
            if "Push" in df_raw.keys():
                push_key = "Push"
            else:
                push_key = None
            cpt.load_pandas(df_raw, push_key=push_key, **kwargs)
            return cpt
        except Exception as err:
            warnings.warn(f"ERROR: PCPTProcessing object not created - {err}")
            return None

    def get_insitutest_detail(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        combine: bool = False,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """Get the detailed information (measurement data) for an in-situ test of give type.

        :param insitutest: Name of the in-situ test
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param combine: Boolean indicating whether raw and processed data needs to be combined (default=False).
            If true, processed data columns are appended to the rawdata dataframe
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
            "insitutest": insitutest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(
            url_data_type, url_params, output_type
        )
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = self._combine_dfs(dfs)
        else:
            df_raw = dfs["rawdata"]
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
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        combine: bool = False,
        cpt: bool = True,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """Get the detailed information (measurement data) for an in-situ test of CPT type (seabed or downhole CPT)

        :param insitutest: Name of the in-situ test
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param combine: Boolean indicating whether raw and processed data needs to be combined (default=False).
            If true, processed data columns are appended to the rawdata dataframe
        :param cpt: Boolean determining whether the in-situ test is a CPT or not.
            If True (default), a PCPTProcessing object is returned.
        :param kwargs: Optional keyword arguments for the cpt data loading.
            Note that further queryset filtering based on model attributes is not possible with this method.
            The in-situ test needs to be fully defined by the required arguments.
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
            "insitutest": insitutest,
        }
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(
            url_data_type, url_params, output_type
        )
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = self._combine_dfs(dfs)
        else:
            df_raw = dfs["rawdata"]
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
            dict_["cpt"] = cpt_
            return dict_
        return dict_

    def insitutest_exists(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the in-situ test answering to the search criteria exists.

        :param insitutest: Name of the in-situ test
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :return: Returns the id if the in-situ test exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest,
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
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_soilprofiles(
        self, latitude: float, longitude: float, radius: float, **kwargs
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
            **kwargs,
        )

    def get_closest_soilprofile(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Get the soil profile closest to a certain point with additional conditions as optional keyword arguments.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='HKN'``
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
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
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
                except Exception:
                    pass
            if profile_title is None:
                profile_title = (
                    f"{df_sum['location_name'].iloc[0]} - {df_sum['title'].iloc[0]}"
                )
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
        except KeyError:
            warnings.warn(
                """
                Something is wrong with the output dataframe:
                check that the database gave a non-empty output.

                Check that you entered correct parameters in your request
                or contact database administrators.
                """
            )
            return None
        except Exception as err:
            warnings.warn(f"Error during loading of soil layers and parameters: {err}")
            return None

    def get_soilprofile_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        convert_to_profile: bool = True,
        profile_title: Union[str, None] = None,
        drop_info_cols: bool = True,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, bool, requests.Response, None]]:
        """Retrieves a soil profile from the owimetadatabase and converts it to a groundhog SoilProfile object.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :param convert_to_profile: Boolean determining whether the soil profile needs to be converted to a groundhog
            SoilProfile object
        :param drop_info_cols: Boolean determining whether or not to drop the columns with additional info
            (e.g. soil description, ...)
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
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "soilprofiledetail"
        df_detail, df_add_detail = self.process_data(
            url_data_type, url_params, output_type
        )
        dict_ = {
            "id": df_add_detail["id"],
            "soilprofilesummary": df_sum,
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }
        if convert_to_profile:
            dsp = self._convert_to_profile(
                df_sum, df_detail, profile_title, drop_info_cols
            )
            dict_["soilprofile"] = dsp
            return dict_
        return dict_

    @staticmethod
    def soilprofile_pisa(
        soil_profile: pd.DataFrame, pw: float = 1.025, sbl: Union[float, None] = None
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
        if sbl is not None:
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
        soilprofile: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the specific soil profile  exists.

        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
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

    def soiltype_exists(self, soiltype: str, **kwargs) -> Union[int, bool]:
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
        soilunit: str,
        projectsite: Union[str, None] = None,
        soiltype: Union[str, None] = None,
        **kwargs,
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
        **kwargs,
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

    def get_batchlabtest_types(
        self, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves the types of batch lab tests available in the database.

        :param kwargs: Keywords arguments for the GET request
        :return: Dataframe with the available InSituTestType records
        """
        url_data_type = "batchlabtesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, kwargs, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_batchlabtests(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        batchlabtest: Union[str, None] = None,
        **kwargs,
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
        self, batchlabtesttype: str, **kwargs
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
        self, latitude: float, longitude: float, radius: float, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Gets all batch lab tests in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the batch lab test summary data for each batch lab test
                in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="batchlabtestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_batchlabtest(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ):
        """Gets the batch lab test closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
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
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_batchlabtest_detail(
        self,
        batchlabtest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
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
        df_detail, df_add_detail = self.process_data(
            url_data_type, url_params, output_type
        )
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        return {
            "id": df_add_detail["id"],
            "summary": df_sum,
            "response": df_add_detail["response"],
            "rawdata": dfs["rawdata"],
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "exists": df_add_sum["existance"],
        }

    def batchlabtest_exists(
        self,
        batchlabtest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the batch lab test answering to the search criteria exists.

        :param batchlabtest: Title of the batch lab test
        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
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
        self, sampletype: str, **kwargs
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
        **kwargs,
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
        self, latitude: float, longitude: float, radius: float, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Gets all geotechnical samples in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for each geotechnical sample
                in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_geotechnicalsample(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Gets the geotechnical sample closest to a certain point with the name containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth: Depth of the central point in meters below seabed
        :param radius: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for each geotechnical sample
                in the specified search area
            - 'id': ID of the closest batch lab test
            - 'title': Title of the closest batch lab test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_3d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            radius_init=radius,
            target_srid=target_srid,
            sampletest=False,
            **kwargs,
        )

    def get_geotechnicalsample_detail(
        self,
        sample: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """Retrieves detailed data for a specific sample.

        :param sample: Title of the sample
        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
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
            "exists": df_add["existance"],
        }

    def geotechnicalsample_exists(
        self,
        sample: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the geotechnical sample answering to the search criteria exists.

        :param sample: Title of the sample
        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
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

    def get_sampletests(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        sample: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sampletest: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves a summary of geotechnical sample lab tests corresponding to the specified search criteria.

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
        :param sampletest: Title of the sample test
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_sampletests(
        self, latitude: float, longitude: float, radius: float, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Gets all sample tests in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the sample test summary data for each sample test in the specified search area
            - 'exists': Boolean indicating whether matching records are found
        """
        return self.get_proximity_entities_2d(
            api_url="sampletestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_sampletest(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Gets the sample test closest to a certain point.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param Depth: Depth of the central point in meters below seabed
        :param radius: Initial search radius around the central point in km, the search radius is increased
            until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``sample__location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the sample test data for each sample test in the specified search area
            - 'id': ID of the closest sample test
            - 'title': Title of the closest sample test
            - 'offset [m]': Offset in meters from the specified point
        """
        return self.get_closest_entity_3d(
            api_url="sampletestproximity",
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def sampletesttype_exists(self, sampletesttype: str, **kwargs) -> Union[int, bool]:
        """Checks if the sample test type answering to the search criteria exists.

        :param sampletesttype: Title of the sample test type
        :return: Returns the id if the sample test type exists, False otherwise
        """
        url_params = {"testtype": sampletesttype}
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_sampletesttypes(
        self, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Retrieves all sample tests types available in owimetadatabase.

        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_data_type = "sampletesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, kwargs, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_sampletest_detail(
        self,
        sampletest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sample: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """Retrieves detailed information on a specific sample test based on the specified search criteria.

        :param sampletest: Title of the sample test
        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
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
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "sampletestdetail"
        df_detail, df_add_detail = self.process_data(
            url_data_type, url_params, output_type
        )
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = self._process_insitutest_dfs(df_detail, cols)
        return {
            "id": df_add_detail["id"],
            "summary": df_sum,
            "rawdata": dfs["rawdata"],
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }

    def sampletest_exists(
        self,
        sampletest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sample: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> Union[int, bool]:
        """Checks if the batch lab test answering to the search criteria exists.

        :param sampletest: Title of the sample test
        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
        :return: Returns the id if the sample test exists, False otherwise
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_soilunit_depthranges(
        self,
        soilunit: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieves the depth ranges for where the soil unit occurs.

        :param soilunit: Title of the soil unit for which depth ranges need to be retrieved
        :param projectsite: Title of the project site (optional)
        :param location: Title of the test location (optional)
        :return: Dataframe with the depth ranges for the soil unit
        """
        url_params = {
            "soilunit": soilunit,
            "projectsite": projectsite,
            "location": location,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soillayer"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def _fulldata_processing(
        self, unitdata, row, selected_depths, func_get_details, depthcol, **kwargs
    ):
        _fulldata = func_get_details(location=row["location_name"], **kwargs)["rawdata"]
        _depthranges = selected_depths[
            selected_depths["location_name"] == row["location_name"]
        ]
        for _, _layer in _depthranges.iterrows():
            _unitdata = _fulldata[
                (_fulldata[depthcol] >= _layer["start_depth"])
                & (_fulldata[depthcol] <= _layer["end_depth"])
            ]
            unitdata = pd.concat([unitdata, _unitdata])
        unitdata.reset_index(drop=True, inplace=True)
        unitdata.loc[:, "location_name"] = row["location_name"]
        unitdata.loc[:, "projectsite_name"] = row["projectsite_name"]
        unitdata.loc[:, "test_type_name"] = row["test_type_name"]
        return unitdata

    def _partialdata_processing(self, unitdata, row, selected_depths, selected_tests):
        _depthranges = selected_depths[
            selected_depths["location_name"] == row["location_name"]
        ]
        for _, _layer in _depthranges.iterrows():
            if (
                row["depth"] >= _layer["start_depth"]
                and row["depth"] <= _layer["end_depth"]
            ):
                _unitdata = selected_tests[selected_tests["id"] == row["id"]]
                unitdata = pd.concat([unitdata, _unitdata])
            else:
                pass
        unitdata.reset_index(drop=True, inplace=True)

    def _process_data_units(
        self,
        soilunit: str,
        func_get: Callable,
        func_get_details: Union[Callable, None] = None,
        depthcol: Union[str, None] = None,
        full: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = func_get(**kwargs)["data"]
        all_unit_data = pd.DataFrame()
        for _, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()
            if row["location_name"] in selected_depths["location_name"].unique():
                if full:
                    unitdata = self._fulldata_processing(
                        unitdata,
                        row,
                        selected_depths,
                        func_get_details,
                        depthcol,
                        **kwargs,
                    )
                else:
                    unitdata = self._partialdata_processing(
                        unitdata, row, selected_depths, selected_tests
                    )
            else:
                print(f"Soil unit not found for {row['location_name']}")
            all_unit_data = pd.concat([all_unit_data, unitdata])
        all_unit_data.reset_index(drop=True, inplace=True)
        return all_unit_data

    def get_unit_insitutestdata(
        self, soilunit: str, depthcol: Union[str, None] = "z [m]", **kwargs
    ) -> pd.DataFrame:
        """Retrieves proportions of in-situ test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with in-situ test data in the selected soil unit.
        """
        return self._process_data_units(
            soilunit,
            self.get_insitutests,
            self.get_insitutest_detail,
            depthcol=depthcol,
            **kwargs,
        )

    def get_unit_batchlabtestdata(
        self, soilunit: str, depthcol: Union[str, None] = "z [m]", **kwargs
    ) -> pd.DataFrame:
        """Retrieves proportions of batch lab test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with batch lab test data in the selected soil unit.
        """
        return self._process_data_units(
            soilunit,
            depthcol,
            self.get_batchlabtests,
            self.get_batchlabtest_detail,
            depthcol=depthcol,
            **kwargs,
        )

    def get_unit_sampletests(self, soilunit: str, **kwargs) -> pd.DataFrame:
        """Retrieves the sample tests data located inside a soil unit.
        The metadata of the samples is filtered based on the depth column.
        Further retrieval of the test data can follow after this method.

        :param soilunit: Name of the soil unit
        :param kwargs: Optional keyword arguments for retrieval of sample tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with sample test metadata in the selected soil unit.
        """
        return self._process_data_units(soilunit, self.get_sampletests, **kwargs)

    def _objects_to_list(self, selected_obj, func_get_detail, data_type):
        obj = []
        for _, row in selected_obj.iterrows():
            try:
                if data_type == "soilprofile":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "soilprofile": row["title"],
                        "drop_info_cols": False,
                        "profile_title": row["location_name"],
                    }
                elif data_type == "cpt":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "insitutest": row["title"],
                        "testtype": row["test_type_name"],
                    }
                else:
                    raise ValueError(f"Data type {data_type} not supported.")
                _obj = func_get_detail(**params)[data_type]
                _obj.set_position(
                    easting=row["easting"],
                    northing=row["northing"],
                    elevation=row["elevation"],
                )
                obj.append(_obj)
            except Exception:
                warnings.warn(
                    f"Error loading {row['projectsite_name']}-{row['location_name']}-{row['title']}"
                )
        return obj

    def get_soilprofile_profile(
        self, lat1: float, lon1: float, lat2: float, lon2: float, band: float = 1000
    ) -> pd.DataFrame:
        """Retrieves soil profiles along a profile line.

        :param lat1: Latitude of the start point
        :param lon1: Longitude of the start point
        :param lat2: Latitude of the end point
        :param lon2: Longitude of the end point
        :param band: Thickness of the band (in m, default=1000m)
        :return: Returns a dataframe with the summary data of the selected soil profiles
        """
        url_params = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "offset": band,
        }
        url_data_type = "soilprofileprofile"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def plot_soilprofile_fence(
        self,
        soilprofiles_df: pd.DataFrame,
        start: str,
        end: str,
        plotmap: bool = False,
        fillcolordict: Dict[str, str] = {
            "SAND": "yellow",
            "CLAY": "brown",
            "SAND/CLAY": "orange",
        },
        logwidth: float = 100.0,
        show_annotations: bool = True,
        general_layout: Dict[Any, Any] = dict(),
        **kwargs,
    ) -> Dict[str, Union[List[pd.DataFrame], go.Figure]]:
        """Creates a fence diagram for soil profiles.

        :param soilprofiles_df: Dataframe with summary data for the selected soil profiles
        :param start: Name of the soil profile at the start
        :param end: Name of the soil profile at the end
        :param plotmap: Boolean determining whether a map with the locations is shown (default=False)
        :param fillcolordict: Dictionary used for mapping soil types to colors
        :param logwidth: Width of the logs in the fence diagram (default=100)
        :param show_annotations: Boolean determining whether annotations are shown (default=True)
        :param general_layout: Dictionary with general layout options (default = dict())
        :param kwargs: Keyword arguments for the get_soilprofiles method
        :return: Dictionary with the following keys:

            - 'profiles': List of SoilProfile objects
            - 'diagram': Plotly figure with the fence diagram
        """
        selected_profiles = soilprofiles_df
        soilprofiles = self._objects_to_list(
            selected_profiles, self.get_soilprofile_detail, "soilprofile"
        )
        fence_diagram_1 = plot_fence_diagram(
            profiles=soilprofiles,
            start=start,
            end=end,
            plotmap=plotmap,
            latlon=True,
            fillcolordict=fillcolordict,
            logwidth=logwidth,
            show_annotations=show_annotations,
            general_layout=general_layout,
            **kwargs,
        )
        return {"profiles": soilprofiles, "diagram": fence_diagram_1}

    def get_insitutests_profile(
        self, lat1: float, lon1: float, lat2: float, lon2: float, band: float = 1000
    ) -> pd.DataFrame:
        """Retrieves in-situ tests along a profile line.

        :param lat1: Latitude of the start point
        :param lon1: Longitude of the start point
        :param lat2: Latitude of the end point
        :param lon2: Longitude of the end point
        :param band: Thickness of the band (in m, default=1000m)
        :return: Returns a dataframe with the summary data of the selected in-situ tests
        """
        url_params = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "offset": band,
        }
        url_data_type = "insitutestprofile"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def plot_cpt_fence(
        self,
        cpt_df: pd.DataFrame,
        start: str,
        end: str,
        band: float = 1000.0,
        scale_factor: float = 10.0,
        extend_profile: bool = True,
        plotmap: bool = False,
        show_annotations: bool = True,
        general_layout: Dict[Any, Any] = dict(),
        uniformcolor: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, Union[List[pd.DataFrame], go.Figure]]:
        """Creates a fence diagram for CPTs.

        :param cpt_df: Dataframe with the summary data of the selected CPTs
        :param start: Name of the location for the start point
        :param end: Name of the location for the end point
        :param band: Thickness of the band (in m, default=1000m)
        :param scale_factor: Width of the CPT axis in the fence diagram (default=10)
        :param extend_profile: Boolean determining whether the profile needs to be extended (default=True)
        :param plotmap: Boolean determining whether a map with the locations is shown (default=False)
        :param show_annotations: Boolean determining whether annotations are shown (default=True)
        :param general_layout: Dictionary with general layout options (default = dict())
        :param uniformcolor: If a valid color is provided (e.g. 'black'), it is used for all CPT traces
        :param kwargs: Keyword arguments for the get_insitutests method
        :return: Dictionary with the following keys:

            - 'cpts': List of CPT objects
            - 'diagram': Plotly figure with the fence diagram
        """
        selected_cpts = cpt_df
        cpts = self._objects_to_list(selected_cpts, self.get_cpttest_detail, "cpt")
        cpt_fence_fig_1 = plot_longitudinal_profile(
            cpts=cpts,
            latlon=True,
            start=start,
            end=end,
            band=band,
            scale_factor=scale_factor,
            extend_profile=extend_profile,
            plotmap=plotmap,
            show_annotations=show_annotations,
            general_layout=general_layout,
            uniformcolor=uniformcolor,
            **kwargs,
        )
        return {"cpts": cpts, "diagram": cpt_fence_fig_1}

    def plot_combined_fence(
        self,
        profiles: List[pd.DataFrame],
        cpts: List[pd.DataFrame],
        startpoint: str,
        endpoint: str,
        band: float = 1000.0,
        scale_factor: float = 10.0,
        extend_profile: bool = True,
        show_annotations: bool = True,
        general_layout: Dict[Any, Any] = dict(),
        fillcolordict: Dict[str, str] = {
            "SAND": "yellow",
            "CLAY": "brown",
            "SAND/CLAY": "orange",
        },
        logwidth: float = 100.0,
        opacity: float = 0.5,
        uniformcolor: Union[str, None] = None,
        **kwargs,
    ) -> Dict[str, go.Figure]:
        """Creates a combined fence diagram with soil profile and CPT data.

        :param profiles: List with georeferenced soil profiles (run plot_soilprofile_fence first)
        :param cpts: List with georeference CPTs (run plot_cpt_fence first)
        :param startpoint: Name of the CPT location for the start point
        :param endpoint: Name of the CPT location for the end point
        :param band: Thickness of the band (in m, default=1000m)
        :param scale_factor: Width of the CPT axis in the fence diagram (default=10)
        :param extend_profile: Boolean determining whether the profile needs to be extended (default=True)
        :param show_annotations: Boolean determining whether annotations are shown (default=True)
        :param general_layout: Dictionary with general layout options (default = dict())
        :param fillcolordict: Dictionary with colors for soil types
        :param logwidth: Width of the log in the fence diagram
        :param opacity: Opacity of the soil profile logs
        :param uniformcolor: If a valid color is provided (e.g. 'black'), it is used for all CPT traces
        :return: Dictionary with the following keys:

            - 'diagram': Plotly figure with the fence diagram for CPTs and soil profiles
        """
        combined_fence_fig_1 = plot_combined_longitudinal_profile(
            cpts=cpts,
            profiles=profiles,
            latlon=True,
            start=startpoint,
            end=endpoint,
            band=band,
            scale_factor=scale_factor,
            logwidth=logwidth,
            opacity=opacity,
            extend_profile=extend_profile,
            show_annotations=show_annotations,
            uniformcolor=uniformcolor,
            fillcolordict=fillcolordict,
            general_layout=general_layout,
            **kwargs,
        )
        return {"diagram": combined_fence_fig_1}

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            comp = deepcompare(self, other)
            assert comp[0], comp[1]
        elif isinstance(other, dict):
            comp = deepcompare(self.__dict__, other)
            assert comp[0], comp[1]
        else:
            assert False, "Comparison is not possible due to incompatible types!"
        return comp[0]
