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

    def get_proximity_entities_2d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        radius: float,
        **kwargs
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
        url_data_type = "/soildata/" + api_url
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}
    
    def _search_any_entity(
        self,
        api_url: str,
        radius_init: int,
        url_params: Dict[str, str],
        radius_max: int = 500
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
            url_data_type = "/soildata/" + api_url
            output_type = "list"
            df, df_add = self.process_data(url_data_type, url_params, output_type)
            if df_add["existance"]:
                break
            warnings.warn("Expanding search radius to")
            radius *= 2
            warnings.warn(f"Expanding search radius to {radius: .1f}km")
            if radius > radius_max:
                raise ValueError("No locations found within 500km radius. Check your input information.")
        return df
    
    def _transform_coord(
        self,
        df: pd.DataFrame,
        longitude: float,
        latitude: float,
        target_srid: str
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
        df["easting [m]"], df["northing [m]"] = transformer.transform(df["easting"], df["northing"])
        point_east, point_north = transformer.transform(longitude, latitude)
        return df, point_east, point_north
    
    def _gather_data_entity(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
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
            "offset [m]": df[df["offset [m]"] == df["offset [m]"].min()]["offset [m]"].iloc[0],
        }

    def get_closest_entity_2d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        radius_init: float = 1.,
        target_srid: str = "25831",
        **kwargs
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
        df, point_east, point_north = self._transform_coord(df, longitude, latitude, target_srid)
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
        **kwargs
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
        df, point_east, point_north = self._transform_coord(df, longitude, latitude, target_srid)
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
        url_data_type = "/soildata/surveycampaign/"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_surveycampaign_detail(self, projectsite=None, campaign=None, **kwargs):
        """
        Get a selected survey campaign
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param campaign: Title of the survey campaign (e.g. "Borehole campaign")
        :return: Dictionary with the following keys:
            - 'id': id of the selected projectsite site
            - 'data': Pandas dataframe with the location data for the individual location
            - 'exists': Boolean indicating whether a matching location is found
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign],
            parameternames=["projectsite", "campaign"],
        )

        url_params = {**url_params, **kwargs}

        resp_assetlocation = requests.get(
            url="%s/surveycampaign/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_assetlocation.text))

        if df.__len__() == 0:
            exists = False
            campaign_id = None
        elif df.__len__() == 1:
            exists = True
            campaign_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one asset location was returned, check search criteria."
            )

        return {"id": campaign_id, "data": df, "exists": exists}

    def get_proximity_testlocations(self, latitude, longitude, radius, **kwargs):
        """
        Get all soil test locations in a certain radius surrounding a point with given lat/lon
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
        self, latitude, longitude, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the soil test location closest to a certain point with the name containing a certain string
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
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
            initialradius=initialradius,
            target_srid=target_srid,
            **kwargs
        )

    def get_testlocations(
        self, projectsite=None, campaign=None, location=None, **kwargs
    ):
        """
        Get the geotechnical test locations corresponding to the given search criteria
        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:
            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {}

        for param, paramname in zip(
            [projectsite, campaign, location], ["projectsite", "campaign", "location"]
        ):
            url_params[paramname] = param

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/testlocation/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df, "exists": exists}

    def get_testlocation_detail(
        self, projectsite=None, location=None, campaign=None, **kwargs
    ):
        """
        Get the detailed information for a geotechnical test location
        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)
        :return: Dictionary with the following keys:
            - 'id': id of the selected test location
            - 'data': Pandas dataframe with the test location data for each location meeting the specified search criteria
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location],
            parameternames=["projectsite", "campaign", "location"],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/testlocation/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            exists = False
            id = None
        elif df.__len__() == 1:
            exists = True
            id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one test location was returned, refine search criteria"
            )

        return {"id": id, "data": df, "exists": exists}

    def testlocation_exists(
        self, projectsite=None, location=None, campaign=None, **kwargs
    ):
        """
        Checks if the test location answering to the search criteria exists

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)

        :return: Returns the id if test location exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location],
            parameternames=["projectsite", "campaign", "location"],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/testlocation/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one test location was returned, refine search criteria"
            )

        return record_id

    def testlocation_location_exists(
        self, projectsite=None, location=None, campaign=None, **kwargs
    ):
        """
        Checks if the location answering to the search criteria for the test location exists.
        If the test location id is required, run the method ``testlocation_exists`` instead.

        :param projectsite: Name of the projectsite under consideration (e.g. "Nobelwind")
        :param campaign: Name of the survey campaign (optional, default is None to return all locations in a projectsite)
        :param location: Name of a specific location (optional, default is None to return all locations in a projectsite)

        :return: Returns the id if test location exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location],
            parameternames=["projectsite", "campaign", "location"],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/testlocation/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["location"].iloc[0]
        else:
            raise ValueError(
                "More than one test location was returned, refine search criteria"
            )

        return record_id

    def plot_testlocations(self, return_fig=False, **kwargs):
        """
        Retrieves soil test locations and generates a Plotly plot to show them
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
        """
        Retrieves the types on in-situ tests available in the database
        :param kwargs: Keywords arguments for the GET request
        :return: Dataframe with the available InSituTestType records
        """
        resp = requests.get(
            url="%s/insitutesttype/" % self.api_root, headers=self.header, params=kwargs
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df, "exists": exists}

    def insitutesttype_exists(self, testtype=None, **kwargs):
        """
        Checks if the in-situ test type answering to the search criteria exists and returns the id

        :param testtype: Title of the in-situ test type (e.g. "Downhole PCPT")
        :return: Returns the id if the in-situ test type exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                testtype,
            ],
            parameternames=[
                "testtype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/insitutesttype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one in-situ test type was returned, refine search criteria"
            )

        return record_id

    def get_insitutests(
        self, projectsite=None, location=None, testtype=None, insitutest=None, **kwargs
    ):
        """
        Get the detailed information (measurement data) for an in-situ test of give type
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test

        :return: Dictionary with the following keys:
            - 'data': Metadata of the insitu tests
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """
        url_params = self.urlparameters(
            parameters=[projectsite, location, testtype, insitutest],
            parameternames=["projectsite", "location", "testtype", "insitutest"],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/insitutestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_proximity_insitutests(self, latitude, longitude, radius, **kwargs):
        """
        Get all in-situ tests in a certain radius surrounding a point with given lat/lon
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
        self, latitude, longitude, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the in-situ test closest to a certain point with the name containing a certain string
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
            initialradius=initialradius,
            target_srid=target_srid,
            **kwargs
        )

    def get_insitutest_detail(
        self,
        projectsite=None,
        location=None,
        testtype=None,
        insitutest=None,
        combine=False,
        **kwargs
    ):
        """
        Get the detailed information (measurement data) for an in-situ test of give type
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
        url_params = self.urlparameters(
            parameters=[projectsite, location, testtype, insitutest],
            parameternames=["projectsite", "location", "testtype", "insitutest"],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/insitutestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            id = None
        elif df_summary.__len__() == 1:
            exists = True
        else:
            raise ValueError(
                "More than one in-situ test was returned, refine your search parameters."
            )

        resp_detail = requests.get(
            url="%s/insitutestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_resp_detail = pd.DataFrame(json.loads(resp_detail.text))
        id = df_resp_detail["id"].iloc[0]
        try:
            df_raw = pd.DataFrame(df_resp_detail["rawdata"].iloc[0]).reset_index(
                drop=True
            )
        except Exception as err:
            df_raw = pd.DataFrame()
        try:
            df_processed = pd.DataFrame(
                df_resp_detail["processeddata"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_processed = pd.DataFrame()
        try:
            df_conditions = pd.DataFrame(
                df_resp_detail["conditions"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_conditions = pd.DataFrame()

        for _df in [df_raw, df_processed, df_conditions]:
            for col in _df.columns:
                try:
                    _df[col] = pd.to_numeric(_df[col], errors="ignore")
                except Exception as err:
                    warnings.warn(str(err))

        # Merge raw and processed cpt data
        if combine:
            try:
                df_raw = pd.merge(
                    df_raw,
                    df_processed,
                    how="inner",
                    on="z [m]",
                    suffixes=["", "_processed"],
                )
            except Exception as err:
                warnings.warn(
                    "ERROR: Combining raw and processed data failed - %s" % str(err)
                )

        return_dict = {
            "id": id,
            "insitutestsummary": df_summary,
            "rawdata": df_raw,
            "processed": df_processed,
            "conditions": df_conditions,
            "response": resp_detail,
            "exists": exists,
        }

        return return_dict

    def get_cpttest_detail(
        self,
        projectsite=None,
        location=None,
        testtype=None,
        insitutest=None,
        combine=False,
        cpt=True,
        **kwargs
    ):
        """
        Get the detailed information (measurement data) for an in-situ test of CPT type (seabed or downhole CPT)
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
        url_params = self.urlparameters(
            parameters=[projectsite, location, testtype, insitutest],
            parameternames=["projectsite", "location", "testtype", "insitutest"],
        )

        resp_summary = requests.get(
            url="%s/insitutestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            id = None
        elif df_summary.__len__() == 1:
            exists = True
        else:
            raise ValueError(
                "More than one in-situ test was returned, refine your search parameters."
            )

        resp_detail = requests.get(
            url="%s/insitutestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_resp_detail = pd.DataFrame(json.loads(resp_detail.text))
        id = df_resp_detail["id"].iloc[0]
        try:
            df_raw = pd.DataFrame(df_resp_detail["rawdata"].iloc[0]).reset_index(
                drop=True
            )
        except Exception as err:
            df_raw = pd.DataFrame()
        try:
            df_processed = pd.DataFrame(
                df_resp_detail["processeddata"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_processed = pd.DataFrame()
        try:
            df_conditions = pd.DataFrame(
                df_resp_detail["conditions"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_conditions = pd.DataFrame()

        for _df in [df_raw, df_processed, df_conditions]:
            for col in _df.columns:
                try:
                    _df[col] = pd.to_numeric(_df[col], errors="ignore")
                except Exception as err:
                    warnings.warn(str(err))

        # Merge raw and processed cpt data
        if combine:
            try:
                df_raw = pd.merge(
                    df_raw,
                    df_processed,
                    how="inner",
                    on="z [m]",
                    suffixes=["", "_processed"],
                )
            except Exception as err:
                warnings.warn(
                    "ERROR: Combining raw and processed data failed - %s" % str(err)
                )

        return_dict = {
            "id": id,
            "insitutestsummary": df_summary,
            "rawdata": df_raw,
            "processed": df_processed,
            "conditions": df_conditions,
            "response": resp_detail,
            "exists": exists,
        }

        if cpt:
            try:
                cpt = PCPTProcessing(title=df_summary["title"].iloc[0])
                if (
                    "Push" in df_raw.keys()
                ):  # Check if a key for the push is available (only for downhole CPTs)
                    push_key = "Push"
                else:
                    push_key = None
                cpt.load_pandas(
                    df_raw, push_key=push_key, **kwargs
                )  # Load the data into the PCPTProcessing object
                return_dict["cpt"] = cpt
            except Exception as err:
                warnings.warn(
                    "ERROR: PCPTProcessing object not created - %s" % str(err)
                )

        return return_dict

    def insitutest_exists(
        self, projectsite=None, location=None, testtype=None, insitutest=None, **kwargs
    ):
        """
        Checks if the in-situ test answering to the search criteria exists

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test

        :return: Returns the id if the in-situ test exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, location, testtype, insitutest],
            parameternames=["projectsite", "location", "testtype", "insitutest"],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/insitutestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one in-situ test was returned, refine search criteria"
            )

        return record_id

    def get_soilprofiles(
        self, projectsite=None, location=None, soilprofile=None, **kwargs
    ):
        """
        Retrieves soil profiles corresponding to the search criteria

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")

        :return: Dictionary with the following keys:
            - 'data': Metadata for the soil profiles
            - 'exists': Boolean indicating whether a matching in-situ test is found
        """

        url_params = {
            **self.urlparameters(
                parameters=[projectsite, location, soilprofile],
                parameternames=["projectsite", "location", "soilprofile"],
            ),
            **kwargs,
        }

        resp_summary = requests.get(
            url="%s/soilprofilesummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_proximity_soilprofiles(self, latitude, longitude, radius, **kwargs):
        """
        Get all soil profiles in a certain radius surrounding a point with given lat/lon
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
        self, latitude, longitude, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the soil profile closest to a certain point with additional conditions as optional keyword arguments
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

    def get_soilprofile_detail(
        self,
        projectsite=None,
        location=None,
        soilprofile=None,
        convert_to_profile=True,
        profile_title=None,
        drop_info_cols=True,
        **kwargs
    ):
        """
        Retrieves a soil profile from the owimetadatabase and converts it to a groundhog SoilProfile object

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

        url_params = self.urlparameters(
            parameters=[projectsite, location, soilprofile],
            parameternames=["projectsite", "location", "soilprofile"],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/soilprofilesummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            id = None
        elif df_summary.__len__() == 1:
            exists = True
        else:
            raise ValueError(
                "More than one soil profile was returned, refine your search parameters."
            )

        resp_detail = requests.get(
            url="%s/soilprofiledetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_resp_detail = pd.DataFrame(json.loads(resp_detail.text))
        id = df_resp_detail["id"].iloc[0]

        return_dict = {
            "id": id,
            "soilprofilesummary": df_summary,
            "response": resp_detail,
            "exists": exists,
        }

        if convert_to_profile:
            try:
                soilprofile_df = (
                    pd.DataFrame(df_resp_detail["soillayer_set"].iloc[0])
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
                    profile_title = "%s - %s" % (
                        df_summary["location_name"].iloc[0],
                        df_summary["title"].iloc[0],
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
                return_dict["soilprofile"] = dsp

            except Exception as err:
                warnings.warn(
                    "Error during loading of soil layers and parameters for %s - %s"
                    % (df_summary["title"].iloc[0], str(err))
                )

        return return_dict

    @staticmethod
    def soilprofile_pisa(soil_profile, pw=1.025, sbl=None):
        """
        :param soil_profile: dataframe, Groundhog SoilProfile object obtained through the get_soilprofile_detail method
        :param pw: float, sea water density (default=1.025 t/m3)
        :param sbl: float, sea bed level in mLAT coordinates

        :return: dataframe containing soil model to carry out FE analysis through ```owi_monopylat``` of monopile
        following PISA guidance.
        """
        # Parameters for PISA model
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
                raise ValueError("Column key {} not in dataframe".format(req_key))

        pisa_profile = deepcopy(soil_profile[required_keys])
        # Calculating depths in mLAT coordinates
        if sbl:
            pisa_profile["Depth from [mLAT]"] = sbl - pisa_profile["Depth from [m]"]
            pisa_profile["Depth to [mLAT]"] = sbl - pisa_profile["Depth to [m]"]
        else:
            raise ValueError(
                "You need to set a value for the mudline depth in mLAT coordinates."
            )

        # Calculating submerged unit weight of soil
        g = 9.81  # gravity acceleration (m/s2)
        pisa_profile["Submerged unit weight from [kN/m3]"] = (
            pisa_profile["Total unit weight [kN/m3]"] - pw * g
        )
        pisa_profile["Submerged unit weight to [kN/m3]"] = (
            pisa_profile["Total unit weight [kN/m3]"] - pw * g
        )

        # Renaming columns
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
        self, projectsite=None, location=None, soilprofile=None, **kwargs
    ):
        """
        Checks if the in-situ test answering to the search criteria exists

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")

        :return: Returns the id if soil profile exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, location, soilprofile],
            parameternames=["projectsite", "location", "soilprofile"],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/soilprofiledetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one soil profile was returned, refine search criteria"
            )

        return record_id

    def soiltype_exists(self, soiltype=None, **kwargs):
        """
        Check if a soiltype with a given name exists
        :param soiltype: Name of the soil type
        :return: id of the soil type if it exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                soiltype,
            ],
            parameternames=[
                "soiltype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/soiltype/" % self.api_root, headers=self.header, params=url_params
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one soil type was returned, refine search criteria"
            )

        return record_id

    def soilunit_exists(self, projectsite=None, soiltype=None, soilunit=None, **kwargs):
        """
        Check if a certain soil unit exists
        :param projectsite: Name of the project site
        :param soiltype: Name of the soil type
        :param soilunit: Name of the soil unit
        :return: id of the soil unit if it exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, soiltype, soilunit],
            parameternames=["projectsite", "soiltype", "soilunit"],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/soilunit/" % self.api_root, headers=self.header, params=url_params
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one soil unit was returned, refine search criteria"
            )

        return record_id

    def get_soilunits(self, projectsite=None, soiltype=None, soilunit=None, **kwargs):
        """
        Find all soil units corresponding to the search parameters
        :param projectsite: Name of the projectsite (e.g. ``"HKN"``)
        :param soiltype: Name of the soil type (e.g. ``"SAND"``)
        :param soilunit: Name of the soil unit (e.g. ``"Asse sand-clay"``)
        :return: Dictionary with the following keys:
            - 'data': Dataframe with the soil units returned from the query
            - 'exists': Boolean containing whether data is in the returned query
        """
        url_params = self.urlparameters(
            parameters=[projectsite, soiltype, soilunit],
            parameternames=["projectsite", "soiltype", "soilunit"],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/soilunit/" % self.api_root, headers=self.header, params=url_params
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_batchlabtest_types(self, **kwargs):
        """
        Retrieves the types of batch lab tests available in the database
        :param kwargs: Keywords arguments for the GET request
        :return: Dataframe with the available InSituTestType records
        """
        resp = requests.get(
            url="%s/batchlabtesttype/" % self.api_root,
            headers=self.header,
            params=kwargs,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df, "exists": exists}

    def batchlabtesttype_exists(self, testtype=None, **kwargs):
        """
        Checks if the in-situ test type answering to the search criteria exists and returns the id

        :param testtype: Title of the in-situ test type (e.g. "Downhole PCPT")
        :return: Returns the id if the in-situ test type exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                testtype,
            ],
            parameternames=[
                "testtype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_testlocations = requests.get(
            url="%s/batchlabtesttype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp_testlocations.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one batch lab test type was returned, refine search criteria"
            )

        return record_id

    def get_batchlabtests(
        self,
        projectsite=None,
        campaign=None,
        location=None,
        testtype=None,
        batchlabtest=None,
        **kwargs
    ):
        """
        Retrieve a summary of batch lab tests corresponding to the specified search criteria.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test
        :return: Dictionary with the following keys
            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, testtype, batchlabtest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "testtype",
                "batchlabtest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/batchlabtestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def batchlabtesttype_exists(self, batchlabtesttype=None, **kwargs):
        """
        Checks if the geotechnical sample type answering to the search criteria exists

        :param sampletype: Title of the sample type

        :return: Returns the id if the sample type exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                batchlabtesttype,
            ],
            parameternames=[
                "testtype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/batchlabtesttype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one batch lab test type was returned, refine search criteria"
            )

        return record_id

    def get_proximity_batchlabtests(self, latitude, longitude, radius, **kwargs):
        """
        Get all batch lab tests in a certain radius surrounding a point with given lat/lon
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
        self, latitude, longitude, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the batch lab test closest to a certain point with the name containing a certain string
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
        projectsite=None,
        location=None,
        testtype=None,
        campaign=None,
        batchlabtest=None,
        **kwargs
    ):
        """
        Retrieve detailed data for a specific batch lab test

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
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, testtype, batchlabtest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "testtype",
                "batchlabtest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/batchlabtestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            id = None
        elif df_summary.__len__() == 1:
            exists = True
        else:
            raise ValueError(
                "More than one batch lab test was returned, refine your search parameters."
            )

        resp_detail = requests.get(
            url="%s/batchlabtestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_resp_detail = pd.DataFrame(json.loads(resp_detail.text))
        id = df_resp_detail["id"].iloc[0]

        try:
            df_raw = pd.DataFrame(df_resp_detail["rawdata"].iloc[0]).reset_index(
                drop=True
            )
        except Exception as err:
            df_raw = pd.DataFrame()
        try:
            df_processed = pd.DataFrame(
                df_resp_detail["processeddata"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_processed = pd.DataFrame()
        try:
            df_conditions = pd.DataFrame(
                df_resp_detail["conditions"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_conditions = pd.DataFrame()

        for _df in [df_raw, df_processed, df_conditions]:
            for col in _df.columns:
                try:
                    _df[col] = pd.to_numeric(_df[col], errors="ignore")
                except:
                    pass

        return {
            "id": id,
            "summary": df_summary,
            "rawdata": df_raw,
            "processed": df_processed,
            "conditions": df_conditions,
            "response": resp_detail,
            "exists": exists,
        }

    def batchlabtest_exists(
        self,
        projectsite=None,
        location=None,
        testtype=None,
        campaign=None,
        batchlabtest=None,
        **kwargs
    ):
        """
        Checks if the batch lab test answering to the search criteria exists

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test

        :return: Returns the id if batch lab test exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, testtype, batchlabtest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "testtype",
                "batchlabtest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/batchlabtestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one batch lab test was returned, refine search criteria"
            )

        return record_id

    def geotechnicalsampletype_exists(self, sampletype=None, **kwargs):
        """
        Checks if the geotechnical sample type answering to the search criteria exists

        :param sampletype: Title of the sample type

        :return: Returns the id if the sample type exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                sampletype,
            ],
            parameternames=[
                "sampletype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/geotechnicalsampletype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one sample type was returned, refine search criteria"
            )

        return record_id

    def get_geotechnicalsamples(
        self,
        projectsite=None,
        campaign=None,
        location=None,
        sampletype=None,
        sample=None,
        **kwargs
    ):
        """
        Retrieve geotechnical samples corresponding to the specified search criteria
        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample
        :return: Dictionary with the following keys
            - 'data': Dataframe with details on the sample
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sampletype, sample],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sampletype",
                "sample",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/geotechnicalsample/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_proximity_geotechnicalsamples(self, latitude, longitude, radius, **kwargs):
        """
        Get all geotechnical samples in a certain radius surrounding a point with given lat/lon
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
        self, latitude, longitude, depth, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the geotechnical sample closest to a certain point with the name containing a certain string
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
        projectsite=None,
        location=None,
        sampletype=None,
        campaign=None,
        sample=None,
        **kwargs
    ):
        """
        Retrieve detailed data for a specific sample.

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
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sampletype, sample],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sampletype",
                "sample",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/geotechnicalsample/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            id = None
        elif df_summary.__len__() == 1:
            exists = True
            id = df_summary["id"].iloc[0]
        else:
            raise ValueError(
                "More than one batch lab test was returned, refine your search parameters."
            )

        return {
            "id": id,
            "data": df_summary,
            "response": resp_summary,
            "exists": exists,
        }

    def geotechnicalsample_exists(
        self,
        projectsite=None,
        location=None,
        sampletype=None,
        campaign=None,
        sample=None,
        **kwargs
    ):
        """
        Checks if the geotechnical sample answering to the search criteria exists

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample

        :return: Returns the id if the geotechnical sample exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sampletype, sample],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sampletype",
                "sample",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/geotechnicalsample/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one geotechnical sample was returned, refine search criteria"
            )

        return record_id

    def get_sampletests(
        self,
        projectsite=None,
        campaign=None,
        location=None,
        sample=None,
        testtype=None,
        sampletest=None,
        **kwargs
    ):
        """
        Retrieve a summary of geotechnical sample lab tests corresponding to the specified search criteria

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
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sample, testtype, sampletest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sample",
                "testtype",
                "sampletest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/sampletestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_proximity_sampletests(self, latitude, longitude, radius, **kwargs):
        """
        Get all sample tests in a certain radius surrounding a point with given lat/lon
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
            **kwargs
        )

    def get_closest_sampletest(
        self, latitude, longitude, depth, initialradius=1, target_srid="25831", **kwargs
    ):
        """
        Get the sample test closest to a certain point
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param Depth: Depth of the central point in meters below seabed
        :param initialradius: Initial search radius around the central point in km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param **kwargs: Optional keyword arguments e.g. ``sample__location__title__icontains='BH'``
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
            initialradius=initialradius,
            target_srid=target_srid,
            **kwargs
        )

    def sampletesttype_exists(self, sampletesttype=None, **kwargs):
        """
        Checks if the sample test type answering to the search criteria exists

        :param sampletesttype: Title of the sample test type

        :return: Returns the id if the sample test type exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[
                sampletesttype,
            ],
            parameternames=[
                "testtype",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/sampletesttype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one sample test type was returned, refine search criteria"
            )

        return record_id

    def get_sampletesttypes(self, **kwargs):
        """
        Retrieve all sample tests types available in owimetadatabase

        :return: Dictionary with the following keys
            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified search criteria exist
        """
        url_params = {**kwargs}

        resp_summary = requests.get(
            url="%s/sampletesttype/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df_summary, "exists": exists}

    def get_sampletest_detail(
        self,
        projectsite=None,
        location=None,
        testtype=None,
        sample=None,
        campaign=None,
        sampletest=None,
        **kwargs
    ):
        """
        Retrieves detailed information on a specific sample test based on the specified search criteria

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
        :param sampletest: Title of the sample test

        :return: Dictionary with the following keys:
            - 'id': id for the selected soil profile
            - 'summary': Metadata for the batch lab test
            - 'response': Response text
            - 'rawdata': Dataframe with the raw data
            - 'processeddata': Dataframe with the raw data
            - 'conditions': Dataframe with test conditions
            - 'exists': Boolean indicating whether a matching record is found
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sample, testtype, sampletest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sample",
                "testtype",
                "sampletest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp_summary = requests.get(
            url="%s/sampletestsummary/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_summary = pd.DataFrame(json.loads(resp_summary.text))

        if df_summary.__len__() == 0:
            exists = False
            sample_test_id = None
        elif df_summary.__len__() == 1:
            exists = True
        else:
            raise ValueError(
                "More than one sample lab test was returned, refine your search parameters."
            )

        resp_detail = requests.get(
            url="%s/sampletestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df_resp_detail = pd.DataFrame(json.loads(resp_detail.text))
        try:
            sample_test_id = df_resp_detail["id"].iloc[0]
        except:
            sample_test_id = None

        try:
            df_raw = pd.DataFrame(df_resp_detail["rawdata"].iloc[0]).reset_index(
                drop=True
            )
        except Exception as err:
            try:
                df_raw = pd.DataFrame([df_resp_detail["rawdata"].iloc[0]]).reset_index(
                    drop=True
                )
            except:
                df_raw = pd.DataFrame()
        try:
            df_processed = pd.DataFrame(
                df_resp_detail["processeddata"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_processed = pd.DataFrame()
        try:
            df_conditions = pd.DataFrame(
                df_resp_detail["conditions"].iloc[0]
            ).reset_index(drop=True)
        except:
            df_conditions = pd.DataFrame()

        for _df in [df_raw, df_processed, df_conditions]:
            for col in _df.columns:
                try:
                    _df[col] = pd.to_numeric(_df[col], errors="ignore")
                except:
                    pass

        return {
            "id": sample_test_id,
            "summary": df_summary,
            "rawdata": df_raw,
            "processed": df_processed,
            "conditions": df_conditions,
            "response": resp_detail,
            "exists": exists,
        }

    def sampletest_exists(
        self,
        projectsite=None,
        location=None,
        testtype=None,
        sample=None,
        campaign=None,
        sampletest=None,
        **kwargs
    ):
        """
        Checks if the batch lab test answering to the search criteria exists

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
        :param sampletest: Title of the sample test

        :return: Returns the id if the sample test exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, campaign, location, sample, testtype, sampletest],
            parameternames=[
                "projectsite",
                "campaign",
                "location",
                "sample",
                "testtype",
                "sampletest",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/sampletestdetail/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one sample test was returned, refine search criteria"
            )

        return record_id

    def get_soilunit_depthranges(
        self, soilunit, projectsite=None, location=None, **kwargs
    ):
        """
        Retrieves the depth ranges for where the soil unit occurs

        :param soilunit: Title of the soil unit for which depth ranges need to be retrieved
        :param projectsite: Title of the project site (optional)
        :param location: Title of the test location (optional)

        :return: Returns the id if the sample test exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[soilunit, projectsite, location],
            parameternames=["soilunit", "projectsite", "location"],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/soillayer/" % self.api_root, headers=self.header, params=url_params
        )
        df = pd.DataFrame(json.loads(resp.text))

        return df

    def get_unit_insitutestdata(self, soilunit, depthcol="z [m]", **kwargs):
        """
        Retrieves proportions of in-situ test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with in-situ test data in the selected soil unit.
        """

        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = self.get_insitutests(**kwargs)["data"]
        all_unit_data = pd.DataFrame()
        for i, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()

            if row["location_name"] in selected_depths["location_name"].unique():
                _fulldata = self.get_insitutest_detail(
                    location=row["location_name"], **kwargs
                )["rawdata"]
                _depthranges = selected_depths[
                    selected_depths["location_name"] == row["location_name"]
                ]
                for j, _layer in _depthranges.iterrows():
                    _unitdata = _fulldata[
                        (_fulldata[depthcol] >= _layer["start_depth"])
                        & (_fulldata[depthcol] <= _layer["end_depth"])
                    ]
                    unitdata = pd.concat([unitdata, _unitdata])
                unitdata.reset_index(drop=True, inplace=True)
                unitdata.loc[:, "location_name"] = row["location_name"]
                unitdata.loc[:, "projectsite_name"] = row["projectsite_name"]
                unitdata.loc[:, "test_type_name"] = row["test_type_name"]
            else:
                pass
            all_unit_data = pd.concat([all_unit_data, unitdata])

        all_unit_data.reset_index(drop=True, inplace=True)

        return all_unit_data

    def get_unit_batchlabtestdata(self, soilunit, depthcol="z [m]", **kwargs):
        """
        Retrieves proportions of batch lab test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with batch lab test data in the selected soil unit.
        """

        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = self.get_batchlabtests(**kwargs)["data"]

        all_unit_data = pd.DataFrame()
        for i, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()

            if row["location_name"] in selected_depths["location_name"].unique():
                _fulldata = self.get_batchlabtest_detail(
                    location=row["location_name"], **kwargs
                )["rawdata"]
                _depthranges = selected_depths[
                    selected_depths["location_name"] == row["location_name"]
                ]
                for j, _layer in _depthranges.iterrows():
                    _unitdata = _fulldata[
                        (_fulldata[depthcol] >= _layer["start_depth"])
                        & (_fulldata[depthcol] <= _layer["end_depth"])
                    ]
                    unitdata = pd.concat([unitdata, _unitdata])
                unitdata.reset_index(drop=True, inplace=True)
                unitdata.loc[:, "location_name"] = row["location_name"]
                unitdata.loc[:, "projectsite_name"] = row["projectsite_name"]
                unitdata.loc[:, "test_type_name"] = row["test_type_name"]
            else:
                print("Soil unit not found for %s" % row["location_name"])
            all_unit_data = pd.concat([all_unit_data, unitdata])

        all_unit_data.reset_index(drop=True, inplace=True)

        return all_unit_data

    def get_unit_sampletests(self, soilunit, **kwargs):
        """
        Retrieves the sample tests data located inside a soil unit.
        The metadata of the samples is filtered based on the depth column.
        Further retrieval of the test data can follow after this method.

        :param soilunit: Name of the soil unit
        :param kwargs: Optional keyword arguments for retrieval of sample tests (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with sample test metadata in the selected soil unit.
        """
        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = self.get_sampletests(**kwargs)["data"]

        all_unit_data = pd.DataFrame()
        for i, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()

            if row["location_name"] in selected_depths["location_name"].unique():

                _depthranges = selected_depths[
                    selected_depths["location_name"] == row["location_name"]
                ]
                for j, _layer in _depthranges.iterrows():
                    if (
                        row["depth"] >= _layer["start_depth"]
                        and row["depth"] <= _layer["end_depth"]
                    ):
                        _unitdata = selected_tests[selected_tests["id"] == row["id"]]
                        unitdata = pd.concat([unitdata, _unitdata])
                    else:
                        pass

                unitdata.reset_index(drop=True, inplace=True)
            else:
                pass
            all_unit_data = pd.concat([all_unit_data, unitdata])

        all_unit_data.reset_index(drop=True, inplace=True)
        return all_unit_data
