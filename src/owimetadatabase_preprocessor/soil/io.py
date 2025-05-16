"""
API client Module for the soil data in the OWIMetadatabase.
"""

import warnings
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
import requests

from owimetadatabase_preprocessor.io import API
from owimetadatabase_preprocessor.soil.processing import SoilDataProcessor


class SoilAPI(API):
    """
    API client to handle HTTP communication for soil data.
    Inherits common functionality from the API class.
    """

    def __init__(
        self,
        api_subdir: str = "/soildata/",
        **kwargs,
    ) -> None:
        """
        Constructor for the SoilAPI.

        :param api_root: Base URL for the API.
        :param api_subdir: Sub-directory for soil data endpoints.
        :param token: API token (if required). A Bearer token will be used.
        :param uname: Username for authentication.
        :param password: Password for authentication.
        """
        super().__init__(**kwargs)
        self.api_root = self.api_root + api_subdir

    def get_proximity_entities_2d(
        self, api_url: str, latitude: float, longitude: float, radius: float, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """Find the entities in a certain radius around a point in 2D (cylindrical search area).

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km
        :param kwargs: Optional keyword arguments for the search
        :return: Dictionary with the following keys:

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
        """
        Search for any entity in a certain radius around a point in 2D
        (cylindrical search area).

        :param api_url: End-point for the API
        :param radius_init: Initial search radius around the central point in km
        :param url_params: Dictionary with the URL parameters for the endpoint
        :param radius_max: Maximum search radius around the central point in km
        :return: Pandas dataframe with the data according to the specified search
            criteria
        """
        radius = radius_init
        while True:
            url_params["offset"] = str(radius)
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

    def get_closest_entity_2d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        radius_init: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """
        Get the entity closest to a certain point in 2D with optional query
        arguments (cylindrical search area).

        :param api_url: End-point for the API
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius_init: Initial search radius around the central point in
            km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g.
            ``campaign__projectsite__title__icontains='HKN'``
        :return:  Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each
                location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        geosearch_params = dict(latitude=latitude, longitude=longitude)
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = SoilDataProcessor.transform_coord(
            df, longitude, latitude, target_srid
        )
        df["offset [m]"] = np.sqrt(
            (df["easting [m]"] - point_east) ** 2
            + (df["northing [m]"] - point_north) ** 2
        )
        return SoilDataProcessor.gather_data_entity(df)

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
        """
        Get the entity closest to a certain point in 3D (spherical search area)
        with optional query arguments.

        :param api_url: End-point for the API
        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth: of the central point in meters below seabed
        :param radius_init: Initial search radius around the central point in
            km, the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param sampletest: Boolean indicating whether a sample or sample test
            needs to be retrieved (default is True to search for sample tests)
        :param kwargs: Optional keyword arguments e.g.
            ``campaign__projectsite__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each
                location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        geosearch_params = dict(latitude=latitude, longitude=longitude)
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = SoilDataProcessor.transform_coord(
            df, longitude, latitude, target_srid
        )
        if not sampletest:
            df["depth"] = 0.5 * (df["top_depth"] + df["bottom_depth"])
        df["offset [m]"] = np.sqrt(
            (df["easting [m]"] - point_east) ** 2
            + (df["northing [m]"] - point_north) ** 2
            + (df["depth"] - depth) ** 2
        )
        return SoilDataProcessor.gather_data_entity(df)

    def get_surveycampaigns(
        self, projectsite: Union[str, None] = None, **kwargs
    ) -> Dict[str, Union[pd.DataFrame, bool, None]]:
        """
        Get all available survey campaigns, specify a projectsite to filter by
        projectsite.

        :param projectsite: String with the projectsite title (e.g. "Nobelwind")
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the location data for each location
                in the projectsite
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
        """
        Get details for a specific survey campaign.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param campaign: Title of the survey campaign (e.g. "Borehole campaign")
        :return: Dictionary with the following keys:

            - 'id': id of the selected projectsite site
            - 'data': Pandas dataframe with the location data for the individual
                location
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
        """
        Get all soil test locations in a certain radius surrounding a point
        with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each
                location in the specified search area
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
        """
        Get the soil test location closest to a certain point with the name
        containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each
                location in the specified search area
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
        """
        Get the geotechnical test locations corresponding to the given search
        criteria.

        :param projectsite: Name of the projectsite under consideration
            (e.g. "Nobelwind", optional, default is None)
        :param campaign: Name of the survey campaign (optional, default is None
            to return all locations in a projectsite)
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each
                location meeting the specified search criteria
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
        :param projectsite: Optional, name of the projectsite under
            consideration (e.g. "Nobelwind")
        :param campaign: Optional, name of the survey campaign (e.g. "Borehole
            campaign")
        :return: Dictionary with the following keys:

            - 'id': id of the selected test location
            - 'data': Pandas dataframe with the test location data for each
                location meeting the specified search criteria
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
        """
        Checks if the test location answering to the search criteria exists.

        :param location: Name of a specific location (e.g. "CPT-888")
        :param projectsite: Optional, name of the projectsite under
            consideration (e.g. "Nobelwind")
        :param campaign: Optional, name of the survey campaign (e.g.
            "Borehole campaign")
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
        """
        Checks if the in-situ test type answering to the search criteria exists
        and returns the id.

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
        """
        Get the detailed information (measurement data) for an in-situ test of
        given type.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param insitutest: Name of the in-situ test
        :return: Dictionary with the following keys:

            - 'data': Metadata of the insitu tests
            - 'exists': Boolean indicating whether a matching in-situ test is
                found
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
        """Get all in-situ tests in a certain radius surrounding a point with
        given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the in-situ test summary data for
                each in-situ test in the specified search area
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
        """
        Get the in-situ test closest to a certain point with the name containing
        a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``campaign__projectsite__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the in-situ test data for each
                in-situ test in the specified search area
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

    def get_closest_insitutest_byname(
        self,
        projectsite: str,
        location: str,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """
        Get the in-situ test closest to a location specified by name.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments for filtering tests
        :return: Dictionary with the following keys:
            - 'data': Pandas dataframe with the in-situ test data
            - 'id': ID of the closest in-situ test
            - 'title': Title of the closest in-situ test
            - 'offset [m]': Offset in meters from the location
        :raises ValueError: If the location does not exist
        """
        # First verify the location exists
        location_id = self.testlocation_exists(
            projectsite=projectsite, location=location
        )

        if not location_id:
            raise ValueError(
                f"Location '{location}' not found in project '{projectsite}'"
            )

        # Get location details to obtain coordinates
        location_details = self.get_testlocation_detail(
            projectsite=projectsite, location=location
        )

        # Extract coordinates from location data
        location_data = location_details["data"]
        latitude = location_data["latitude"].iloc[0]
        longitude = location_data["longitude"].iloc[0]

        # Use existing method to find closest in-situ test
        return self.get_closest_insitutest(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_closest_soilprofile_byname(
        self,
        projectsite: str,
        location: str,
        radius: float = 1.0,
        target_srid: str = "25831",
        retrieve_details: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """
        Get the soil profile closest to a location specified by name.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param retrieve_details: Boolean determining whether the soil profile detail
            needs to be retrieved. Default is true in which case the result of
            get_soilprofile_detail is returned
        :param verbose: Boolean determining whether to print info about found profile
        :param kwargs: Optional keyword arguments for filtering profiles
        :return: Dictionary with the following keys:
            - 'data': Pandas dataframe with the soil profile data
            - 'id': ID of the closest soil profile
            - 'title': Title of the closest soil profile
            - 'offset [m]': Offset in meters from the location
            If retrieve_details is True, returns the full soil profile details instead
        :raises ValueError: If the location does not exist
        """
        # First verify the location exists
        location_id = self.testlocation_exists(
            projectsite=projectsite, location=location
        )

        if not location_id:
            raise ValueError(
                f"Location '{location}' not found in project '{projectsite}'"
            )

        # Get location details to obtain coordinates
        location_details = self.get_testlocation_detail(
            projectsite=projectsite, location=location
        )

        # Extract coordinates from location data
        location_data = location_details["data"]
        latitude = location_data["latitude"].iloc[0]
        longitude = location_data["longitude"].iloc[0]

        # Use existing method to find closest soil profile
        closest_profile = self.get_closest_soilprofile(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            target_srid=target_srid,
            **kwargs,
        )

        if verbose:
            print(
                f"Soil profile {closest_profile['title']} found at {closest_profile['offset [m]']:.1f}m offset"
            )

        if retrieve_details:
            return self.get_soilprofile_detail(
                projectsite=projectsite,
                location=location,
                soilprofile=closest_profile["title"],
                **kwargs,
            )

        return closest_profile

    def get_insitutest_detail(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        combine: bool = False,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, int, bool, requests.Response, None]]:
        """
        Get the detailed information (measurement data) for an in-situ test of
        give type.

        :param insitutest: Name of the in-situ test
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param combine: Boolean indicating whether raw and processed data needs
            to be combined (default=False). If true, processed data columns are
            appended to the rawdata dataframe
        :param kwargs: Optional keyword arguments for further queryset filtering
            based on model attributes.
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
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = SoilDataProcessor.combine_dfs(dfs)
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
        """
        Get the detailed information (measurement data) for an in-situ test of
        CPT type (seabed or downhole CPT)

        :param insitutest: Name of the in-situ test
        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param testtype: Name of the test type (e.g. "PCPT")
        :param combine: Boolean indicating whether raw and processed data needs
            to be combined (default=False).
            If true, processed data columns are appended to the rawdata dataframe
        :param cpt: Boolean determining whether the in-situ test is a CPT or not.
            If True (default), a PCPTProcessing object is returned.
        :param kwargs: Optional keyword arguments for the cpt data loading.
            Note that further queryset filtering based on model attributes is
            not possible with this method. The in-situ test needs to be fully
            defined by the required arguments.
        :return: Dictionary with the following keys:

            - 'id': id of the selected test
            - 'insitutestsummary': Metadata of the insitu tests
            - 'rawdata': Raw data
            - 'processed': Processed data
            - 'conditions': Test conditions
            - 'response': Response text
            - 'cpt': PCPTProcessing object (only if the CPT data is successfully
                loaded)
            - 'exists': Boolean indicating whether a matching in-situ test is
                found
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
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        if combine:
            df_raw = SoilDataProcessor.combine_dfs(dfs)
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
            cpt_ = SoilDataProcessor.process_cpt(df_sum, df_raw, **kwargs)
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
            - 'exists': Boolean indicating whether a matching in-situ test is
                found
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
        """
        Get all soil profiles in a certain radius surrounding a point with
        given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the soil profile summary data for
                each soil profile in the specified search area
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
        """Get the soil profile closest to a certain point with additional
        conditions as optional keyword arguments.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='HKN'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the soil profile data for each soil
                profile in the specified search area
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
        """
        Retrieves a soil profile from the owimetadatabase and converts it to a
        groundhog SoilProfile object.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the test location (e.g. "CPT-7C")
        :param soilprofile: Title of the soil profile (e.g. "Borehole log")
        :param convert_to_profile: Boolean determining whether the soil profile
            needs to be converted to a groundhog SoilProfile object
        :param drop_info_cols: Boolean determining whether or not to drop the
            columns with additional info (e.g. soil description, ...)
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
            dsp = SoilDataProcessor.convert_to_profile(
                df_sum, df_detail, profile_title, drop_info_cols
            )
            dict_["soilprofile"] = dsp
            return dict_
        return dict_

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
        """
        Retrieves a summary of batch lab tests corresponding to the specified
        search criteria.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param testtype: Title of the test type
        :param batchlabtest: Title of the batch lab test
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the specified
                search criteria exist
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
        """
        Checks if the geotechnical sample type answering to the search criteria
        exists.

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
        """
        Gets all batch lab tests in a certain radius surrounding a point with
        given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the batch lab test summary data
                for each batch lab test in the specified search area
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
        """
        Gets the batch lab test closest to a certain point with the name
        containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the batch lab test data for each
                batch lab test in the specified search area
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
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
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
        """
        Checks if the geotechnical sample type answering to the search criteria
        exists.

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
        """
        Retrieves geotechnical samples corresponding to the specified search
        criteria.

        :param projectsite: Project site name (e.g. 'Nobelwind')
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sampletype: Title of the sample type
        :param sample: Title of the sample
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the sample
            - 'exists': Boolean indicating whether records meeting the specified
                search criteria exist
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
        """
        Gets all geotechnical samples in a certain radius surrounding a point
        with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for
                each geotechnical sample in the specified search area
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
        """
        Gets the geotechnical sample closest to a certain point with the name
        containing a certain string.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param depth: Depth of the central point in meters below seabed
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the geotechnical sample data for
                each geotechnical sample in the specified search area
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
        """
        Checks if the geotechnical sample answering to the search criteria exists.

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
        """
        Retrieves a summary of geotechnical sample lab tests corresponding to
        the specified search criteria.

        :param projectsite: Title of the project site
        :param campaign: Title of the survey campaign
        :param location: Title of the test location
        :param sample: Title of the sample
        :param testtype: Title of the test type
        :param sampletest: Title of the sample test
        :return: Dictionary with the following keys

            - 'data': Dataframe with details on the batch lab test
            - 'exists': Boolean indicating whether records meeting the
                specified search criteria exist
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
        """
        Gets all sample tests in a certain radius surrounding a point with given lat/lon.

        :param latitude: Latitude of the central point in decimal format
        :param longitude: Longitude of the central point in decimal format
        :param radius: Radius around the central point in km
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the sample test summary data for each
                sample test in the specified search area
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
        :param radius: Initial search radius around the central point in km,
            the search radius is increased until locations are found
        :param target_srid: SRID for the offset calculation in meters
        :param kwargs: Optional keyword arguments e.g. ``sample__location__title__icontains='BH'``
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the sample test data for each sample
                test in the specified search area
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
            - 'exists': Boolean indicating whether records meeting the
                specified search criteria exist
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
        """
        Retrieves detailed information on a specific sample test based on the
        specified search criteria.

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
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
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
        """
        Checks if the batch lab test answering to the search criteria exists.

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
        """
        Retrieves the depth ranges for where the soil unit occurs.

        :param soilunit: Title of the soil unit for which depth ranges need to
            be retrieved
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

    def get_unit_insitutestdata(
        self, soilunit: str, depthcol: Union[str, None] = "z [m]", **kwargs
    ) -> pd.DataFrame:
        """
        Retrieves proportions of in-situ test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests
            (e.g. ``projectsite`` and ``testtype``)
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
        """
        Retrieves proportions of batch lab test data located inside a soil unit.
        The data in the ``rawdata`` field is filtered based on the depth column.

        :param soilunit: Name of the soil unit
        :param depthcol: Name of the column with the depth in the ``rawdata`` field
        :param kwargs: Optional keyword arguments for retrieval of in-situ tests
            (e.g. ``projectsite`` and ``testtype``)
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
        """
        Retrieves the sample tests data located inside a soil unit.
        The metadata of the samples is filtered based on the depth column.
        Further retrieval of the test data can follow after this method.

        :param soilunit: Name of the soil unit
        :param kwargs: Optional keyword arguments for retrieval of sample tests
            (e.g. ``projectsite`` and ``testtype``)
        :return: Dataframe with sample test metadata in the selected soil unit.
        """
        return self._process_data_units(soilunit, self.get_sampletests, **kwargs)

    def get_soilprofile_profile(
        self, lat1: float, lon1: float, lat2: float, lon2: float, band: float = 1000
    ) -> pd.DataFrame:
        """
        Retrieves soil profiles along a profile line.

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

    def _process_data_units(
        self,
        soilunit: str,
        func_get: Callable,
        func_get_details: Union[Callable, None] = None,
        depthcol: Union[str, None] = None,
        full: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        # TODO: Add docstring
        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = func_get(**kwargs)["data"]
        all_unit_data = pd.DataFrame()
        for _, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()
            if row["location_name"] in selected_depths["location_name"].unique():
                if full:
                    unitdata = SoilDataProcessor.fulldata_processing(
                        unitdata,
                        row,
                        selected_depths,
                        func_get_details,
                        depthcol,
                        **kwargs,
                    )
                else:
                    unitdata = SoilDataProcessor.partialdata_processing(
                        unitdata, row, selected_depths, selected_tests
                    )
            else:
                print(f"Soil unit not found for {row['location_name']}")
            all_unit_data = pd.concat([all_unit_data, unitdata])
        all_unit_data.reset_index(drop=True, inplace=True)
        return all_unit_data
