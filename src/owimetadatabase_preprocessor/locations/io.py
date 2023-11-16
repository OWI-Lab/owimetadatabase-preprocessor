import json

import pandas as pd
import plotly.express as px  # type: ignore
import requests

from owimetadatabase_preprocessor.io import API


class LocationsAPI(API):
    """
    Class to connect to the location data API with methods to retrieve data

    A number of methods are provided to query the database via the owimetadatabase API.
    In the majority of cases, the methods return a dataframe based on the URL parameters provided.
    The methods are written such that a number of mandatory URL parameters are required (see documentation of the methods).
    The URL parameters can be expanded with Django-style additional filtering arguments (e.g.
    ``location__title__icontains="BB"``) as optional keyword arguments. Knowledge of the Django models is required for this
    (see ``owimetadatabase`` code).
    """

    @staticmethod
    def urlparameters(parameters, parameternames):
        """
        Returns a dictionary with URL parameters based on lists of parameters and parameter names

        :param parameters: List with parameters
        :param parameternames: List with parameter names
        :return: Dictionary with the URL parameters
        """
        url_params = {}

        for param, paramname in zip(parameters, parameternames):
            url_params[paramname] = param

        return url_params

    def get_projectsites(self, **kwargs):
        """
        Get all available projects

        :return:  Dictionary with the following keys:
            - 'data': Pandas dataframe with the location data for each project
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {}

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/locations/projectsites/" % self.api_root,
            headers=self.header,
            params=url_params,
        )

        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df, "exists": exists}

    def get_projectsite_detail(self, projectsite, **kwargs):
        """
        Get details for a specific projectsite

        :param projectsite: Title of the projectsite
        :return:  Dictionary with the following keys:
            - 'id': id of the selected project site
            - 'data': Pandas dataframe with the location data for each projectsite
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = self.urlparameters(
            parameters=[
                projectsite,
            ],
            parameternames=[
                "projectsite",
            ],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/locations/projectsites/" % self.api_root,
            headers=self.header,
            params=url_params,
        )

        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            exists = False
            project_id = None
        elif df.__len__() == 1:
            exists = True
            project_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one project site was returned, check search criteria."
            )

        return {"id": project_id, "data": df, "exists": exists}

    def assetlocation_exists(self, projectsite=None, location=None, **kwargs):
        """
        Checks if the asset location answering to the search criteria exists

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the asset location (e.g. "BBK05")
        :return: Returns the asset location id if the asset location exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, location],
            parameternames=["projectsite", "assetlocation"],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/locations/assetlocations/" % self.api_root,
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
                "More than one asset location was returned, refine search criteria"
            )

        return record_id

    def assetlocation_location_exists(self, projectsite=None, location=None, **kwargs):
        """
        Checks if the location answering to the search criteria for the asset location exists
        If the asset location id is required, run the method ``assetlocation_exists`` instead.

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param location: Name of the asset location (e.g. "BBK05")
        :return: Returns the location id if the asset location exists, False otherwise
        """
        url_params = self.urlparameters(
            parameters=[projectsite, location],
            parameternames=["projectsite", "assetlocation"],
        )

        url_params = {**url_params, **kwargs}

        resp = requests.get(
            url="%s/locations/assetlocations/" % self.api_root,
            headers=self.header,
            params=url_params,
        )
        df = pd.DataFrame(json.loads(resp.text))

        if df.__len__() == 0:
            record_id = False
        elif df.__len__() == 1:
            record_id = df["location"].iloc[0]
        else:
            raise ValueError(
                "More than one asset location was returned, refine search criteria"
            )

        return record_id

    def get_assetlocations(self, projectsite=None, assetlocation=None):
        """
        Get all available asset locations, specify a projectsite or filter by projectsite

        :param projectsite: String with the projectsite title (e.g. "Nobelwind")
        :param assetlocation: String with the asset location title (e.g. "NW2A04")
        :return: Dictionary with the following keys:
            - 'data': Pandas dataframe with the location data for each location in the projectsite
            - 'exists': Boolean indicating whether matching records are found
        """
        url_params = {}

        if projectsite is not None:
            url_params["projectsite__title"] = projectsite
        elif assetlocation is not None and projectsite is None:
            url_params["title"] = assetlocation

        if self.header is not None:
            resp_assetlocation = requests.get(
                url="%s/locations/assetlocations/" % self.api_root,
                headers=self.header,
                params=url_params,
            )
        else:
            if self.uname is None or self.password is None:
                e = [
                    "Either self.header or self.uname and ",
                    "self.password must be defined.",
                ]
                raise ValueError("".join(e))
            else:
                resp_assetlocation = requests.get(
                    url="%s/locations/assetlocations/" % self.api_root,
                    auth=self.auth,
                    params=url_params,
                )

        df = pd.DataFrame(json.loads(resp_assetlocation.text))

        if df.__len__() == 0:
            exists = False
        else:
            exists = True

        return {"data": df, "exists": exists}

    def get_assetlocation_detail(self, projectsite, assetlocation):
        """
        Get a selected turbine

        :param projectsite: Name of the projectsite (e.g. "Nobelwind")
        :param assetlocation: Title of the asset location (e.g. "BBK05")
        :return: Dictionary with the following keys:
            - 'id': id of the selected projectsite site
            - 'data': Pandas dataframe with the location data for the individual location
            - 'exists': Boolean indicating whether a matching location is found
        """
        if self.header is not None:
            resp_assetlocation = requests.get(
                url="%s/locations/assetlocations/" % self.api_root,
                headers=self.header,
                params=dict(projectsite=projectsite, assetlocation=assetlocation),
            )
        else:
            if self.uname is None or self.password is None:
                e = [
                    "Either self.header or self.uname and ",
                    "self.password must be defined.",
                ]
                raise ValueError("".join(e))
            else:
                resp_assetlocation = requests.get(
                    url="%s/locations/assetlocations/" % self.api_root,
                    auth=self.auth,
                    params=dict(projectsite=projectsite, assetlocation=assetlocation),
                )

        if resp_assetlocation.status_code != 200:
            e = [
                "Error ",
                resp_assetlocation.status_code,
                ".\n",
                resp_assetlocation.reason,
            ]
            raise Exception("".join(e))

        if not resp_assetlocation.json():
            raise ValueError("No asset locations found. Check request criteria.")

        df = pd.DataFrame(json.loads(resp_assetlocation.text))

        if df.__len__() == 0:
            exists = False
            asset_id = None
        elif df.__len__() == 1:
            exists = True
            asset_id = df["id"].iloc[0]
        else:
            raise ValueError(
                "More than one asset location was returned, check search criteria."
            )

        return {"id": asset_id, "data": df, "exists": exists}

    def plot_assetlocations(self, return_fig=False, **kwargs):
        """
        Retrieves asset locations and generates a Plotly plot to show them

        :param return_fig: Boolean indicating whether the Plotly figure object needs to be returned
        (default is False which simply shows the plot)
        :param kwargs: Keyword arguments for the search (see ``get_assetlocations``)
        :return: Plotly figure object with selected asset locations plotted on OpenStreetMap tiles (if requested)
        """
        assetlocations = self.get_assetlocations(**kwargs)["data"]
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
