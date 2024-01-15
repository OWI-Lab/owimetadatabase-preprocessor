"""Module to connect to the database API to retrieve and operate on geometry data."""

from typing import Dict, Union

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.io import API


class GeometryAPI(API):
    """Class to connect to the geometry data API with methods to retrieve data."""

    def get_subassemblies(
        self,
        projectsite: Union[str, None] = None,
        assetlocation: Union[str, None] = None,
        subassembly_type: Union[str, None] = None,
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all structure subassemblies blocks for a given location.

        :param projectsite: Title of the projectsite.
        :param assetlocation: Title of the asset location.
        :param subassembly_type: Type of the subassembly.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}
        if projectsite is not None:
            url_params["asset__projectsite__title"] = projectsite
        if assetlocation is not None:
            url_params["asset__title"] = assetlocation
        if subassembly_type is not None:
            url_params["subassembly_type"] = subassembly_type
        url_data_type = "/geometry/userroutes/subassemblies"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_buildingblocks(
        self,
        projectsite: Union[str, None] = None,
        assetlocation: Union[str, None] = None,
        subassembly_type: Union[str, None] = None,
        subassembly_id: Union[str, None] = None,
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """
        Get all building blocks for a given location.

        :param projectsite: Title of the projectsite.
        :param assetlocation: Title of the asset location.
        :param subassembly_type: Type of the subassemblies.
        :param subassembly_id: ID of the subassembly.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}
        if projectsite is not None:
            url_params["sub_assembly__asset__projectsite__title"] = projectsite
        if assetlocation is not None:
            url_params["sub_assembly__asset__title"] = assetlocation
        if subassembly_type is not None:
            url_params["sub_assembly__subassembly_type"] = subassembly_type
        if subassembly_id is not None:
            url_params["sub_assembly__id"] = str(subassembly_id)
        url_data_type = "/geometry/userroutes/buildingblocks"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_materials(self) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """
        Get all the materials of building blocks.

        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}  # type: Dict[str, str]
        url_data_type = "/geometry/userroutes/materials"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_owt_geometry_processor(
        self, turbine: str, tower_base: float, monopile_head: float
    ) -> OWT:
        """Return the required processing class."""
        materials = self.get_materials()["data"]
        subassemblies = self.get_subassemblies(assetlocation=turbine)["data"]
        return OWT(self, materials, subassemblies, tower_base, monopile_head)
