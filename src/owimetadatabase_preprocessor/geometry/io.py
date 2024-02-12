"""Module to connect to the database API to retrieve and operate on geometry data."""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from owimetadatabase_preprocessor.geometry.processing import OWT, OWTs
from owimetadatabase_preprocessor.geometry.structures import SubAssembly
from owimetadatabase_preprocessor.io import API
from owimetadatabase_preprocessor.locations.io import LocationsAPI


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
        self,
        turbines: Union[str, List[str]],
        tower_base: Union[float, List[float], None] = None,
        monopile_head: Union[float, List[float], None] = None,
    ) -> OWTs:
        """Return the required processing class."""
        materials = self.get_materials()["data"]
        owts = []
        turbines = [turbines] if isinstance(turbines, str) else turbines
        if not isinstance(tower_base, List) and not isinstance(monopile_head, List):
            tower_base = [tower_base] * len(turbines)
            monopile_head = [monopile_head] * len(turbines)
        for i in range(len(turbines)):
            subassemblies = self.get_subassemblies(assetlocation=turbines[i])["data"]
            location = LocationsAPI(header=self.header).get_assetlocation_detail(
                assetlocation=turbines[i]
            )["data"]
            owts.append(
                OWT(
                    self,
                    materials,
                    subassemblies,
                    location,
                    tower_base[i],
                    monopile_head[i],
                )
            )
        return OWTs(turbines, owts)

    def plot_turbines(
        self, turbines: Union[List[str], str], figures_per_line: int = 5
    ) -> None:
        """Plot turbines' frontal geometry."""
        materials = self.get_materials()["data"]
        turbines = [turbines] if isinstance(turbines, str) else turbines
        if len(turbines) > figures_per_line:
            n_rows = len(turbines) // figures_per_line + 1
            n_cols = figures_per_line
            rows = [i for i in range(1, n_rows + 1) for _ in range(n_cols)]
            cols = [i for _ in range(n_rows) for i in range(1, n_cols + 1)]
        else:
            n_rows = 1
            n_cols = len(turbines)
            rows = [1 for _ in range(n_cols)]
            cols = [i for i in range(1, n_cols + 1)]
        autosize = False if len(turbines) < 3 else True
        fig = make_subplots(n_rows, n_cols, subplot_titles=turbines)
        for i, turbine in enumerate(turbines):
            subassemblies = self.get_subassemblies(assetlocation=turbine)["data"]
            for j, sa in subassemblies.iterrows():
                subassembly = SubAssembly(materials, sa.to_dict(), api_object=self)
                subassembly.building_blocks
                plotly_data = subassembly.plotly()
                fig.add_trace(plotly_data[0], row=rows[i], col=cols[i])
            plotly_layout = plotly_data[1]
            if i > 0:
                plotly_layout["scene" + str(i + 1)] = plotly_layout["scene"]
                plotly_layout["yaxis" + str(i + 1)] = plotly_layout["yaxis"]
                plotly_layout["yaxis" + str(i + 1)]["scaleanchor"] = "x" + str(i + 1)
                plotly_layout.pop("scene")
                plotly_layout.pop("yaxis")
                plotly_layout["yaxis" + str(j + 1)].pop("title")
            fig.update_layout(plotly_layout, autosize=autosize)
        fig.show()
