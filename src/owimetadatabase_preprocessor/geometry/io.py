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

    def __init__(
        self,
        api_root: str = "https://owimetadatabase.owilab.be/api/v1",
        api_subdir: str = "/geometry/userroutes/",
        token: Union[str, None] = None,
        uname: Union[str, None] = None,
        password: Union[str, None] = None,
        **kwargs,
    ) -> None:
        """Create an instance of the GeometryAPI class with the required parameters.

        :param api_root: Optional: root URL of the API endpoint, the default working database url is provided.
        :param api_subdir: Optional: subdirectory of the API endpooint url for specific type of data.
        :param token: Optional: token to access the API.
        :param uname: Optional: username to access the API.
        :param password: Optional: password to access the API.
        :param kwargs: Additional parameters to pass to the API.
        :return: None
        """
        super().__init__(api_root, token, uname, password, **kwargs)
        if token:
            credentials = {"token": token}
        elif uname and password:
            credentials = {"uname": uname, "password": password}
        elif kwargs is not None:
            credentials = {}
        else:
            raise ValueError("No credentials provided.")
        self.loc_api = LocationsAPI(api_root=self.api_root, **credentials, **kwargs)
        self.api_root = self.api_root + api_subdir

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
        url_data_type = "subassemblies"
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
        """Get all building blocks for a given location.

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
        url_data_type = "buildingblocks"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_materials(self) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all the materials of building blocks.

        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}  # type: Dict[str, str]
        url_data_type = "materials"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_subassembly_objects(
        self, turbine: str, subassembly: str = None
    ) -> Dict[str, SubAssembly]:
        """Get all subassemblies for a given turbine, divided by type.

        :param turbine: Turbine title (e.g. 'BBC01')
        :param subassembly: Sub-assembly type (e.g. 'MP', 'TW', 'TP')
        :return: Dictionary with the following keys:

            - "TW": SubAssembly object for the tower
            - "TP": SubAssembly object for the transition piece
            - "MP": SubAssembly object for the monopile
        """
        url_data_type = "subassemblies"
        if subassembly is not None:
            url_params = {"asset__title": turbine, "subassembly_type": subassembly}
        else:
            url_params = {"asset__title": turbine}
        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)
        if not resp.json():
            raise ValueError("No subassemblies found for " + str(turbine))

        material_data = self.get_materials()
        if material_data["exists"]:
            materials = material_data["data"]
        else:
            raise ValueError("No materials found in the database.")

        sas = [SubAssembly(materials, item, api_object=self) for item in resp.json()]
        sas_types = [j["subassembly_type"] for j in resp.json()]
        subassemblies = {k: v for (k, v) in zip(sas_types, sas)}
        return subassemblies

    def get_owt_geometry_processor(
        self,
        turbines: Union[str, List[str]],
        tower_base: Union[float, List[float], None] = None,
        monopile_head: Union[float, List[float], None] = None,
    ) -> OWTs:
        """Return the required processing class.

        :param turbines: Title(s) of the turbines.
        :param tower_base: Optional: height(s) of the tower base.
        :param monopile_head: Optional: height(s) of the monopile head.
        :return: OWTs object: containing information about all the requested turbines.
        """
        materials_data = self.get_materials()
        if materials_data["exists"]:
            materials = materials_data["data"]
        else:
            raise ValueError("No materials found in the database.")
        owts = []
        turbines = [turbines] if isinstance(turbines, str) else turbines
        if not isinstance(tower_base, List) and not isinstance(monopile_head, List):
            tower_base = [tower_base] * len(turbines)  # type: ignore
            monopile_head = [monopile_head] * len(turbines)  # type: ignore
        for i in range(len(turbines)):
            subassemblies_data = self.get_subassemblies(assetlocation=turbines[i])
            location_data = self.loc_api.get_assetlocation_detail(
                assetlocation=turbines[i]
            )
            if subassemblies_data["exists"] and location_data["exists"]:
                subassemblies = subassemblies_data["data"]
                location = location_data["data"]
            elif not subassemblies_data["exists"] and not location_data["exists"]:
                raise ValueError(
                    f"No subassemblies and location found for turbine {turbines[i]}."
                )
            elif not subassemblies_data["exists"]:
                raise ValueError(f"No subassemblies found for turbine {turbines[i]}.")
            elif not location_data["exists"]:
                raise ValueError(f"No location found for turbine {turbines[i]}.")
            else:
                raise ValueError("Unexpected error.")
            owts.append(
                OWT(
                    self,
                    materials,
                    subassemblies,
                    location,
                    tower_base[i] if isinstance(tower_base, List) else tower_base,
                    (
                        monopile_head[i]
                        if isinstance(monopile_head, List)
                        else monopile_head
                    ),
                )
            )
        return OWTs(turbines, owts)

    def plot_turbines(
        self,
        turbines: Union[List[str], str],
        figures_per_line: int = 5,
        return_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """Plot turbines' frontal geometry.

        :param turbines: Title(s) of the turbines.
        :param figures_per_line: Number of figures per line.
        :param return_fig: Optional: whether to return the figure.
        :param show_fig: Optional: whether to show the figure.
        :return: Plotly figure object with selected turbines front profiles (if requested) or nothing.
        """
        materials_data = self.get_materials()
        if materials_data["exists"]:
            materials = materials_data["data"]
        else:
            raise ValueError("No materials found in the database.")
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
            subassemblies_data = self.get_subassemblies(assetlocation=turbine)
            if subassemblies_data["exists"]:
                subassemblies = subassemblies_data["data"]
            else:
                raise ValueError(f"No subassemblies found for turbine {turbine}.")
            for j, sa in subassemblies.iterrows():
                subassembly = SubAssembly(materials, sa.to_dict(), api_object=self)
                subassembly.building_blocks
                plotly_data = subassembly.plotly()
                for k in range(len(plotly_data[0])):
                    fig.add_trace(plotly_data[0][k], row=rows[i], col=cols[i])
            plotly_layout = plotly_data[1]
            if i > 0:
                plotly_layout["scene" + str(i + 1)] = plotly_layout["scene"]
                plotly_layout["yaxis" + str(i + 1)] = plotly_layout["yaxis"]
                plotly_layout["yaxis" + str(i + 1)]["scaleanchor"] = "x" + str(i + 1)
                plotly_layout.pop("scene")
                plotly_layout.pop("yaxis")
                plotly_layout["yaxis" + str(i + 1)].pop("title")
            fig.update_layout(plotly_layout, autosize=autosize)
        if return_fig and show_fig:
            fig.show()
            return fig
        elif show_fig:
            fig.show()
        elif return_fig:
            return fig
