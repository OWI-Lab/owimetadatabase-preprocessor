"""Module to connect to the database API to retrieve and operate on geometry data."""

import warnings
from contextlib import contextmanager
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly as plt
from plotly.subplots import make_subplots

from owimetadatabase_preprocessor.geometry.processing import OWT, OWTs
from owimetadatabase_preprocessor.geometry.structures import SubAssembly
from owimetadatabase_preprocessor.io import API
from owimetadatabase_preprocessor.locations.io import LocationsAPI


class GeometryAPI(API):
    """Class to connect to the geometry data API with methods to retrieve data."""

    def __init__(
        self,
        api_subdir: str = "/geometry/userroutes/",
        **kwargs,
    ) -> None:
        """Create an instance of the GeometryAPI class with the required parameters.

        :param api_subdir: Optional: subdirectory of the API endpooint url for specific type of data.
        :param kwargs: Additional parameters to pass to the API (see the base class).
        :return: None
        """
        super().__init__(**kwargs)
        self.loc_api = LocationsAPI(**kwargs)
        self.api_root = self.api_root + api_subdir

    @contextmanager
    def _temp_api_root(self, new_api_root: str):
        """Temporarily change the api_root."""
        original_root = self.api_root
        self.api_root = new_api_root
        try:
            yield
        finally:
            self.api_root = original_root

    def get_model_definitions(
        self,
        projectsite: Union[str, None] = None,
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all relevant model definitions.

        :param projectsite: Optional: Title of the projectsite.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the model definitions
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}
        if projectsite is not None:
            url_params["site"] = projectsite
        url_data_type = "modeldefinitions"
        output_type = "list"
        with self._temp_api_root(self.api_root.replace("userroutes", "routes")):
            df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_modeldefinition_id(
        self,
        assetlocation: Union[str, None] = None,
        projectsite: Union[str, None] = None,
        model_definition: Union[str, None] = None,
    ) -> Dict[str, Union[int, np.int64, bool, None]]:
        """Get the ID of a model definition.
        Either the asset location or the project site must be specified.

        :param assetlocation: Optional: Title of the asset location.
        :param projectsite: Optional: Title of the projectsite.
        :param model_definition: Optional: Title of the model definition.
        :return: Dictionary with the following keys:

            - "id": ID of the specified model definition
            - "multiple_modeldef": Boolean indicating whether there are multiple model definitions
                                   for the asset location in general
        """
        if assetlocation is None and projectsite is None:
            raise ValueError(
                "At least either of the related `assetlocation` or `projectsite` must be specified!"
            )

        result = {}
        if projectsite is None:
            location_data = self.loc_api.get_assetlocation_detail(
                assetlocation=assetlocation
            )
            if location_data["exists"]:
                location = location_data["data"]
            else:
                raise ValueError(f"No location found for asset {assetlocation}.")
            projectsite = location["projectsite_name"].loc[0]
        model_definitions_data = self.get_model_definitions(projectsite=projectsite)
        if model_definitions_data["exists"]:
            model_definitions = model_definitions_data["data"]
        else:
            raise ValueError(
                f"No model definitions found for project site {projectsite}."
            )
        if model_definition is None and len(model_definitions) > 1:
            raise ValueError(
                f"Multiple model definitions found for project site {projectsite}. Please specify which one to use."
            )
        if model_definition is None:
            model_definition_id = model_definitions["id"].values[0]
            result["id"] = model_definition_id
            result["multiple_modeldef"] = False
        else:
            matching_definitions = model_definitions[
                model_definitions["title"] == model_definition
            ]
            if matching_definitions.empty:
                raise ValueError(
                    f"Model definition '{model_definition}' not found for project site {projectsite}."
                )
            if len(matching_definitions) > 1:
                raise ValueError(
                    f"Multiple model definitions found for '{model_definition}' in project site {projectsite}.\n"
                    f"Please check the data consistency."
                )
            model_definition_id = matching_definitions["id"].values[0]
            result["id"] = model_definition_id
            result["multiple_modeldef"] = len(model_definitions) > 1
        return result

    def get_subassemblies(
        self,
        projectsite: Union[str, None] = None,
        assetlocation: Union[str, None] = None,
        subassembly_type: Union[str, None] = None,
        model_definition: Union[str, None] = None,
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all relevant structure subassemblies.
        If you specify a model definition, you also must specify either the projectsite or the asset location.

        :param projectsite: Optional: Title of the projectsite.
        :param assetlocation: Optional: Title of the asset location.
        :param subassembly_type: Optional: Type of the subassembly.
        :param model_definition: Optional: Title of the model definition.
        :return: Dictionary with the following keys:

            - "data": Pandas dataframe with the location data for each project
            - "exists": Boolean indicating whether matching records are found
        """
        url_params = {}
        func_args = {}
        if projectsite is not None:
            url_params["asset__projectsite__title"] = projectsite
            func_args["projectsite"] = projectsite
        if assetlocation is not None:
            url_params["asset__title"] = assetlocation
            func_args["assetlocation"] = assetlocation
        if subassembly_type is not None:
            url_params["subassembly_type"] = subassembly_type
        if model_definition is not None:
            if projectsite is not None or assetlocation is not None:
                func_args["model_definition"] = model_definition
                modeldef_data = self.get_modeldefinition_id(**func_args)
                if modeldef_data["id"] is not None:
                    url_params["model_definition"] = str(modeldef_data["id"])
                else:
                    raise ValueError(
                        f"No model definition {model_definition} found for project site {projectsite} "
                        f"or asset location {assetlocation}."
                    )
            else:
                raise ValueError(
                    "If you specify a model definition, you also must specify either the projectsite or the asset location!"
                )
        url_data_type = "subassemblies"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_buildingblocks(
        self,
        projectsite: Union[str, None] = None,
        assetlocation: Union[str, None] = None,
        subassembly_type: Union[str, None] = None,
        subassembly_id: Union[int, np.int64, None] = None,
    ) -> Dict[str, Union[pd.DataFrame, bool, np.int64, None]]:
        """Get all relevant building blocks.

        :param projectsite: Optional: Title of the projectsite.
        :param assetlocation: Optional: Title of the asset location.
        :param subassembly_type: Optional: Type of the subassemblies (e.g. 'MP', 'TW', 'TP').
        :param subassembly_id: Optional: ID of the subassembly.
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
        self,
        turbine: str,
        subassembly: Union[str, None] = None,
        model_definition_id: Union[int, np.int64, None] = None,
    ) -> Dict[str, SubAssembly]:
        """Get all subassemblies for a given turbine, divided by type.

        :param turbine: Turbine title
        :param subassembly: Sub-assembly type (e.g. 'MP', 'TW', 'TP')
        :param model_definition_id: ID of the model definition to filter the subassemblies.
        :return: Dictionary with the following keys:

            - "TW": SubAssembly object for the tower
            - "TP": SubAssembly object for the transition piece
            - "MP": SubAssembly object for the monopile
        """
        url_data_type = "subassemblies"
        url_params = {"asset__title": turbine}
        if subassembly is not None:
            url_params["subassembly_type"] = subassembly
        if model_definition_id is not None:
            url_params["model_definition"] = str(model_definition_id)
        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)
        if not resp.json():
            raise ValueError("No subassemblies found for " + str(turbine))

        material_data = self.get_materials()
        if material_data["exists"]:
            materials = material_data["data"]
        else:
            raise ValueError("No materials found in the database.")

        subassemblies = {}
        for item in resp.json():
            subassembly_type = item["subassembly_type"]
            subassembly_obj = SubAssembly(materials, item, api_object=self)
            if subassembly_type in subassemblies:
                if not isinstance(subassemblies[subassembly_type], list):
                    subassemblies[subassembly_type] = [subassemblies[subassembly_type]]
                subassemblies[subassembly_type].append(subassembly_obj)
            else:
                subassemblies[subassembly_type] = subassembly_obj

        return subassemblies

    def _check_if_need_modeldef(self, subassemblies, turbine):
        """Helper function for some public methods to check if the user needs to specify a model definition."""
        sa_list_length = len(list(subassemblies["subassembly_type"].values))
        sa_unique_list_length = len(set(list(subassemblies["subassembly_type"].values)))
        if sa_list_length > sa_unique_list_length:
            raise ValueError(
                f"Multiple model definitions found for turbine {turbine}. Please specify which one to use."
            )

    def get_owt_geometry_processor(
        self,
        turbines: Union[str, List[str]],
        model_definition: Union[str, None] = None,
        tower_base: Union[float, List[float], None] = None,
        monopile_head: Union[float, List[float], None] = None,
    ) -> OWTs:
        """Return the required processing class.
        Will return data even if some turbines have issues given that at least one is fully OK.

        :param turbines: Title(s) of the turbines.
        :param model_definition: Optional: Title of the model definition.
        :param tower_base: Optional: Height(s) of the tower base.
        :param monopile_head: Optional: Height(s) of the monopile head.
        :return: OWTs object: containing information about all the requested turbines.
        """
        materials_data = self.get_materials()
        if materials_data["exists"]:
            materials = materials_data["data"]
        else:
            raise ValueError("No materials found in the database.")
        owts = []
        successful_turbines = []
        errors = []
        turbines = [turbines] if isinstance(turbines, str) else turbines
        if not isinstance(tower_base, List) and not isinstance(monopile_head, List):
            tower_base = [tower_base] * len(turbines)  # type: ignore
            monopile_head = [monopile_head] * len(turbines)  # type: ignore
        for i, turbine in enumerate(turbines):
            try:
                location_data = self.loc_api.get_assetlocation_detail(
                    assetlocation=turbine
                )
                if location_data["exists"]:
                    location = location_data["data"]
                else:
                    raise ValueError(f"No location found for turbine {turbine}.")
                projectsite = location["projectsite_name"].loc[0]
                subassemblies_data = self.get_subassemblies(
                    projectsite=projectsite,
                    assetlocation=turbine,
                    model_definition=model_definition,
                )
                if subassemblies_data["exists"]:
                    subassemblies = subassemblies_data["data"]
                    self._check_if_need_modeldef(subassemblies, turbine)
                else:
                    raise ValueError(
                        f"No subassemblies found for turbine {turbine}. Please check model definition or database data."
                    )
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
                successful_turbines.append(turbine)
            except ValueError as e:
                errors.append(str(e))
        if errors:
            if successful_turbines:
                warnings.warn(
                    f"There were some errors during processing the request. "
                    f"But some turbines were processed successfully: {', '.join(successful_turbines)}."
                    f"\nErrors:\n" + "\n".join(errors),
                )
            else:
                raise ValueError("\n".join(errors))
        return OWTs(successful_turbines, owts)

    def get_monopile_pyles(
        self,
        projectsite,
        assetlocation,
        cutoff_point=np.nan,
        model_definition: Union[str, None] = None,
    ):
        """
        Returns a dataframe with the monopile geometry with the mudline as reference

        :param projectsite: Name of the project site
        :param assetlocation: Name of the wind turbine location
        :param cutoff_point: Elevation of the load application point in (mLAT) above the mudline
        :param model_definition: Optional: Title of the model definition.
        :return: DataFrame with the monopile geometry.
        """
        # Retrieve the monopile cans
        bbs = self.get_buildingblocks(
            projectsite=projectsite, assetlocation=assetlocation, subassembly_type="MP"
        )
        # Retrieve the monopile subassembly
        sas = self.get_subassemblies(
            projectsite=projectsite,
            assetlocation=assetlocation,
            subassembly_type="MP",
            model_definition=model_definition,
        )
        if sas["exists"]:
            subassemblies = sas["data"]
            self._check_if_need_modeldef(subassemblies, assetlocation)
        else:
            raise ValueError(
                f"No subassemblies found for turbine {assetlocation}. Please check model definition or database data."
            )
        # Water depth
        location_data = self.loc_api.get_assetlocation_detail(
            assetlocation=assetlocation, projectsite=projectsite
        )
        if location_data["exists"]:
            location = location_data["data"]
            water_depth = location["elevation"].values[0]
        else:
            raise ValueError(
                f"No location found for turbine {assetlocation} and hence no water depth can be retrieved."
            )

        # Calculate the pile penetration
        toe_depth_lat = sas["data"]["z_position"].iloc[0]
        penetration = -((1e-3 * toe_depth_lat) - water_depth)

        # Create the pile for subsequent response analysis
        pile = pd.DataFrame()

        for i, row in bbs["data"].iterrows():
            if i != 0:
                pile.loc[i, "Depth to [m]"] = (
                    penetration - 1e-3 * bbs["data"].loc[i - 1, "z_position"]
                )
                pile.loc[i, "Depth from [m]"] = penetration - 1e-3 * row["z_position"]
                pile.loc[i, "Pile material"] = row["material_name"]
                pile.loc[i, "Pile material submerged unit weight [kN/m3]"] = (
                    1e-2 * row["density"] - 10
                )
                pile.loc[i, "Wall thickness [mm]"] = row["wall_thickness"]
                pile.loc[i, "Diameter [m]"] = (
                    1e-3
                    * 0.5
                    * (row["bottom_outer_diameter"] + row["top_outer_diameter"])
                )
                pile.loc[i, "Youngs modulus [GPa]"] = row["youngs_modulus"]
                pile.loc[i, "Poissons ratio [-]"] = row["poissons_ratio"]

        pile.sort_values("Depth from [m]", inplace=True)
        pile.reset_index(drop=True, inplace=True)

        # Cut off at the mudline
        if not np.math.isnan(cutoff_point):
            pile = pile.loc[pile["Depth to [m]"] > cutoff_point].reset_index(drop=True)
            pile.loc[0, "Depth from [m]"] = cutoff_point

        return pile

    def plot_turbines(
        self,
        turbines: Union[List[str], str],
        figures_per_line: int = 5,
        return_fig: bool = False,
        model_definition: Union[str, None] = None,
    ) -> Union[plt.graph_objects.Figure, None]:
        """Plot turbines' frontal geometry.

        :param turbines: Title(s) of the turbines.
        :param figures_per_line: Number of figures per line.
        :param return_fig: Boolean indicating whether to return the figure.
        :param model_definition: Optional: Title of the model definition.
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
            subassemblies_data = self.get_subassemblies(
                assetlocation=turbine,
                model_definition=model_definition,
            )
            if subassemblies_data["exists"]:
                subassemblies = subassemblies_data["data"]
                self._check_if_need_modeldef(subassemblies, turbine)
            else:
                raise ValueError(
                    f"No subassemblies found for turbine {turbine}. Please check model definition or database data."
                )
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
        if return_fig:
            return fig
        else:
            fig.show()
