"""
This module receive processed data (or even raw data if needed) from the 
API client and data processor, build the Plotly figures, and either return 
or show them.
"""
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from typing import Any, Dict, List, Union
from groundhog.siteinvestigation.insitutests.pcpt_processing import (
    plot_combined_longitudinal_profile,
    plot_longitudinal_profile,
)
from owimetadatabase_preprocessor.soil.io import SoilAPI
from owimetadatabase_preprocessor.soil.processing.soil_pp import SoilDataProcessor

# TODO : plot_cpt_fence method is commented out in the original code
# since it's believed it will not work properly. It calls an instance method
# get_cpttest_detail()from SoilAPI class. It should be fixed.

class SoilPlot:
    """Class to visualize soil data using Plotly."""

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

        :param profiles: List with georeferenced soil profiles 
            (run plot_soilprofile_fence first)
        :param cpts: List with georeference CPTs 
            (run plot_cpt_fence first)
        :param startpoint: Name of the CPT location for the start point
        :param endpoint: Name of the CPT location for the end point
        :param band: Thickness of the band (in m, default=1000m)
        :param scale_factor: Width of the CPT axis in the fence diagram 
            (default=10)
        :param extend_profile: Boolean determining whether the profile needs 
            to be extended (default=True)
        :param show_annotations: Boolean determining whether annotations are 
            shown (default=True)
        :param general_layout: Dictionary with general layout options 
            (default = dict())
        :param fillcolordict: Dictionary with colors for soil types
        :param logwidth: Width of the log in the fence diagram
        :param opacity: Opacity of the soil profile logs
        :param uniformcolor: If a valid color is provided (e.g. 'black'), it 
            is used for all CPT traces
        :return: Dictionary with the following keys:

            - 'diagram': Plotly figure with the fence diagram for CPTs and 
                soil profiles
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

    def plot_testlocations(self, return_fig: bool = False, **kwargs) -> None:
        """
        Retrieves soil test locations and generates a Plotly plot to show them.

        :param return_fig: Boolean indicating whether the Plotly figure object 
            needs to be returned (default is False which simply shows the plot)
        :param kwargs: Keyword arguments for the search 
            (see ``get_testlocations``)
        :return: Plotly figure object with selected asset locations plotted 
            on OpenStreetMap tiles (if requested)
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

    # def plot_cpt_fence(
    #     self,
    #     cpt_df: pd.DataFrame,
    #     start: str,
    #     end: str,
    #     band: float = 1000.0,
    #     scale_factor: float = 10.0,
    #     extend_profile: bool = True,
    #     plotmap: bool = False,
    #     show_annotations: bool = True,
    #     general_layout: Dict[Any, Any] = dict(),
    #     uniformcolor: Union[str, None] = None,
    #     **kwargs,
    # ) -> Dict[str, Union[List[pd.DataFrame], go.Figure]]:
    #     """Creates a fence diagram for CPTs.

    #     :param cpt_df: Dataframe with the summary data of the selected CPTs
    #     :param start: Name of the location for the start point
    #     :param end: Name of the location for the end point
    #     :param band: Thickness of the band (in m, default=1000m)
    #     :param scale_factor: Width of the CPT axis in the fence diagram 
    #         (default=10)
    #     :param extend_profile: Boolean determining whether the profile needs 
    #         to be extended (default=True)
    #     :param plotmap: Boolean determining whether a map with the locations is 
    #         shown (default=False)
    #     :param show_annotations: Boolean determining whether annotations are 
    #         shown (default=True)
    #     :param general_layout: Dictionary with general layout options 
    #         (default = dict())
    #     :param uniformcolor: If a valid color is provided (e.g. 'black'), it is 
    #         used for all CPT traces
    #     :param kwargs: Keyword arguments for the get_insitutests method
    #     :return: Dictionary with the following keys:

    #         - 'cpts': List of CPT objects
    #         - 'diagram': Plotly figure with the fence diagram
    #     """
    #     selected_cpts = cpt_df
    #     cpts = SoilDataProcessor._objects_to_list(selected_cpts, self.get_cpttest_detail, "cpt")
    #     cpt_fence_fig_1 = plot_longitudinal_profile(
    #         cpts=cpts,
    #         latlon=True,
    #         start=start,
    #         end=end,
    #         band=band,
    #         scale_factor=scale_factor,
    #         extend_profile=extend_profile,
    #         plotmap=plotmap,
    #         show_annotations=show_annotations,
    #         general_layout=general_layout,
    #         uniformcolor=uniformcolor,
    #         **kwargs,
    #     )
    #     return {"cpts": cpts, "diagram": cpt_fence_fig_1}