"""Module defining API to retrieve/plot specific fatigue data from the owimetadatabase."""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from owimetadatabase_preprocessor.fatigue.data_objects import (
    FATIGUE_DETAILS_COLORS,
    FatigueDetail,
    FatigueSubAssembly,
    SNCurve,
)
from owimetadatabase_preprocessor.geometry.io import GeometryAPI, LocationsAPI
from owimetadatabase_preprocessor.io import API


class FatigueAPI(API):
    """Class to connect to the fatigue data API with methods to retrieve data.

    A number of methods are provided to query the database via the owimetadatabase API.
    For FatigueAPI the ``get_*`` methods return lists of custom objects storing/processing fatigue data.
    The methods are written such that a number of mandatory URL parameters are required (see documentation of the methods).
    The URL parameters can be expanded with Django-style additional filtering arguments
    (e.g. ``title__icontains="BBG01"``) as optional keyword arguments.
    Knowledge of the Django models is required for this (see ``owimetadatabase`` code).
    """

    def __init__(
        self,
        api_subdir: str = "/fatigue/userroutes/",
        **kwargs,
    ) -> None:
        """Constructor for the FatigueAPI class.

        :api_subdir: Subdirectory for the API.
        :param kwargs: Additional keyword arguments (see the base class).
        """
        super().__init__(**kwargs)
        self.geo_api = GeometryAPI(**kwargs)
        self.loc_api = LocationsAPI(**kwargs)
        self.api_root = self.api_root + api_subdir

    def get_sncurves(self, **kwargs) -> List[SNCurve]:
        """Get all available SN curves requested by the user.

        :param kwargs: Any API filter, e.g. 'title__icontains=NRTA1'
        :return: List of SNCurve objects representing SN curves
        """
        url_data_type = "sncurve"
        url_params = kwargs

        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)

        if not resp.json():
            raise ValueError("No SN curves found for the specified parameters.")
        sncurves = [SNCurve(item, api_object=self) for item in resp.json()]
        return sncurves

    def get_fatiguedetails(self, **kwargs) -> List[FatigueDetail]:
        """Get all fatigue details according to the specified search parameters (see kwargs).

        :param kwargs: Any API filter, e.g. 'title__icontains': 'NW2F04_MP' for a specific turbine and subassembly
        :return: List of FatigueDetail objects representing fatigue data for specified elements
        """
        url_data_type = "fatiguedetail"
        url_params = kwargs

        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)

        if not resp.json():
            raise ValueError("No fatigue details found for the specified parameters.")
        fatigue_details = [FatigueDetail(item, api_object=self) for item in resp.json()]
        return fatigue_details

    def get_fatiguesubassembly(
        self, turbine: str, subassembly: str = None
    ) -> List[FatigueSubAssembly]:
        """Get all fatigue details for a given turbine/turbine subassembly.

        :param turbine: Turbine title (e.g. 'BBC01')
        :param subassembly: Sub-assembly type (e.g. 'MP', 'TW', 'TP')
        :return: List of FatigueSubAssembly objects representing subassemblies
        """
        url_data_type = "subassemblies"
        if subassembly is not None:
            url_params = {"asset__title": turbine, "subassembly_type": subassembly}
        else:
            url_params = {"asset__title": turbine}
        resp = self.geo_api.send_request(url_data_type, url_params)
        self.geo_api.check_request_health(resp)
        if not resp.json():
            raise ValueError("No subassemblies found for " + str(turbine))
        sas_types = [j["subassembly_type"] for j in resp.json()]
        sas = [FatigueSubAssembly(item, api_object=self) for item in resp.json()]
        subassemblies = {k: v for (k, v) in zip(sas_types, sas)}
        return subassemblies

    def fatiguedetails_df(
        self,
        turbines: Union[str, List[str], np.ndarray] = None,
        projectsite_name: str = None,
    ) -> pd.DataFrame:
        """Return a dataframe with all fatigue details for given turbine(s).

        :param turbines: Turbine name(s)
        :param projectsite_name: Name of the projectsite
        :return: Pandas DataFrame with all fatigue details for given turbine(s)
        """
        if isinstance(turbines, str):
            turbines = [turbines]
        df = []
        if turbines is not None:
            for turbine in turbines:
                url_data_type = "fatiguedetail"
                url_params = {"asset_name": turbine}
                resp = self.send_request(url_data_type, url_params)
                self.check_request_health(resp)
                if resp.json():
                    df.append(pd.DataFrame(resp.json()))
                else:
                    raise ValueError("No fatigue details found for {}.".format(turbine))
        else:
            if projectsite_name is None:
                raise ValueError("No projectsite_name defined.")
            url_data_type = "fatiguedetail"
            url_params = {"projectsite_name": projectsite_name}
            resp = self.send_request(url_data_type, url_params)
            self.check_request_health(resp)
            if resp.json():
                if resp.json()["message"] == "Endpoint request timed out":
                    raise TimeoutError("Endpoint request timed out")
                df.append(pd.DataFrame(resp.json()))
            else:
                raise ValueError("No fatigue details found.")
        if resp.json():
            fd_pos = [
                FatigueDetail(item, api_object=self).buildingblock.position
                for item in resp.json()
            ]
            fd_pos_dict = {"x_position": [], "y_position": [], "z_position": []}
            for fd_p in fd_pos:
                fd_pos_dict["x_position"].append(fd_p.x)
                fd_pos_dict["y_position"].append(fd_p.y)
                fd_pos_dict["z_position"].append(fd_p.z)
        df = pd.concat(df).reset_index()
        df = pd.concat([df, pd.DataFrame(fd_pos_dict)], axis=1)
        df["damage"] = 20 / df[["fatiguelifeout", "fatiguelifein"]].min(axis=1)
        if df["damage"].isnull().values.any():
            # for na_damage in list(df[df['damage'].isnull()]['title']):
            #     warnings.warn('There are NaN in fatigue life at: ' + na_damage)
            df = df[df["damage"].notna()]
        return df

    def fatiguedetails_animatedquickview(
        self,
        turbines: Union[str, List[str], np.ndarray] = None,
        projectsite_name: str = None,
        showmudline: bool = True,
        show: bool = True,
    ) -> Dict[str, Union[go.Figure, pd.DataFrame]]:
        """Plot (animated) fatigue data information for given turbine(s).

        :param turbines: Turbine name(s)
        :param projectsite_name: Name of the projectsite
        :param showmudline: Whether to show mudline in the plot
        :param show: Whether to show the plot
        :return: Dictionary with the fatigue data DataFrame and Plotly figure
        """
        # ? Dataset
        dataset = self.fatiguedetails_df(
            turbines=turbines,
            projectsite_name=projectsite_name,
        )

        # ? Lists of variables of interest
        # * assets: turbines
        # * polymorphic_ctypes: types of fatigue details
        # * subtypes: tower, monopile, transition piece
        assets = dataset["asset_name"].unique()
        pbar = tqdm(assets)
        subtypes = dataset["subassembly_type"].unique()
        polymorphic_ctypes = dataset["polymorphic_ctype"].unique()

        # ? Figure instantiation
        fig_dict = {"data": [], "layout": {}, "frames": []}

        # ? Fill-in most of layout
        # * Basics
        # fig_dict['layout']['xaxis'] = {'title': 'Length, mm'}
        fig_dict["layout"]["hovermode"] = "closest"
        fig_dict["layout"]["height"] = 800
        fig_dict["layout"]["width"] = 700
        fig_dict["layout"]["margin"] = {"l": 50, "t": 50, "r": 50, "b": 50, "pad": 4}
        fig_dict["layout"]["paper_bgcolor"] = "#ffffff"
        fig_dict["layout"]["plot_bgcolor"] = "#ffffff"
        fig_dict["layout"]["showlegend"] = True
        fig_dict["layout"]["autosize"] = True
        # * Interactivity
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 300,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                        "label": "▶️",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸️",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

        # ? Initialise data
        asset = dataset["asset_name"][0]
        fig_dict["data"] = self._add_data_to_fatiguesubassembly(
            dataset,
            asset,
            subtypes,
            polymorphic_ctypes,
            is_frame=False,
            showmudline=showmudline,
        )[0]["data"]

        # ? Make frames
        # * Initialise slider
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Asset: ",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }
        # * Loop through assets
        for asset in pbar:
            pbar.set_description("Processing %s" % asset)
            # - generate frame
            frame = self._add_data_to_fatiguesubassembly(
                dataset,
                asset,
                subtypes,
                polymorphic_ctypes,
                is_frame=True,
                showmudline=showmudline,
            )[0]
            fig_dict["frames"].append(frame)
            # - update slider
            slider_step = {
                "args": [
                    [asset],
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": asset,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        # ? Finalise figure
        fig_dict["layout"]["sliders"] = [sliders_dict]

        min_y = []
        max_y = []
        for data in fig_dict["data"]:
            for k, y in data.items():
                if k == "y":
                    min_y.append(min(y))
                    max_y.append(max(y))
        min_y = min(min_y)
        max_y = max(max_y)

        fig_dict["layout"]["yaxis"] = {
            "title": "Height, mm",
            "scaleanchor": "x",
            "scaleratio": 1,
            "range": [min_y, max_y],
        }
        if show:
            fig = go.Figure(fig_dict)
            fig.show()

        return {"DataFrame": dataset, "Plotly": fig_dict}

    def fatiguedetails_serializedquickview(
        self,
        turbines: Union[str, List[str], np.ndarray] = None,
        projectsite_name: str = None,
        showmudline: bool = True,
        x_step: float = 10000.0,
        show: bool = True,
        marker_scaler: int = 6,
    ) -> Dict[str, Union[go.Figure, pd.DataFrame]]:
        """Plot (static) fatigue data information for given turbine(s).

        :param turbines: Turbine name(s)
        :param projectsite_name: Name of the projectsite
        :param showmudline: Whether to show mudline in the plot
        :param x_step:
        :param show: Whether to show the figure
        :param marker_scaler:
        :return: Dictionary with the fatigue data DataFrame and Plotly figure
        """
        # ? Dataset
        dataset = self.fatiguedetails_df(
            turbines=turbines,
            projectsite_name=projectsite_name,
        )

        # ? Lists of variables of interest
        # * assets: turbines
        # * polymorphic_ctypes: types of fatigue details
        # * subtypes: tower, monopile, transition piece
        assets = dataset["asset_name"].unique()
        pbar = tqdm(assets)
        subtypes = dataset["subassembly_type"].unique()
        polymorphic_ctypes = dataset["polymorphic_ctype"].unique()

        # ? Figure instantiation
        fig_dict = {
            "data": [],
            "layout": {},
        }

        # ? Fill-in most of layout
        # * Basics
        # fig_dict['layout']['xaxis'] = {'title': 'Length, mm'}
        fig_dict["layout"]["hovermode"] = "closest"
        # fig_dict['layout']['height'] = 800
        # fig_dict['layout']['width'] = 1500
        fig_dict["layout"]["margin"] = {"l": 50, "t": 50, "r": 50, "b": 50, "pad": 4}
        fig_dict["layout"]["paper_bgcolor"] = "#ffffff"
        fig_dict["layout"]["plot_bgcolor"] = "#ffffff"
        fig_dict["layout"]["showlegend"] = True
        fig_dict["layout"]["autosize"] = True
        # * Loop through assets
        for nr, asset in enumerate(pbar):
            pbar.set_description("Processing %s" % asset)
            # - generate frame
            fig_dict["data"].extend(
                self._add_data_to_fatiguesubassembly(
                    dataset,
                    asset,
                    subtypes,
                    polymorphic_ctypes,
                    is_frame=False,
                    showmudline=showmudline,
                    x_offset=nr * x_step,
                    x_step=x_step,
                    marker_scaler=marker_scaler,
                )[0]["data"]
            )

        # ? Finalise figure
        min_y = []
        max_y = []
        for data in fig_dict["data"]:
            for k, y in data.items():
                if k == "y":
                    min_y.append(min(y))
                    max_y.append(max(y))
        min_y = min(min_y)
        max_y = max(max_y)

        fig_dict["layout"]["yaxis"] = {
            "title": "Height, mm",
            "scaleanchor": "x",
            "scaleratio": 1,
        }
        if show:
            fig = go.Figure(fig_dict)
            fig.show()

        return {"DataFrame": dataset, "Plotly": fig_dict}

    def _add_data_to_fatiguesubassembly(
        self,
        df: pd.DataFrame,
        asset: str,
        subtypes: List[str],
        polymorphic_ctypes: List[str],
        is_frame: bool = False,
        showmudline: bool = False,
        x_offset: float = 0.0,
        x_step: float = 10000,
        marker_scaler: int = 8,
    ):
        """Helper internal method to add data to the fatigue subassembly plot."""
        if is_frame:
            f = {"data": [], "name": str(asset)}
        else:
            f = {"data": []}
        df_by_asset = df[df["asset_name"] == asset]
        subass = self.geo_api.get_subassembly_objects(asset)

        if showmudline:
            url_data_type = "assetlocations"
            url_params = {"title": asset}
            elevation_req = self.loc_api.send_request(url_data_type, url_params)
            self.loc_api.check_request_health(elevation_req)
            elevation = elevation_req.json()[0]["elevation"]
            mudline_dict = {
                "x": [x_offset - x_step / 2, x_offset + x_step / 2],
                "y": [elevation * 1000] * 2,
                "mode": "lines",
                "name": "Mudline",
                "hoverinfo": "text",
                "hovertext": asset
                + " mudline elevation: "
                + str(np.round(elevation, 1))
                + "m",
                "line": {"color": "SaddleBrown", "width": 4},
            }
            f["data"].append(mudline_dict)
            waterlevel_dict = {
                "x": [x_offset - x_step / 2, x_offset + x_step / 2],
                "y": [0, 0],
                "fill": "tonexty",
                "mode": "lines",
                "name": "Water level",
                "hoverinfo": "text",
                "hovertext": "Water level",
                "line": {"color": "DodgerBlue", "width": 0.5},
            }
            f["data"].append(waterlevel_dict)

        bod = {}
        tod = {}
        sub_col = {}
        sub_z = {}
        sub_h = {}
        for sub_key, sub in subass.items():
            bod[sub.type] = []
            tod[sub.type] = []
            sub_col[sub.type] = []
            sub_z[sub.type] = []
            sub_h[sub.type] = []
            for bb in sub.building_blocks:
                bod[sub.type].append(bb.bottom_outer_diameter)
                tod[sub.type].append(bb.top_outer_diameter)
                sub_col[sub.type].append(sub.color)
                sub_z[sub.type].append(sub.position.z)
                sub_h[sub.type].append(sub.height)

        bod = {k: np.nanmax(np.array(v, dtype=np.float64)) for (k, v) in bod.items()}
        tod = {k: np.nanmax(np.array(v, dtype=np.float64)) for (k, v) in tod.items()}
        sub_col = {k: list(set(v))[0] for (k, v) in sub_col.items()}
        sub_z = {k: list(set(v))[0] for (k, v) in sub_z.items()}
        sub_h = {k: list(set(v))[0] for (k, v) in sub_h.items()}
        for subtype in subtypes:
            df_by_ass_and_sub = df_by_asset[df_by_asset["subassembly_type"] == subtype]
            x = [
                _ + x_offset
                for _ in [
                    -tod[subtype] / 2,
                    tod[subtype] / 2,
                    0,
                    0,
                    -bod[subtype] / 2,
                    bod[subtype] / 2,
                ]
            ]
            structure_dict = {
                "x": x,
                "y": [sub_z[subtype] + sub_h[subtype]] * 3 + [sub_z[subtype]] * 3,
                "mode": "lines",
                "name": subtype,
                "hoverinfo": "text",
                "hovertext": asset + "_" + subtype,
                "line": {"color": sub_col[subtype], "width": 1},
            }
            f["data"].append(structure_dict)

            for polymorphic_ctype in polymorphic_ctypes:
                df_by_ass_sub_and_res = df_by_ass_and_sub[
                    df_by_ass_and_sub["polymorphic_ctype"] == polymorphic_ctype
                ]
                if len(df_by_ass_sub_and_res) == 0:
                    continue
                l_h = len(list(df_by_ass_sub_and_res["title"]))
                flout = [
                    ["fatigue life out: "] * l_h,
                    list(df_by_ass_sub_and_res["fatiguelifeout"]),
                    [" yrs"] * l_h,
                ]
                flin = [
                    ["fatigue life in: "] * l_h,
                    list(df_by_ass_sub_and_res["fatiguelifein"]),
                    [" yrs"] * l_h,
                ]
                snout = [
                    ["SN curve out: "] * l_h,
                    list(df_by_ass_sub_and_res["sncurveout"]),
                ]
                snin = [
                    ["SN curve in: "] * l_h,
                    list(df_by_ass_sub_and_res["sncurvein"]),
                ]
                scfout = [["SCF out: "] * l_h, list(df_by_ass_sub_and_res["scfout"])]
                scfin = [["SCF in: "] * l_h, list(df_by_ass_sub_and_res["scfin"])]
                msf = [
                    ["Material safety factor: "] * l_h,
                    list(df_by_ass_sub_and_res["materialsafetyfactor"]),
                ]

                se = [
                    ["Scale effect: "] * l_h,
                    list(df_by_ass_sub_and_res["scaleeffect"]),
                ]
                try:
                    flout = [
                        "{}<b>{}</b>{}".format(a_, int(b_), c_)
                        for a_, b_, c_ in zip(*flout)
                    ]
                    snout = ["{}{}".format(a_, s_) for a_, s_ in zip(*snout)]
                    scfout = ["{}{}".format(a_, s_) for a_, s_ in zip(*scfout)]
                except Exception:
                    flout = [""] * l_h
                    snout = [""] * l_h
                    scfout = [""] * l_h
                try:
                    flin = [
                        "{}<b>{}</b>{}".format(a_, int(b_), c_)
                        for a_, b_, c_ in zip(*flin)
                    ]
                    snin = ["{}{}".format(a_, s_) for a_, s_ in zip(*snin)]
                    scfin = ["{}{}".format(a_, s_) for a_, s_ in zip(*scfin)]
                except Exception:
                    flin = [""] * l_h
                    snin = [""] * l_h
                    scfin = [""] * l_h
                try:
                    msf = ["{}{}".format(a_, m_) for a_, m_ in zip(*msf)]
                except Exception:
                    msf = [""] * l_h
                try:
                    se = ["{}{}".format(a_, e_) for a_, e_ in zip(*se)]
                except Exception:
                    se = [""] * l_h
                fl = [
                    "<br><i>{}</i></br>{}</br>{}</br>{}</br>{}</br>{}</br>{}</br>{}</br>{}".format(
                        t_, a_, b_, si_, so_, scfi_, scfo_, e_, se_
                    )
                    for t_, a_, b_, si_, so_, scfi_, scfo_, e_, se_ in zip(
                        df_by_ass_sub_and_res["title"],
                        flin,
                        flout,
                        snin,
                        snout,
                        scfin,
                        scfout,
                        msf,
                        se,
                    )
                ]
                x = [_ + x_offset for _ in list(df_by_ass_sub_and_res["y_position"])]
                data_dict = {
                    "x": x,
                    "y": [
                        z_lat + z_
                        for (z_lat, z_) in zip(
                            list(df_by_ass_sub_and_res["z_position"]),
                            [sub_z[subtype]] * l_h,
                        )
                    ],
                    "mode": "markers",
                    "hoverinfo": "text",
                    "hovertext": fl,
                    "text": list(df_by_ass_sub_and_res["title"]),
                    "marker": {
                        "color": FATIGUE_DETAILS_COLORS[polymorphic_ctype],
                        "sizemode": "area",
                        "sizeref": df["damage"].max() / marker_scaler**2,
                        "size": list(df_by_ass_sub_and_res["damage"]),
                    },
                    "name": str(subtype) + " - " + str(polymorphic_ctype),
                }
                f["data"].append(data_dict)

        return (f, sub_z)
