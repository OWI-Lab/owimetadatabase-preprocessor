"""Module defining classes handling different kinds of fatigue data."""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import warnings
from copy import deepcopy
from itertools import cycle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from pandas import DataFrame

from owimetadatabase_preprocessor.geometry.structures import (
    PLOT_SETTINGS_SUBASSEMBLY,
    BuildingBlock,
    Material,
    Position,
    SubAssembly,
)
from owimetadatabase_preprocessor.utility.utils import hex_to_dec

COLOR_LIST = [
    "#4e79a7",
    "#a0cbe8",
    "#f28e2b",
    "#59a14f",
    "#8cd17d",
    "#b6992d",
    "#f1ce63",
    "#499894",
    "#e15759",
    "#79706e",
    "#d37295",
    "#b07aa1",
    "#d4a6c8",
    "#9d7660",
]


COLOR_LIST_LEN = len(COLOR_LIST)


FATIGUE_DETAILS_COLORS = {
    i: COLOR_LIST[np.random.randint(COLOR_LIST_LEN)] for i in range(100)
}


# FATIGUE_DETAILS_COLORS = {
#     45: COLOR_LIST[9],
#     41: COLOR_LIST[1],
#     40: COLOR_LIST[10],
#     43: COLOR_LIST[3],
#     38: COLOR_LIST[11],
#     36: COLOR_LIST[5],
#     # 'EarthingPlate': COLOR_LIST[6],
#     # 'Hook': COLOR_LIST[7],
#     # 'Rail': COLOR_LIST[8],
# }


class SNCurve:
    """Class to store and handle retrieved SN curves.

    Each SN curve data requires an API call to the Fatigue API to retrieve the
    necessary data through Owimetadatabase. SNCurve instances should be created by the FatigueAPI class.
    """

    _color_ids = cycle(range(COLOR_LIST_LEN))

    def __init__(
        self,
        json_file: Dict[str, Union[None, str, np.int64, np.float64]],
        api_object=None,
    ) -> None:
        """Constructor for the SNCurve class.

        :param json_file: The JSON object containing the SN curve data.
        :param api_object: The FatigueAPI instance that created the SNCurve instance.
        """
        self.api = api_object
        self.id = json_file["id"]
        self.title = json_file["title"]
        if json_file["description"]:
            self.description = json_file["description"]
        else:
            self.description = ""
        self.json_file = json_file
        self.k = json_file["k"]

        m = self.json_file["m"]
        log_a = self.json_file["log_a"]
        self._m = (
            np.array(m, dtype=np.float64)
            if m is not None and isinstance(m, (np.float64, int, list))
            else m
        )
        self._log_a = (
            np.array(log_a, dtype=np.float64)
            if log_a is not None and isinstance(log_a, (np.float64, int, list))
            else log_a
        )

        self._n_knee = self.json_file["n_knee"]
        self.environment = self.json_file["environment"]
        self.curve = self.json_file["title"]
        self.curve_ini = self.curve[4] if self.curve[5] == "-" else self.curve[4:6]
        self.norm = self.json_file["guideline"]

        self.unit_string = "MPa"  # TODO: Add unit_string to owimetadb
        cmap = ListedColormap(hex_to_dec(COLOR_LIST))  # plt.cm.get_cmap('Set3')
        # plt.rcParams['axes.color_cycle'] = COLOR_LIST
        self.color = cmap(next(self._color_ids))
        if isinstance(m, Iterable) and len(m) > 1:
            self.bi_linear = True
            self._check_bi_linear()
        else:
            self.bi_linear = False
        if self.bi_linear and self._n_knee is None:
            raise ValueError(
                "For bi-linear S-N Curves a Knee point (N_knee) has to be defined"
            )

    @property
    def m(self) -> Union[list, np.ndarray]:
        """m parameter of the SN-Curve."""
        return self._m

    @m.setter
    def m(self, value: Union[list, np.ndarray]) -> None:
        self._m = value

    @property
    def log_a(self) -> Union[list, np.ndarray]:
        """log_a parameter of the SN-Curve."""
        return self._log_a

    @log_a.setter
    def log_a(self, value: Union[list, np.ndarray]) -> None:
        self._log_a = value

    @property
    def n_knee(self) -> float:
        """Knee point of the SN-Curve."""
        return self._n_knee

    @n_knee.setter
    def n_knee(self, value: float) -> None:
        self._n_knee = value

    @property
    def name(self) -> str:
        """Name of the SN-Curve"""
        if self.curve is not None:
            if self.environment is not None:
                if self.norm is not None:
                    return self.norm + ": " + self.curve_ini + " - " + self.environment
                else:
                    return self.curve + " - " + self.environment
            else:
                return self.curve
        else:
            return ""

    @property
    def color_str(self) -> str:
        """Returns the color attribute as a string suitable for plot.ly (hex).

        :return: str of color
        """
        if not isinstance(self.color, str):
            return "#{:02x}{:02x}{:02x}".format(
                int(self.color[0] * 255),
                int(self.color[1] * 255),
                int(self.color[2] * 255),
            )
        else:
            return self.color

    def n(
        self, sigma: Union[List[np.float64], np.ndarray]
    ) -> Union[np.float64, np.ndarray]:
        """Return the number of cycles for a certain stress range.

        :param sigma: Stress ranges for which the maximum number of cycles is to be calculated
        """
        if not isinstance(sigma, np.ndarray):
            sigma = np.array(sigma)

        def _calc_n(sigma, m, log_a):
            return 10 ** (log_a) * np.power(sigma, -m)

        if not self.bi_linear:
            n = _calc_n(sigma, self.m, self.log_a)
        else:
            threshold = np.float64(self.sigma(self.n_knee))
            n = np.zeros(np.shape(sigma))
            n[sigma >= threshold] = _calc_n(
                sigma[sigma >= threshold], self.m[0], self.log_a[0]
            )
            n[sigma < threshold] = _calc_n(
                sigma[sigma < threshold], self.m[1], self.log_a[1]
            )
        return n

    def sigma(
        self, n: Union[List[np.float64], np.ndarray]
    ) -> Union[np.float64, np.ndarray]:
        """Return the stress ranges for a certain number of n.

        :param n: Number of cycles for which the stress ranges have to be calculated
        """
        if not isinstance(n, np.ndarray):
            n = np.array(n)

        def _calc_sigma(n, m, log_a):
            return (10**log_a) ** (1 / m) * np.power(n, -1 / m)

        if not self.bi_linear:
            sigma = _calc_sigma(n, self.m, self.log_a)
        else:
            sigma = np.zeros(np.shape(n))
            sigma[n <= self.n_knee] = _calc_sigma(
                n[n <= self.n_knee], self.m[0], self.log_a[0]
            )
            sigma[n > self.n_knee] = _calc_sigma(
                n[n > self.n_knee], self.m[1], self.log_a[1]
            )
        return sigma

    def _check_bi_linear(self) -> None:
        sigma_1 = self.sigma(self.n_knee)
        sigma_2 = self.sigma(self.n_knee + 1e-3)
        if np.abs(sigma_1 - sigma_2) > 1e-1:
            w = [
                "Both ends of the bi-linear curve \033[95m",
                self.name,
                "\033[0m do not meet up at the knee-point. ",
                "Check the SN curve definition.",
            ]
            warnings.warn("".join(w))

    def _sn_curve_data_points(
        self,
        n: Union[List[np.float64], np.ndarray, None] = None,
        sigma: Union[List[np.float64], np.ndarray, None] = None,
    ) -> Union[List[np.float64], np.ndarray]:
        if sigma is None:
            if n is None:
                # Default range of n
                n = np.logspace(4, 9, num=20, base=10)
        else:
            if n is not None:
                raise TypeError("Either n or sigma shall be NoneType")
            n = self.N(sigma)
        if self.bi_linear:
            n = np.sort(np.append(n, np.array(self.n_knee)))
        sigma = self.sigma(n)
        return n, sigma

    def plotly(
        self,
        n: Union[List[np.float64], np.ndarray, None] = None,
        sigma: Union[List[np.float64], np.ndarray, None] = None,
        show: bool = True,
    ) -> Tuple[List[go.Scattergl], go.Layout]:
        """Use plotly to plot the SN curve

        :param n: Number of cycles for which the stress ranges have to be calculated and the plot shown.
        :param sigma: Stress ranges for which the maximum number of cycles is to be calculated and the plot shown.
        :param show: If True, the plot will be shown.
        :return: data, layout
        """
        n, sigma = self._sn_curve_data_points(n, sigma)
        data = [
            go.Scattergl(
                x=n,  # assign x as the dataframe column 'x'
                y=sigma,
                name=self.name,
                line=dict(color=self.color_str, width=1),
            )
        ]
        layout = go.Layout(
            xaxis=dict(
                title=go.layout.xaxis.Title(text="Number of cycles"), type="log"
            ),
            yaxis=dict(
                title=go.layout.yaxis.Title(text="Stress range, " + self.unit_string),
                type="log",
            ),
        )
        if show:
            fig = go.Figure(data=data, layout=layout)
            fig.show()
        return data, layout

    def as_dict(self) -> Dict[str, Any]:
        """Returns the SN curve description as a dictionary."""
        return {
            "name": self.name,
            "units": self.unit_string,
            "m": self.m,
            "log_a": self.log_a,
            "n_knee": self.n_knee,
        }

    def as_df(self) -> DataFrame:
        """Returns the SN curve description as a DataFrame."""
        d = self.as_dict()
        try:
            del d["title"]
        except Exception:
            pass
        df = DataFrame.from_dict(d, orient="index", columns=[self.title])
        return df

    def _repr_html_(self):
        return self.as_df().to_html()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FatigueDetail:
    """Class to store the fatigue data of turbine elements.

    Each fatigue detail data requires an API call to the
    FatigueAPI class to retrieve the necessary data through
    Owimetadatabase. FatigueDetail instances should be created by the FatigueAPI class.

    :param name: The name of the detail.
    :param projectsite: The project site of the detail.
    :param fd_type: The type of the detail.
    :param title: The title of the detail.
    :param description: The description of the detail.
    :param modeldefinition: The model definition of the detail.
    :param fatiguelifein: The fatigue life of the detail.
    :param fatiguelifeout: The fatigue life of the detail.
    :param scfin: The SCF of the detail.
    :param scfout: The SCF of the detail.
    :param materialsafatyfactor: The material safety factor of the detail.
    :param scaleeffect: The scale effect of the detail.
    :param sncurves: The SN curves of the detail.
    :param material: The material of the detail.
    :param buildingblock: The building block of the detail.
    :param height: The height of the detail.
    """

    def __init__(
        self,
        json_file: Dict[str, Union[None, str, np.int64, np.float64]],
        api_object=None,
        subassembly: SubAssembly = None,
    ) -> None:
        """Constructor for the FatigueDetail class.

        :param json: The JSON object containing the fatigue data.
        :param api_object: The FatigueAPI instance that created the FatigueDetail instance.
        :param subassembly: The SubAssembly instance that created the FatigueDetail instance.
        """
        self.api = api_object
        self.json = json_file

        self.asset = json_file["asset_name"]
        self.subassembly_type = json_file["subassembly_type"]
        self.subassembly_name = json_file["subassembly_name"]
        self.projectsite = json_file["projectsite_name"]
        self.fd_type = json_file["polymorphic_ctype"]
        self.title = json_file["title"]
        self.description = json_file["description"]
        self.modeldefinition = json_file["modeldefinition"]
        self.fatiguelifein = json_file["fatiguelifein"]
        self.fatiguelifeout = json_file["fatiguelifeout"]
        self.scfin = json_file["scfin"] if "scfin" in json_file else None
        self.scfout = json_file["scfout"] if "scfout" in json_file else None
        self.materialsafetyfactor = json_file["materialsafetyfactor"]
        self.scaleeffect = json_file["scaleeffect"]

        self._sncurves = None
        self._buildingblock = None
        self._buildingblocktop = None

        self.subassembly = subassembly
        self.material = None

        if "material" in self.json and self.subassembly:
            material_title = self.json["material"]
            for mat in self.subassembly.materials:
                if mat["title"] == material_title:
                    self.material = mat
                    break

    @property
    def sncurves(self) -> Dict[Dict[str, str], SNCurve]:
        """SN curves of the detail.

        :return: Dictionary with SN curves of the detail.
        """
        # !TODO: Understand why multiple filters are not applied by Django REST.
        # NOTE: For this reason, the code hereafter uses a workaround.
        if self._sncurves:
            return self._sncurves

        if self.json["sncurvein"] is not None and self.json["sncurveout"] is not None:
            prmtr = {
                "sncurvein": self.json["sncurvein"],
                "sncurveout": self.json["sncurveout"],
            }
        elif self.json["sncurvein"] and not self.json["sncurveout"]:
            prmtr = {
                "sncurvein": self.json["sncurvein"],
            }
        elif self.json["sncurveout"] and not self.json["sncurvein"]:
            prmtr = {
                "sncurveout": self.json["sncurveout"],
            }

        sns = []
        for p in prmtr:
            if len(sns) > 0:
                sns += [
                    self.api.send_request(
                        url_data_type="sncurve",
                        url_params={"title": prmtr[p]},
                    ).json()[0]
                ]
            else:
                sns = [
                    self.api.send_request(
                        url_data_type="sncurve",
                        url_params={"title": prmtr[p]},
                    ).json()[0]
                ]
        if len(sns) > 0:
            self._sncurves = {k: SNCurve(sn) for (k, sn) in zip(prmtr, sns)}
            return self._sncurves
        else:
            raise ValueError("No SN curves found.")

    @property
    def position(self) -> Position:
        """Position of the detail."""
        if "vertical_position_reference_sistem" in self.json:
            return Position(
                x=self.json["x_position"],
                y=self.json["y_position"],
                z=self.json["z_position"],
                reference_system=self.json["vertical_position_reference_system"],
            )
        return self.buildingblock.position

    @property
    def buildingblock(self) -> BuildingBlock:
        """Building block to which the detail belongs."""
        if self._buildingblock:
            return self._buildingblock
        if self.fd_type != "Rail":
            prmtr = self.json["tubularsection"]
            bb = self.api.geo_api.send_request(
                url_data_type="buildingblocks",
                url_params={"id": prmtr},
            )
            self._buildingblock = BuildingBlock(bb.json()[0])
        return self._buildingblock

    @property
    def buildingblocktop(self):
        """Top building block."""
        if self._buildingblocktop is not None:
            return self._buildingblocktop
        if self.fd_type == 45:  # CW
            prmtrtop = self.json["tubularsection"] + 1
            bbtop = self.api.geo_api.send_request(
                url_data_type="buildingblocks",
                url_params={"id": prmtrtop},
            )
            self._buildingblocktop = BuildingBlock(bbtop.json()[0])
        return self._buildingblocktop

    @property
    def wall_thickness(self) -> List[float]:
        """Wall thickness."""
        wt = [self.buildingblock.wall_thickness]
        if self.buildingblocktop is not None:
            wt.append(self.buildingblocktop.wall_thickness)
        return wt

    @property
    def height(self) -> float:
        """Height of the detail."""
        return self.buildingblock.height if self.buildingblock.height is not None else 0

    @property
    def marker(self):
        if self.fd_type in (36, 40, "Hook"):  # BL, ITS, Hook
            return {
                "mode": "markers",
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
                "radius": 4,
                "hovertext": "<br>".join(
                    [
                        "<i>" + str(self.title) + "</i>",
                        str(self.fd_type),
                        "x: " + str(self.position.x),
                        "y: " + str(self.position.y),
                        "z: " + str(self.position.z),
                        "Fatigue life in: " + str(self.fatiguelifein),
                        "Fatigue life out: " + str(self.fatiguelifeout),
                        "SN curve in: " + str(self.json["sncurvein"]),
                        "SN curve out: " + str(self.json["sncurveout"]),
                    ]
                ),
            }
        else:
            return None

    @property
    def line(self):
        if self.fd_type in (45, 43, 38):  # CW, ST, types of tubular sections
            return {
                "mode": "lines",
                "x": [
                    -self.buildingblock.top_outer_diameter / 2,
                    self.buildingblock.top_outer_diameter / 2,
                ],
                "y": [
                    -self.buildingblock.top_outer_diameter / 2,
                    self.buildingblock.top_outer_diameter / 2,
                ],
                "z": [
                    self.position.z + self.buildingblock.height,
                    self.position.z + self.buildingblock.height,
                ],
                "hovertext": "<br>".join(
                    [
                        "<i>" + str(self.title) + "</i>",
                        str(self.fd_type),
                        "x: " + str(self.position.x),
                        "y: " + str(self.position.y),
                        "z: " + str(self.position.z),
                        "Fatigue life in: " + str(self.fatiguelifein),
                        "Fatigue life out: " + str(self.fatiguelifeout),
                        "SN curve in: " + str(self.json["sncurvein"]),
                        "SN curve out: " + str(self.json["sncurveout"]),
                    ]
                ),
            }
        elif self.fd_type == 41:  # Longitudinal weld
            return {
                "mode": "lines",
                "x": [
                    self.position.y,
                    self.position.y,
                ],
                "z": [self.position.z, self.position.z + self.buildingblock.height],
                "hovertext": "<br>".join(
                    [
                        "<i>" + str(self.title) + "</i>",
                        str(self.fd_type),
                        "x: " + str(self.position.x),
                        "y: " + str(self.position.y),
                        "z: " + str(self.position.z),
                        "height: " + str(self.height),
                        "Fatigue life in: " + str(self.fatiguelifein),
                        "Fatigue life out: " + str(self.fatiguelifeout),
                        "SN curve in: " + str(self.json["sncurvein"]),
                        "SN curve out: " + str(self.json["sncurveout"]),
                    ]
                ),
            }
        else:
            return None

    def as_dict(self, identify_sncurves: bool = False) -> Dict[str, Any]:
        """Returns the fatigue detail description as a dictionary."""
        if identify_sncurves:
            _as_dict = {
                "detailtype": self.fd_type,
                "title": self.title,
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
                "site": self.projectsite,
                "asset": self.asset,
                "modeldefinition": self.modeldefinition,
                "subassembly": self.subassembly_name[-2:],
                "fatiguelifein": self.fatiguelifein,
                "fatiguelifeout": self.fatiguelifeout,
                "sncurvein": (
                    self.sncurves["sncurvein"] if "sncurvein" in self.sncurves else None
                ),
                "sncurveout": (
                    self.sncurves["sncurveout"]
                    if "sncurveout" in self.sncurves
                    else None
                ),
                "description": self.description if self.description else "-",
                "scfin": self.scfin,
                "scfout": self.scfout,
                "materialsafetyfactor": (
                    self.materialsafetyfactor
                    if "materialsafetyfactor" in self.json
                    else None
                ),
                "scaleeffect": self.scaleeffect if "scaleeffect" in self.json else None,
            }
        else:
            _as_dict = {
                "detailtype": self.fd_type,
                "title": self.title,
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
                "site": self.projectsite,
                "asset": self.asset,
                "subassembly": self.subassembly_name[-2:],
                "fatiguelifein": self.fatiguelifein,
                "fatiguelifeout": self.fatiguelifeout,
                "sncurvein": (
                    self.json["sncurvein"] if "sncurvein" in self.json else None
                ),
                "sncurveout": (
                    self.json["sncurveout"] if "sncurveout" in self.json else None
                ),
                "description": self.description if self.description else "-",
                "scfin": self.scfin,
                "scfout": self.scfout,
                "materialsafetyfactor": (
                    self.materialsafetyfactor
                    if "materialsafetyfactor" in self.json
                    else None
                ),
                "scaleeffect": self.scaleeffect if "scaleeffect" in self.json else None,
            }
        return _as_dict

    def as_df(self) -> DataFrame:
        """Returns the fatigue detail description as a DataFrame."""
        d = self.as_dict()
        del d["title"]
        df = DataFrame.from_dict(d, orient="index", columns=[self.title])
        return df

    def _repr_html_(self):
        return self.as_df().to_html()

    def __str__(self):
        msg = [
            f"{self.title} (type: {self.fd_type}) ",
            f"life IN: {self.fatiguelifein} - ",
            f"life OUT: {self.fatiguelifeout}",
        ]
        return "".join(msg)

    def __repr__(self):
        return self.__str__()


class FatigueSubAssembly:
    """Class storing/operating fatigue data related to specific subassembly-turbine cobination.

    Each subassembly data requires an API call to the FatigueAPI to retrieve the
    necessary data through Owimetadatabase. FatigueSubAssembly instances
    should be created by the FatigueAPI class.

    :param api: FatigueAPI instance.
    :param id: Subassembly ID.
    :param title: The title of the subassembly.
    :param description: The description of the subassembly.
    :param position: The position of the subassembly.
    :param sa_type: The type of the subassembly ('TP', 'MP', 'TW', etc.).
    :param source: The source of the subassembly.
    :param asset: The parent turbine of the subassembly (e.g. 'BBA01').
    :param subassembly: The SubAssembly object.
    :param fatiguedetails: A list of FatigueDetail instances.
    :param height: The height of the subassembly.
    :param color: The color-code of the subassembly (for plotting purpose).
    :param absolute_bottom: The absolute bottom of the subassembly.
    :param absolute_top: The absolute top of the subassembly.
    :param properties: A dictionary of the subassembly properties.
    """

    def __init__(
        self,
        json: Dict[str, Union[None, str, np.int64, np.float64]],
        api_object=None,
    ) -> None:
        """Constructor for the FatigueSubAssembly class.

        :param json: The JSON object containing the geometry data of the subassembly.
        :param api_object: The FatigueAPI instance that created the FatigueSubAssembly instance.
        """
        self.api = api_object

        self.id = json["id"]
        self.title = json["title"]
        self.description = json["description"]
        # self.buildingblock = json['buildingblock']
        self.position = Position(
            x=json["x_position"],
            y=json["y_position"],
            z=json["z_position"],
            reference_system=json["vertical_position_reference_system"],
        )
        self.sa_type = json["subassembly_type"]
        self.source = json["source"]
        self._asset = json["asset"]

        self.turbine = self.asset
        self.fds = None
        self._subassembly = None

        materials = self.api.geo_api.send_request(
            url_data_type="materials", url_params={}
        )
        self._materials = [Material(m) for m in materials.json()]

    @property
    def asset(self):
        """Turbine name."""
        # TODO: This has to be worked out properly.
        # ! The LocationsAPI class is not mature yet.
        # ! This is how I would make it...
        # ! req = _make_request(
        # !     api=self.api,
        # !     url='%s/locations/routes/assetlocation',
        # !     params={'id': self._asset}
        # ! )
        # ! if req.status_code != 200:
        # !     return None
        # ! else:
        # !     return req.json()[0]['title']

        return self.title.split("_")[0]

    @property
    def subassembly(self):
        """Subassembly object."""
        self._subassembly = self.api.geo_api.get_subassembly_objects(
            turbine=self.asset,
            subassembly=self.sa_type,
        )
        return list(self._subassembly.values())[0]

    @property
    def color(self):
        """Color for the subassembly."""
        return PLOT_SETTINGS_SUBASSEMBLY[self.sa_type]["color"]

    @property
    def height(self) -> float:
        """Height of the subassembly."""
        height = 0
        for fd in self.fatiguedetails:
            if fd.fd_type == 45:  # CircumferentialWeld type
                if fd.buildingblock.height:
                    height += fd.buildingblock.height
        return height

    @property
    def fatiguedetails(self) -> List[FatigueDetail]:
        """Fatigue details of the subassembly."""
        if self.fds:
            return self.fds

        if self.api is None:
            raise ValueError("No API configured.")
        else:
            fds = self.api.send_request(
                url_data_type="fatiguedetail",
                url_params={"title__icontains": self.title},
            )
            if fds.json():
                if len(fds.json()) > 0:
                    self.fds = [
                        FatigueDetail(
                            fd, api_object=self.api, subassembly=self.subassembly
                        )
                        for fd in fds.json()
                    ]
                    return self.fds
                else:
                    return None
                    # raise ValueError('No fatigue details found.')
            else:
                return None
                # raise ValueError('No fatigue details found.')

    def plotly(
        self,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        x_step: float = 10000.0,
        showlegend: bool = True,
        showmudline: bool = True,
        showplot: bool = True,
    ) -> Dict[str, Any]:
        """Use plotly to plot the subassembly."""
        fig_dict = {"data": [], "layout": {}}

        sub_data, sub_layout = self.subassembly.plotly(x_offset)
        fig_dict["data"].extend(sub_data)
        fig_dict["layout"] = deepcopy(sub_layout)

        markers = []
        lines = []
        if self.fatiguedetails is not None:
            for fd in self.fatiguedetails:
                if fd.fatiguelifein is None and fd.fatiguelifeout is not None:
                    fatlife = fd.fatiguelifeout
                elif fd.fatiguelifein is not None and fd.fatiguelifeout is None:
                    fatlife = fd.fatiguelifein
                elif fd.fatiguelifein is not None and fd.fatiguelifeout is not None:
                    fatlife = np.min([fd.fatiguelifein, fd.fatiguelifeout])
                else:
                    fatlife = 99999999
                    # continue
                if fd.marker:
                    marker_dict = {
                        "x": [x_offset + fd.marker["y"]],  # Here it uses y instead of x
                        "y": [fd.marker["z"] + self.position.z],
                        "mode": "markers",
                        "marker": {
                            "sizemode": "area",
                            "sizeref": fatlife / 15**2,
                            "size": [6],
                            "color": FATIGUE_DETAILS_COLORS[fd.fd_type],
                        },
                        "hovertext": fd.marker["hovertext"]
                        + "<br>z_pos: "
                        + str(self.position.z),
                        "hoverinfo": "text",
                        "name": fd.title,
                        "showlegend": showlegend,
                    }
                    markers.append(marker_dict)
                elif fd.line:
                    line_dict = {
                        "x": [x_offset + x for x in fd.line["x"]],
                        "y": [z + self.position.z for z in fd.line["z"]],
                        "mode": "lines",
                        "line": {
                            "width": 1,
                            "color": FATIGUE_DETAILS_COLORS[fd.fd_type],
                        },
                        "hovertext": fd.line["hovertext"],
                        "hoverinfo": "text",
                        "name": fd.title,
                        "showlegend": showlegend,
                    }
                    lines.append(line_dict)

        fig_dict["data"].extend(markers)
        fig_dict["data"].extend(lines)

        if showmudline and self.sa_type == "MP":
            elevation_req = self.api.loc_api.send_request(
                url_data_type="assetlocations",
                url_params={"title": self.asset},
            )
            elevation = elevation_req.json()[0]["elevation"]
            mudline_dict = {
                "x": [x_offset - x_step / 2, x_offset + x_step / 2],
                "y": [elevation * 1000] * 2,
                "mode": "lines",
                "name": "Mudline",
                "hoverinfo": "text",
                "hovertext": self.asset
                + " mudline elevation: "
                + str(np.round(elevation, 1))
                + "m",
                "showlegend": False,
                "line": {"color": "SaddleBrown", "width": 4},
            }
            fig_dict["data"].append(mudline_dict)
            waterlevel_dict = {
                "x": [x_offset - x_step / 2, x_offset + x_step / 2],
                "y": [0, 0],
                "fill": "tonexty",
                "mode": "lines",
                "name": "Water level",
                "hoverinfo": "text",
                "hovertext": "Water level",
                "showlegend": False,
                "line": {"color": "DodgerBlue", "width": 0.5},
            }
            fig_dict["data"].append(waterlevel_dict)
        if showplot:
            fig = go.Figure(fig_dict["data"], layout=fig_dict["layout"])
            fig.show()
        return fig_dict

    def as_df(
        self, include_absolute_postion: bool = True, identify_sncurves: bool = False
    ) -> DataFrame:
        """Returns the subassembly fatigue data as a DataFrame."""
        df = DataFrame()
        out = []
        for fd in self.fatiguedetails:
            out.append(fd.as_dict(identify_sncurves=identify_sncurves))
        df = DataFrame(out)
        df = df.set_index("title")
        df = df.sort_values("z", ascending=False)
        cols_at_end = ["description"]
        df = df[
            [c for c in df if c not in cols_at_end]
            + [c for c in cols_at_end if c in df]
        ]
        if include_absolute_postion:
            df["absolute_position, m"] = (df["z"] + self.position.z) / 1000
        return df

    @property
    def absolute_bottom(self) -> float:
        """Absolute bottom."""
        temp_df = self.as_df(include_absolute_postion=True)
        return temp_df["absolute_position, m"].iloc[-1]

    @property
    def absolute_top(self) -> float:
        """Absolute top."""
        temp_df = self.as_df(include_absolute_postion=True)
        temp_df.dropna(inplace=True, how="any", axis=0)  # Drop all masses etc
        return round(
            (
                temp_df["absolute_position, m"].iloc[0]
                + temp_df["height"].iloc[0] / 1000
            ),
            3,
        )

    @property
    def properties(self) -> Dict[str, float]:
        """Subassembly properties."""
        property_dict = {"height": self.height}
        return property_dict

    def _repr_html_(self):
        html_str = self.as_df()._repr_html_()
        return html_str

    def __str__(self):
        return f"{self.title}"

    def __repr__(self):
        return self.__str__()
