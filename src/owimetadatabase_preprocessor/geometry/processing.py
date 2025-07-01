"Module containing the processing functions for the geometry data."

import typing
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Union, cast

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.geometry.structures import DataSA, SubAssembly
from owimetadatabase_preprocessor.utility.utils import (
    deepcompare,
)  # custom_formatwarning, deepcompare

# warnings.simplefilter("always")
# warnings.formatwarning = custom_formatwarning


ATTR_PROC = [
    "pile_toe",
    "rna",
    "tower",
    "transition_piece",
    "monopile",
    "tw_lumped_mass",
    "tp_lumped_mass",
    "mp_lumped_mass",
    "tp_distributed_mass",
    "mp_distributed_mass",
    "grout",
]
ATTR_SPEC = ["full_structure", "tp_skirt", "substructure"]
ATTR_FULL = [
    "all_tubular_structures",
    "all_distributed_mass",
    "all_lumped_mass",
    "all_turbines",
]


class OWT(object):
    """Class to process the geometry data of a single OWT.

    :param api: API object used to call get_* methods.
    :param materials: Pandas dataframe with the materials data.
    :param sub_assemblies: Dictionary of the subassemblies.
    :param tw_sub_assemblies: Pandas dataframe with the tower subassemblies data for a given turbine.
    :param tp_sub_assemblies: Pandas dataframe with the transition piece subassemblies data for a given turbine.
    :param mp_sub_assemblies: Pandas dataframe with the monopile subassemblies data for a given turbine.
    :param tower_base: Elevation of the OWT tower base in mLAT.
    :param pile_head: Elevation of the pile head in mLAT.
    :param water_depth: Water depth in mLAT.
    :param pile_toe: Elevation of the pile toe in mLAT.
    :param rna: Pandas dataframe with the RNA data.
    :param tower: Pandas dataframe with the tower data.
    :param transition_piece: Pandas dataframe with the transition piece data.
    :param monopile: Pandas dataframe with the monopile data.
    :param tw_lumped_mass: Pandas dataframe with the lumped masses data for the tower.
    :param tp_lumped_mass: Pandas dataframe with the lumped masses data for the transition piece.
    :param mp_lumped_mass: Pandas dataframe with the lumped masses data for the monopile.
    :param tp_distributed_mass: Pandas dataframe with the distributed masses data for the transition piece.
    :param mp_distributed_mass: Pandas dataframe with the distributed masses data for the monopile.
    :param grout: Pandas dataframe with the grout data.
    :param full_structure: Pandas dataframe with the full structure data.
    :param tp_skirt: Pandas dataframe with the transition piece skirt data.
    :param substructure: Pandas dataframe with the substructure data.
    """

    _init_proc: bool
    _init_spec_part: bool
    _init_spec_full: bool
    api: Any
    materials: pd.DataFrame
    sub_assemblies: Dict[str, SubAssembly]
    tw_sub_assemblies: Union[pd.DataFrame, None]
    tp_sub_assemblies: Union[pd.DataFrame, None]
    mp_sub_assemblies: Union[pd.DataFrame, None]
    tower_base: Union[np.float64, None]
    pile_head: Union[np.float64, None]
    water_depth: np.float64
    pile_toe: Union[np.float64, None]
    rna: Union[pd.DataFrame, None]
    tower: Union[pd.DataFrame, None]
    transition_piece: Union[pd.DataFrame, None]
    monopile: Union[pd.DataFrame, None]
    tw_lumped_mass: Union[pd.DataFrame, None]
    tp_lumped_mass: Union[pd.DataFrame, None]
    mp_lumped_mass: Union[pd.DataFrame, None]
    tp_distributed_mass: Union[pd.DataFrame, None]
    mp_distributed_mass: Union[pd.DataFrame, None]
    grout: Union[pd.DataFrame, None]
    full_structure: Union[pd.DataFrame, None]
    tp_skirt: Union[pd.DataFrame, None]
    substructure: Union[pd.DataFrame, None]

    def __init__(
        self,
        api: Any,
        materials: Union[pd.DataFrame, bool, np.int64, None],
        subassemblies: Union[pd.DataFrame, bool, np.int64, None],
        location: Union[pd.DataFrame, bool, np.int64, None],
        tower_base: Union[np.float64, None] = None,
        pile_head: Union[np.float64, None] = None,
    ) -> None:
        """Create an instance of the OWT class with the required parameters.

        :param api: API object used to call get_* methods.
        :param materials: Pandas dataframe with the materials data.
        :param subassemblies: Pandas dataframe with the subassemblies data for a given turbine.
        :param location: Pandas dataframe with the location data for a given turbine.
        :param tower_base: Optional: elevation of the OWT tower base in mLAT.
        :param pile_head: Optional: elevation of the pile head in mLAT.
        :return: None
        """
        self._init_proc = False
        self._init_spec_part = False
        self._init_spec_full = False
        self.api = api
        self.materials = materials
        self._set_subassemblies(subassemblies)
        self.tw_sub_assemblies = None
        self.tp_sub_assemblies = None
        self.mp_sub_assemblies = None
        self._set_members()
        for attr in ATTR_PROC:
            setattr(self, attr, None)
        for attr in ATTR_SPEC:
            setattr(self, attr, None)
        self.water_depth = location["elevation"].values[0]
        if not tower_base or not pile_head:
            if "TW" in self.sub_assemblies.keys():
                self.tower_base = self.sub_assemblies["TW"].absolute_bottom
            elif "TP" in self.sub_assemblies.keys():
                self.tower_base = self.sub_assemblies["TP"].absolute_top
            else:
                self.tower_base = None
            if "MP" in self.sub_assemblies.keys():
                self.pile_head = self.sub_assemblies["MP"].absolute_top
            else:
                self.pile_head = None
        else:
            self.tower_base = tower_base
            self.pile_head = pile_head

    def _set_subassemblies(self, subassemblies: pd.DataFrame) -> None:
        """Create a dictionary containing the subassemblies of the OWT.

        :param subassemblies: Pandas dataframe with the subassemblies data for a given turbine.
        :return: None
        """
        subassemblies_types = [
            sa["subassembly_type"] for _, sa in subassemblies.iterrows()
        ]
        subassemblies_list = [
            SubAssembly(self.materials, cast(DataSA, sa.to_dict()), api_object=self.api)
            for _, sa in subassemblies.iterrows()
        ]
        self.sub_assemblies = {
            k: v for (k, v) in zip(subassemblies_types, subassemblies_list)
        }

    def _set_members(self) -> None:
        """Identify and stores in separate data frames each part of the support structure (tower=TW, transition piece=TP,
        monopile=MP).

        :return: None
        """
        for k, v in self.sub_assemblies.items():
            if k == "TW":
                self.tw_sub_assemblies = v.as_df()
            if k == "TP":
                self.tp_sub_assemblies = v.as_df()
            if k == "MP":
                self.mp_sub_assemblies = v.as_df()

    def set_df_structure(self, idx: str) -> pd.DataFrame:
        """Calculate and/or converts geometrical data of subassemblies from the database.

        :param idx: Possible index to identify corresponding subassembly.
        :return: Data frame containing geometry data from database wth z in mLAT system.
        """
        cols = ["OD", "height", "mass", "volume", "wall_thickness", "x", "y", "z"]
        if idx == "tw":
            if self.tw_sub_assemblies is None:
                raise ValueError("Tower subassembly data not found.")
            df_index = self.tw_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tw_sub_assemblies.loc[df_index, cols])
            depth_to = self.tower_base + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        elif idx == "tp":
            if self.tp_sub_assemblies is None:
                raise ValueError("Transition piece subassembly data not found.")
            # We don't take into account the grout, this element will be modelled as a distributed lumped mass.
            df_index = (self.tp_sub_assemblies.index.str.contains(idx)) & (
                ~self.tp_sub_assemblies.index.str.contains("grout")
            )
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols])
            bottom_tp = self.tower_base - df["height"].sum() * 1e-3
            depth_to = bottom_tp + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        elif idx == "mp":
            if self.mp_sub_assemblies is None:
                raise ValueError("Monopile subassembly data not found.")
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.mp_sub_assemblies.loc[df_index, cols])
            toe = self.pile_head - df["height"].sum() * 1e-3
            self.pile_toe = round(toe, 3)
            depth_to = toe + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        else:
            raise ValueError("Unknown index.")
        df["Elevation from [mLAT]"] = depth_from
        df["Elevation to [mLAT]"] = depth_to
        # Round elevations to mm to avoid numerical inconsistencies later when setting altitude values to apply loads.
        df = df.round({"Elevation from [mLAT]": 3, "Elevation to [mLAT]": 3})
        return df

    def process_structure_geometry(self, idx: str) -> pd.DataFrame:
        """Calculate and/or converts geometrical data of subassemblies from the database to use as input for FE models.

        :param idx: Possible index to identify corresponding subassembly.
        :return: Dataframe consisting of the required data to build FE models.
        """
        df = self.set_df_structure(idx)
        df["height"] = pd.to_numeric(df["height"])
        df["wall_thickness"] = pd.to_numeric(df["wall_thickness"])
        df.rename(columns={"wall_thickness": "Wall thickness [mm]"}, inplace=True)
        df.rename(columns={"volume": "Volume [m3]"}, inplace=True)
        d_to = [d.split("/", 1)[0] for d in df["OD"].values]
        d_from = [
            d.split("/", 1)[1] if len(d.split("/", 1)) > 1 else d.split("/", 1)[0]
            for d in df["OD"].values
        ]
        df["Diameter from [m]"] = np.array(d_from, dtype=float) * 1e-3
        df["Diameter to [m]"] = np.array(d_to, dtype=float) * 1e-3
        df["rho [t/m]"] = df["mass"] / df["height"]
        df["Mass [t]"] = df["mass"] * 1e-3
        df["Height [m]"] = df["height"] * 1e-3
        df["Youngs modulus [GPa]"] = 210
        df["Poissons ratio [-]"] = 0.3
        cols = [
            "Elevation from [mLAT]",
            "Elevation to [mLAT]",
            "Height [m]",
            "Diameter from [m]",
            "Diameter to [m]",
            "Volume [m3]",
            "Wall thickness [mm]",
            "Youngs modulus [GPa]",
            "Poissons ratio [-]",
            "Mass [t]",
            "rho [t/m]",
        ]
        return df[cols]

    def process_rna(self) -> None:
        """Set dataframe containing the required properties to model the RNA system.

        :return: None
        """
        if self.tw_sub_assemblies is None:
            raise ValueError("Tower subassembly data not found.")
        rna_index = self.tw_sub_assemblies.index.str.contains("RNA")
        rna = deepcopy(
            self.tw_sub_assemblies.loc[
                rna_index, ["mass", "moment_of_inertia", "x", "y", "z", "description"]
            ]
        )
        mi = rna["moment_of_inertia"].values
        i_xx, i_yy, i_zz = [], [], []
        for m in mi:
            i_xx.append(m["x"] * 1e-3)
            i_yy.append(m["y"] * 1e-3)
            i_zz.append(m["z"] * 1e-3)
        rna["Ixx [tm2]"] = i_xx
        rna["Iyy [tm2]"] = i_yy
        rna["Izz [tm2]"] = i_zz
        rna["Mass [t]"] = rna["mass"] * 1e-3
        rna["X [m]"] = rna["x"] * 1e-3
        rna["Y [m]"] = rna["y"] * 1e-3
        rna["Z [mLAT]"] = self.tower_base + rna["z"] * 1e-3
        rna.rename(columns={"description": "Description"}, inplace=True)
        cols = [
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Mass [t]",
            "Ixx [tm2]",
            "Iyy [tm2]",
            "Izz [tm2]",
            "Description",
        ]
        self.rna = rna[cols]

    def set_df_appurtenances(self, idx: str) -> pd.DataFrame:
        """Set dataframe containing the required properties to model concentrated masses from database subassemblies.

        :param idx: Index to identify corresponding subassembly with possible values: 'TW', 'TP', 'MP'.
        :return: Data frame containing lumped masses data from database with Z coordinates in mLAT system.
        """
        cols = ["mass", "x", "y", "z", "description"]
        if idx == "TW":
            if self.tw_sub_assemblies is None:
                raise ValueError("Tower subassembly data not found.")
            df_index = self.tw_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tw_sub_assemblies.loc[df_index, cols])
            df["Z [mLAT]"] = self.tower_base + df["z"] * 1e-3
        elif idx == "TP":
            if self.tp_sub_assemblies is None:
                raise ValueError("Transition piece subassembly data not found.")
            df_index = self.tp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols + ["height"]])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].isnull()]
            bottom = self.sub_assemblies["TP"].position.z * 1e-3  # m
            df["Z [mLAT]"] = bottom + df["z"] * 1e-3  # m
        elif idx == "MP":
            if self.mp_sub_assemblies is None:
                raise ValueError("Monopile subassembly data not found.")
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.mp_sub_assemblies.loc[df_index, cols + ["height"]])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].isnull()]
            bottom = self.pile_toe
            df["Z [mLAT]"] = bottom + df["z"] * 1e-3
        else:
            raise ValueError("Unknown index.")
        return df

    def process_lumped_masses(self, idx: str) -> pd.DataFrame:
        """Create dataframe containing the required properties to model lumped mass appurtenances. Note that
        if the preprocessor package does not find any appurtenances it'll return an empty dataframe.

        :param idx:  Index to identify corresponding subassembly with possible values: 'TW', 'TP', 'MP'.
        :return: Dataframe.
        """
        df = self.set_df_appurtenances(idx)
        df["Mass [t]"] = df.mass * 1e-3
        df["X [m]"] = df.x * 1e-3
        df["Y [m]"] = df.y * 1e-3
        df.rename(columns={"description": "Description"}, inplace=True)
        cols = ["X [m]", "Y [m]", "Z [mLAT]", "Mass [t]", "Description"]
        return df[cols]

    def set_df_distributed_appurtenances(self, idx: str) -> pd.DataFrame:
        """Set dataframe containing the required properties to model distributed lumped masses from database.

        :param idx: Index to identify corresponding subassembly with possible values: 'TW', 'TP', 'MP'.
        :return: Dataframe containing distributed lumped masses data from database. Z coordinates in mLAT system.
        """
        cols = ["mass", "x", "y", "z", "height", "volume", "description"]
        if idx == "TP":
            if self.tp_sub_assemblies is None:
                raise ValueError("Transition piece subassembly data not found.")
            df_index = self.tp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].notnull()]
            bottom_tp = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom_tp + df["z"] * 1e-3
        elif idx == "MP":
            if self.mp_sub_assemblies is None:
                raise ValueError("Monopile subassembly data not found.")
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.mp_sub_assemblies.loc[df_index, cols])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].notnull()]
            bottom = self.pile_toe
            df["Z [mLAT]"] = bottom + df["z"] * 1e-3
        elif idx == "grout":
            if self.tp_sub_assemblies is None:
                raise ValueError("Transition piece subassembly data not found.")
            df_index = self.tp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].notnull()]
            bottom_tp = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom_tp + df["z"] * 1e-3
        else:
            raise ValueError(
                "Unknown index or non distributed lumped masses located outside the transition piece."
            )
        return df

    def process_distributed_lumped_masses(self, idx: str) -> pd.DataFrame:
        """Create dataframe containing the required properties to model uniformly distributed appurtenances. Note that
        if the preprocessor package does not find any appurtenances it'll return an empty dataframe.

        :param idx: Index to identify corresponding subassembly with possible values: 'TP', 'MP'.
        :return: Dataframe.
        """
        df = self.set_df_distributed_appurtenances(idx)
        df["Mass [t]"] = df["mass"] * 1e-3
        df["X [m]"] = df["x"] * 1e-3
        df["Y [m]"] = df["y"] * 1e-3
        df["Height [m]"] = df["height"] * 1e-3
        df.rename(columns={"volume": "Volume [m3]"}, inplace=True)
        df.rename(columns={"description": "Description"}, inplace=True)
        cols = [
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Height [m]",
            "Mass [t]",
            "Volume [m3]",
            "Description",
        ]
        return df[cols]

    def process_structure(self, option="full") -> None:
        """Set dataframe containing the required properties to model the tower geometry, including the RNA system.

        :param option: Option to process the data for a specific subassembly. Possible values:

            - "full": To process all the data for all subassemblies.
            - "tower": To process only the data for the tower subassembly.
            - "TP": To process only the data for the transition piece subassembly.
            - "monopile": To process only the data for the monopile foundation subassembly.
        :return: None
        """
        self._init_proc = True
        if option == "full":
            self.process_rna()
            self.tower = self.process_structure_geometry("tw")
            self.transition_piece = self.process_structure_geometry("tp")
            self.monopile = self.process_structure_geometry("mp")
            self.tw_lumped_mass = self.process_lumped_masses("TW")
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")
            self.grout = self.process_distributed_lumped_masses("grout")
        elif option == "TW":
            self.process_rna()
            self.tower = self.process_structure_geometry("tw")
            self.tw_lumped_mass = self.process_lumped_masses("TW")
        elif option == "TP":
            self.transition_piece = self.process_structure_geometry("tp")
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
            self.grout = self.process_distributed_lumped_masses("grout")
        elif option == "MP":
            self.monopile = self.process_structure_geometry("mp")
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")

    @staticmethod
    def can_adjust_properties(row: pd.Series) -> pd.Series:
        """Recalculation of can properties based on section properties and can elevations: height [m],
        volume [m3], mass [t], rho [t/m].

        :param row: Original can properties.
        :return: Pandas series of recalculated can properties.
        """
        density = row["Mass [t]"] / row["Volume [m3]"]
        height = row["Elevation from [mLAT]"] - row["Elevation to [mLAT]"]
        r1 = row["Diameter from [m]"] / 2
        r2 = row["Diameter to [m]"] / 2
        volume_out = 1 / 3 * np.pi * (r1**2 + r1 * r2 + r2**2) * height
        wall_thickness = row["Wall thickness [mm]"] * 1e-3
        r1 = r1 - wall_thickness
        r2 = r2 - wall_thickness
        volume_in = 1 / 3 * np.pi * (r1**2 + r1 * r2 + r2**2) * height
        volume = volume_out - volume_in
        mass = volume * density
        rho_m = mass / height
        can_properties = pd.Series(
            data=[height, volume, mass, rho_m],
            index=["Height [m]", "Volume [m3]", "Mass [t]", "rho [t/m]"],
        )
        return can_properties

    def can_modification(
        self,
        df: pd.DataFrame,
        altitude: Union[np.float64, None],
        position: str = "bottom",
    ) -> pd.DataFrame:
        """Change can properties based on the altitude.

        :param df: Dataframe containing the can properties.
        :param altitude: Altitude in mLAT.
        :param position: Position of the can with respect to the altitude with possible values: "bottom" or "top".
        :return: Dataframe with the modified can properties.
        """
        if position == "bottom":
            ind = -1
            _col = " to "
        else:
            ind = 0
            _col = " from "
        df.loc[df.index[ind], "Elevation" + _col + "[mLAT]"] = altitude  # type: ignore
        elevation = [
            df.iloc[ind]["Elevation from [mLAT]"],
            df.iloc[ind]["Elevation to [mLAT]"],
        ]
        diameters = [df.iloc[ind]["Diameter from [m]"], df.iloc[ind]["Diameter to [m]"]]
        df.loc[df.index[ind], "Diameter" + _col + "[m]"] = np.interp(
            [altitude], elevation, diameters  # type: ignore
        )[0]
        cols = ["Height [m]", "Volume [m3]", "Mass [t]", "rho [t/m]"]
        df.loc[df.index[ind], cols] = self.can_adjust_properties(df.iloc[ind])
        return df

    def assembly_tp_mp(self) -> None:
        """Process TP structural item to assembly with MP foundation ensuring continuity. TP skirt is processed
        as well.

        :return: None
        """
        self._init_spec_part = True
        if (self.transition_piece is not None) and (self.monopile is not None):
            mp_head = self.pile_head
            tp = self.transition_piece
            df = deepcopy(tp[tp["Elevation from [mLAT]"] > mp_head])
            if df.loc[df.index[0], "Elevation to [mLAT]"] != mp_head:
                # Not bolted connection (i.e. Rentel) preprocessing needed
                tp1 = self.can_modification(df, mp_head, position="bottom")
                self.substructure = pd.concat([tp1, deepcopy(self.monopile)])
            else:
                # Bolted connection, nothing to do
                self.substructure = pd.concat([df, deepcopy(self.monopile)])
            df = deepcopy(tp[tp["Elevation to [mLAT]"] < mp_head])
            self.tp_skirt = self.can_modification(df, mp_head, position="top")
        else:
            raise TypeError("TP or MP items need to be processed before!")

    def assembly_full_structure(self) -> None:
        """Process the full structure of the OWT: tower + tp combiantion with monopile.

        :return: None
        """
        self._init_spec_full = True
        if self.substructure is not None:
            if self.tower is not None:
                self.full_structure = pd.concat([self.tower, self.substructure])
            else:
                raise TypeError("Tower needs to be processed before!")
        else:
            raise TypeError("Substructure needs to be processed before!")

    def extend_dfs(self) -> None:
        """Extend the dataframes with the subassembly columns.

        :return: None
        """
        for attr in ATTR_PROC:
            df = getattr(self, attr)
            if df is not None:
                if "tower" in attr or "tw_" in attr or "rna" in attr:
                    df["Subassembly"] = "TW"
                    setattr(self, attr, df)
                elif "tp_" in attr or "transition" in attr or "grout" in attr:
                    df["Subassembly"] = "TP"
                    setattr(self, attr, df)
                elif "mp_" in attr or "monopile" in attr:
                    df["Subassembly"] = "MP"
                    setattr(self, attr, df)
        if "TP" in self.sub_assemblies.keys() and "MP" in self.sub_assemblies.keys():
            self.assembly_tp_mp()
        else:
            self._init_spec_part = True
            self.tp_skirt = None
        if "TW" in self.sub_assemblies.keys():
            self._init_spec_full = True
            if self.substructure is not None:
                self.assembly_full_structure()
            else:
                self.full_structure = None
        else:
            self.full_structure = None
            self._init_spec_full = True

    @typing.no_type_check
    def transform_monopile_geometry(
        self,
        cutoff_point: np.floating = np.nan,
    ) -> pd.DataFrame:
        """Returns a dataframe with the monopile geometry with the mudline as reference

        :param cutoff_point: Depth from the mudline to cut the monopile geometry.
        :return: Dataframe with the monopile geometry.
        """
        toe_depth_lat = self.sub_assemblies["MP"].position.z
        penetration = -((1e-3 * toe_depth_lat) - self.water_depth)
        pile = pd.DataFrame()
        if self.mp_sub_assemblies is not None:
            df = self.mp_sub_assemblies.copy()
        else:
            raise ValueError("Monopile subassembly data not found.")
        df.reset_index(inplace=True)
        for i, row in df.iterrows():
            if i != 0:
                pile.loc[i, "Elevation from [m]"] = (
                    penetration - 1e-3 * df["z"].iloc[i - 1]
                )
                pile.loc[i, "Elevation to [m]"] = penetration - 1e-3 * row["z"]
                pile.loc[i, "Pile material"] = (
                    self.sub_assemblies["MP"].bb[0].material.title
                )
                pile.loc[i, "Pile material submerged unit weight [kN/m3]"] = (
                    1e-2 * self.sub_assemblies["MP"].bb[0].material.density - 10
                )
                pile.loc[i, "Wall thickness [mm]"] = row["wall_thickness"]
                bot_od = row["OD"].split("/")[0] if "/" in row["OD"] else row["OD"]
                top_od = row["OD"].split("/")[1] if "/" in row["OD"] else row["OD"]
                pile.loc[i, "Diameter [m]"] = (
                    1e-3 * 0.5 * (float(bot_od) + float(top_od))
                )
                pile.loc[i, "Youngs modulus [GPa]"] = (
                    self.sub_assemblies["MP"].bb[0].material.young_modulus
                )
                pile.loc[i, "Poissons ratio [-]"] = (
                    self.sub_assemblies["MP"].bb[0].material.poisson_ratio
                )
        if not np.math.isnan(cutoff_point):
            pile = pile.loc[pile["Elevation to [m]"] > cutoff_point].reset_index(
                drop=True
            )
            pile.loc[0, "Elevation from [m]"] = cutoff_point
        return pile

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            comp = deepcompare(self, other)
            assert comp[0], comp[1]
        elif isinstance(other, dict):
            comp = deepcompare(self.__dict__, other)
            assert comp[0], comp[1]
        else:
            assert False, "Comparison is not possible due to incompatible types!"
        return comp[0]

    def __getattribute__(self, name: str) -> object:
        if name in ATTR_PROC and not self._init_proc:
            warnings.warn(
                f"Attribute '{name}' accessed before processing. \
                    Run process_structure() first if you want to process values."
            )
        elif name in ATTR_SPEC and not self._init_spec_part:
            warnings.warn(
                f"Attribute '{name}' accessed before processing. \
                    Run assembly_tp_mp() first if you want to process values."
            )
        elif name in ATTR_SPEC and not self._init_spec_full:
            warnings.warn(
                f"Attribute '{name}' accessed before processing. \
                    Run assembly_full_structure() first if you want to process values."
            )
        return object.__getattribute__(self, name)


class OWTs(object):
    """Class to process the geometry data of multiple OWTs.

    :param owts: List of OWT objects.
    :param api: API object used to call get_* methods.
    :param materials: Pandas dataframe with the materials data.
    :param sub_assemblies: Dictionary of dictionaries of the subassemblies for each turbine.
    :param tower_base: Dictionary of the elevation of the OWT tower base in mLAT for each turbine.
    :param pile_head: Dictionary of the elevation of the pile head in mLAT for each turbine.
    :param water_depth: Dictionary of the water depth in mLAT for each turbine.
    :param tw_sub_assemblies: Dataframe of the tower subassemblies data from each turbine.
    :param tp_sub_assemblies: Dataframe of the transition piece subassemblies data from each turbine.
    :param mp_sub_assemblies: Dataframe of the monopile subassemblies data from each turbine.
    :param pile_toe: Dataframe of the elevation of the pile toe in mLAT from each turbine.
    :param rna: Dataframe of the RNA data from each turbine.
    :param tower: Dataframe of the tower data from each turbine.
    :param transition_piece: Dataframe of the transition piece data from each turbine.
    :param monopile: Dataframe of the monopile data from each turbine.
    :param tw_lumped_mass: Dataframe of the lumped masses data of the tower from each turbine.
    :param tp_lumped_mass: Dataframe of the lumped masses data of the transition piece from each turbine.
    :param mp_lumped_mass: Dataframe of the lumped masses data of the monopile from each turbine.
    :param tp_distributed_mass: Dataframe of the distributed masses data of the transition piece from each turbine.
    :param mp_distributed_mass: Dataframe of the distributed masses data of the monopile from each turbine.
    :param grout: Dataframe of the grout data from each turbine.
    :param full_structure: Dataframe of the full structure data from each turbine.
    :param tp_skirt: Dataframe of the transition piece skirt data from each turbine.
    :param substructure: Dataframe of the substructure data from each turbine.
    :param all_turbines: Dataframe of the general geometry data from each turbine.
    :param all_tubular_structures: Dataframe of the tubular structures data from each turbine.
    :param all_distributed_mass: Dataframe of the distributed masses data from each turbine.
    :param all_lumped_mass: Dataframe of the lumped masses data from each turbine.
    """

    def __init__(
        self,
        turbines: List[str],
        owts: List[OWT],
    ) -> None:
        """Create an instance of the OWTs class with the required parameters.

        :param turbines: List of turbine titles.
        :param owts: List of OWT objects.
        :return:
        """
        self.owts = {k: v for k, v in zip(turbines, owts)}
        self.api = self.owts[turbines[0]].api
        self.materials = self.owts[turbines[0]].materials
        for attr in ["sub_assemblies", "tower_base", "pile_head", "water_depth"]:
            dict_ = {
                k: getattr(owt, attr) for k, owt in zip(turbines, self.owts.values())
            }
            setattr(self, attr, dict_)
        for attr in ["tw_sub_assemblies", "tp_sub_assemblies", "mp_sub_assemblies"]:
            sa_turb_list = [
                getattr(owt, attr)
                for owt in self.owts.values()
                if getattr(owt, attr) is not None
            ]
            if sa_turb_list == []:
                df = None
            else:
                df = pd.concat(sa_turb_list)
            setattr(self, attr, df)
        for attr in ATTR_PROC:
            setattr(self, attr, [])
        for attr in ATTR_SPEC:
            setattr(self, attr, [])
        for attr in ATTR_FULL:
            setattr(self, attr, [])
        self._init = False

    def _concat_list(self, attr_list: List[str]) -> None:
        """Internal method to concatenate lists of dataframes for attributes.

        :param attr_list: List of attributes to concatenate.
        :return: None
        """
        for attr in attr_list:
            attr_val = getattr(self, attr)
            if attr_val is None or attr_val == [] or all(v is None for v in attr_val):
                df = None
            else:
                df = pd.concat(attr_val)
            setattr(self, attr, df)

    def _assembly_turbine(self) -> None:
        """Method to assemble general geometry data of all specified turbines.

        :return: None
        """
        cols = [
            "Turbine name",
            "Water depth [m]",
            "Monopile toe [m]",
            "Monopile head [m]",
            "Tower base [m]",
            "Monopile height [m]",
            "Monopile mass [t]",
            "Transition piece height [m]",
            "Transition piece mass [t]",
            "Tower height [m]",
            "Tower mass [t]",
            "RNA mass [t]",
        ]
        df_list = []
        for attr in ATTR_PROC:
            df = getattr(self, attr)
            # if df is None:
            #     raise ValueError(f"Attribute '{attr}' is None.")
        for turb in self.owts.keys():
            df_list.append(
                [
                    turb,
                    self.water_depth[turb],
                    self.pile_toe[turb],  # type: ignore
                    self.pile_head[turb],
                    self.tower_base[turb],
                    (
                        self.owts[turb].monopile["Height [m]"].sum()
                        if self.owts[turb].monopile is not None
                        else None
                    ),
                    (
                        (
                            self.owts[turb].monopile["Mass [t]"].sum()
                            + self.owts[turb].mp_distributed_mass["Mass [t]"].sum()
                            + self.owts[turb].mp_lumped_mass["Mass [t]"].sum()
                        )
                        if self.owts[turb].monopile is not None
                        else None
                    ),
                    (
                        self.owts[turb].transition_piece["Height [m]"].sum()
                        if self.owts[turb].transition_piece is not None
                        else None
                    ),
                    (
                        (
                            self.owts[turb].transition_piece["Mass [t]"].sum()
                            + self.owts[turb].tp_distributed_mass["Mass [t]"].sum()
                            + self.owts[turb].tp_lumped_mass["Mass [t]"].sum()
                            + self.owts[turb].grout["Mass [t]"].sum()
                        )
                        if self.owts[turb].transition_piece is not None
                        else None
                    ),
                    (
                        self.owts[turb].tower["Height [m]"].sum()
                        if self.owts[turb].tower is not None
                        else None
                    ),
                    (
                        (
                            self.owts[turb].tower["Mass [t]"].sum()
                            + self.owts[turb].tw_lumped_mass["Mass [t]"].sum()
                        )
                        if self.owts[turb].tower is not None
                        else None
                    ),
                    (
                        self.owts[turb].rna["Mass [t]"].sum()
                        if self.owts[turb].rna is not None
                        else None
                    ),
                ]
            )
        df = pd.DataFrame(df_list, columns=cols)
        self.all_turbines = df.round(2)

    def process_structures(self) -> None:
        """Set dataframes containing the required properties to model the tower geometry, including the RNA system.

        :return: None
        """
        attr_list = ATTR_PROC + ATTR_SPEC + ATTR_FULL
        attr_list.remove("all_turbines")
        if self._init:
            return
        self._init = True
        for owt in self.owts.values():
            if not len(owt.sub_assemblies) == 3:
                for sa in owt.sub_assemblies.keys():
                    owt.process_structure(option=sa)
            else:
                owt.process_structure()
            owt.extend_dfs()
            for attr in attr_list:
                if attr == "pile_toe":
                    self.pile_toe.append(getattr(owt, attr))
                elif attr == "all_tubular_structures":
                    self.all_tubular_structures.extend(
                        [owt.tower, owt.transition_piece, owt.monopile]
                    )
                elif attr == "all_distributed_mass":
                    self.all_distributed_mass.extend(
                        [
                            owt.tp_distributed_mass,
                            owt.grout,
                            owt.mp_distributed_mass,
                        ]
                    )
                elif attr == "all_lumped_mass":
                    if isinstance(owt.rna, pd.DataFrame):
                        cols = [
                            "X [m]",
                            "Y [m]",
                            "Z [mLAT]",
                            "Mass [t]",
                            "Description",
                            "Subassembly",
                        ]
                        rna_ = owt.rna[cols]
                    else:
                        rna_ = owt.rna
                    self.all_lumped_mass.extend(
                        [
                            rna_,
                            owt.tw_lumped_mass,
                            owt.tp_lumped_mass,
                            owt.mp_lumped_mass,
                        ]
                    )
                else:
                    attr_val = getattr(self, attr)
                    owt_attr_val = getattr(owt, attr)
                    attr_val.append(owt_attr_val)
        attr_list.remove("pile_toe")
        self.pile_toe = {k: v for k, v in zip(self.owts.keys(), self.pile_toe)}  # type: ignore
        self._concat_list(attr_list)
        self._assembly_turbine()

    def select_owt(self, turbine: Union[str, int]) -> OWT:
        """Select OWT object from the OWTs object.

        :param turbine: Title of the turbine or itss index in the original list of turbine titles (from get method).
        :return: OWT object.
        """
        if isinstance(turbine, int):
            return self.owts[list(self.owts.keys())[turbine]]
        elif isinstance(turbine, str):
            return self.owts[turbine]
        else:
            raise ValueError(
                "You must specify a single turbine title or \
                its index from the the get method input turbine list."
            )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            comp = deepcompare(self, other)
            assert comp[0], comp[1]
        elif isinstance(other, dict):
            comp = deepcompare(self.__dict__, other)
            assert comp[0], comp[1]
        else:
            assert False, "Comparison is not possible due to incompatible types!"
        return comp[0]

    def __getattribute__(self, name):
        if name in ATTR_PROC + ATTR_SPEC + ATTR_FULL and not self._init:
            warnings.warn(
                f"Attribute '{name}' accessed before processing. \
                    Run process_structures() first if you want to process values."
            )
        return object.__getattribute__(self, name)
