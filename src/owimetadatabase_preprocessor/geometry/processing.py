"Module containing the processing functions for the geometry data."

from copy import deepcopy
from typing import Dict, Union

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.geometry.structures import SubAssembly


class OWT(object):
    def __init__(
        self,
        api,
        materials: pd.DataFrame,
        subassemblies: pd.DataFrame,
        tower_base=None,
        pile_head=None,
    ) -> None:
        """Get all subassemblies for a given Turbine.

        :param subassemblies: Pandas dataframe with the subassemblies data for a given turbine.
        :param tower_base: Elevation of the OWT tower base in mLAT.
        :param pile_head: Elevation of the pile head in mLAT.

        :return:
        """
        self.api = api
        self.materials = materials
        self._set_subassemblies(subassemblies)
        self.tower_sub_assemblies = None
        self.tp_sub_assemblies = None
        self.mp_sub_assemblies = None
        self._set_members()
        self.tower_base = tower_base
        self.pile_head = pile_head
        self.pile_toe = None
        self.rna = None
        self.tower_geometry = None
        self.transition_piece = None
        self.monopile = None
        self.substructure = None
        self.tp_skirt = None
        self.tower_lumped_mass = None
        self.tp_lumped_mass = None
        self.mp_lumped_mass = None
        self.tp_distributed_mass = None
        self.mp_distributed_mass = None

    def _set_members(self) -> None:
        """Identifies and stores in separate data frames each part of the support structure (tower=TW, transition piece=TP,
        monopile=MP).
        """
        for k, v in self.sub_assemblies.items():
            if k == "TW":
                self.tower_sub_assemblies = v.as_df()
            if k == "TP":
                self.tp_sub_assemblies = v.as_df()
            if k == "MP":
                self.mp_sub_assemblies = v.as_df()

    def _set_subassemblies(self, subassemblies: pd.DataFrame) -> None:
        """Creates a dictionary containing the subassemblies of the OWT."""
        subassemblies_types = [
            sa["subassembly_type"] for _, sa in subassemblies.iterrows()
        ]
        subassemblies_list = [
            SubAssembly(self.materials, sa.to_dict(), api_object=self.api)
            for _, sa in subassemblies.iterrows()
        ]
        self.sub_assemblies = {
            k: v for (k, v) in zip(subassemblies_types, subassemblies_list)
        }

    def set_df_structure(self, idx: str) -> pd.DataFrame:
        """Calculates and/or converts geometrical data of subassemblies from the database.

        :param idx: Possible index to identify corresponding subassembly.
        :return: Data frame containing geometry data from database wth z in mLAT system.
        """
        cols = ["OD", "height", "mass", "volume", "wall_thickness", "x", "y", "z"]
        if idx == "tw":
            df_index = self.tower_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tower_sub_assemblies.loc[df_index, cols])            
            depth_to = self.tower_base + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        elif idx == "tp":
            # We don't take into account the grout, this element will be modelled as a distributed lumped mass.
            df_index = (self.tp_sub_assemblies.index.str.contains(idx)) & (
                ~self.tp_sub_assemblies.index.str.contains("grout")
            )
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols])
            bottom_tp = self.tower_base - df["height"].sum() * 1e-3
            depth_to = bottom_tp + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        elif idx == "mp":
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.mp_sub_assemblies.loc[df_index, cols])
            toe = self.pile_head - df["height"].sum() * 1e-3
            self.pile_toe = round(toe, 3)
            depth_to = toe + df.z * 1e-3
            depth_from = depth_to + df.height * 1e-3
        else:
            raise ValueError("Unknown index.")
        df["Depth from [mLAT]"] = depth_from
        df["Depth to [mLAT]"] = depth_to
        # Round elevations to mm to avoid numerical inconsistencies later when setting altitude values to apply loads.
        df = df.round({"Depth from [mLAT]": 3, "Depth to [mLAT]": 3})
        return df

    def process_structure_geometry(self, idx: str) -> pd.DataFrame:
        """Calculates and/or converts geometrical data of subassemblies from the database to use as input for FE models.

        :param idx: Possible index to identify corresponding subassembly.
        :return: Dataframe consisting of the required data to build FE models.
        """
        df = self.set_df_structure(idx)
        df.height = pd.to_numeric(df.height)
        df.wall_thickness = pd.to_numeric(df.wall_thickness)
        df.rename(columns={"wall_thickness": "Wall thickness [mm]"}, inplace=True)
        df.rename(columns={"volume": "Volume [m3]"}, inplace=True)
        # Outer diameter saved as string, need to be split an converted to float
        od = df.OD.values
        d_to = [d.split("/", 1)[0] for d in od]
        d_from = []
        for d in od:
            d_i = d.split("/", 1)
            if len(d_i) > 1:
                d_from.append(d_i[1])
            else:
                d_from.append(d_i[0])
        df["Diameter from [m]"] = np.array(d_from, dtype=float) * 1e-3  # to meters
        df["Diameter to [m]"] = np.array(d_to, dtype=float) * 1e-3  # to meters

        # Creating additional columns with required mechanical properties to build up tbe FE model
        df["rho [t/m]"] = df.mass / df.height
        df["Mass [t]"] = df.mass * 1e-3
        df["Height [m]"] = df.height * 1e-3
        df["Youngs modulus [GPa]"] = 210
        df["Poissons ratio [-]"] = 0.3

        # Selecting only needed columns for FE modelling
        cols = [
            "Depth from [mLAT]",
            "Depth to [mLAT]",
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

    def process_rna(self) -> pd.DataFrame:
        """Sets dataframe containing the required properties to model the RNA system.

        :return:
        """
        rna_index = self.tower_sub_assemblies.index.str.contains("RNA")
        rna = deepcopy(
            self.tower_sub_assemblies.loc[
                rna_index, ["mass", "moment_of_inertia", "x", "y", "z"]
            ]
        )
        # Moments of Inertia - [tonnes m2]
        mi = rna["moment_of_inertia"].values
        i_xx = []
        i_yy = []
        i_zz = []
        for i in range(len(mi)):
            i_xx.append(mi[i]["x"] * 1e-3)
            i_yy.append(mi[i]["y"] * 1e-3)
            i_zz.append(mi[i]["z"] * 1e-3)

        # Adding columns
        rna["Ixx [tm2]"] = i_xx
        rna["Iyy [tm2]"] = i_yy
        rna["Izz [tm2]"] = i_zz
        rna["Mass [t]"] = rna.mass * 1e-3
        rna["X [m]"] = rna.x * 1e-3
        rna["Y [m]"] = rna.y * 1e-3
        rna["Z [mLAT]"] = self.tower_base + rna.z * 1e-3

        # Setting DataFrame
        cols = [
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Mass [t]",
            "Ixx [tm2]",
            "Iyy [tm2]",
            "Izz [tm2]",
        ]
        self.rna = rna[cols]

    def set_df_appurtenances(self, idx):
        """
        Sets dataframe containing the required properties to model concentrated masses from database subassemblies.

        :param idx: string. Possible values:
            * 'TW'
            * 'TP'
            * 'MP'

        :return: data frame containing lumped masses data from database. Z coordinates in mLAT system.
        """
        if idx == "TW":
            # Tower appurtenances
            df_index = self.tower_sub_assemblies.index.str.contains(idx)
            df = deepcopy(
                self.tower_sub_assemblies.loc[df_index, ["mass", "x", "y", "z"]]
            )
            # Conversion of local z coordinates to elevation in mLAT system
            df["Z [mLAT]"] = self.tower_base + df.z * 1e-3
        elif idx == "TP":
            # Transition piece
            df_index = self.tp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(
                self.tp_sub_assemblies.loc[df_index, ["mass", "x", "y", "z", "height"]]
            )
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df.height = pd.to_numeric(df.height)
            df = df[df.height.isnull()]
            # Conversion of local z coordinates to elevation in mLAT system
            bottom = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom + df.z * 1e-3
        elif idx == "MP":
            # Monopile
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(
                self.mp_sub_assemblies.loc[df_index, ["mass", "x", "y", "z", "height"]]
            )
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df.height = pd.to_numeric(df.height)
            df = df[df.height.isnull()]
            # Conversion of local z coordinates to elevation in mLAT system
            bottom = self.pile_toe
            df["Z [mLAT]"] = bottom + df.z * 1e-3
        else:
            raise ValueError("Unknown index.")

        return df

    def process_lumped_masses(self, idx):
        """
        Creates dataframe containing the required properties to model lumped mass appurtenances. Note that
        if the preprocessor package does not find any appurtenances it'll return an empty dataframe.

        :param idx: string. Possible values:
            * 'TW'
            * 'TP'
            * 'MP'

        :return: dataframe.
        """
        df = self.set_df_appurtenances(idx)
        # Setting units
        df["Mass [t]"] = df.mass * 1e-3
        df["X [m]"] = df.x * 1e-3
        df["Y [m]"] = df.y * 1e-3

        # Setting DataFrame
        cols = ["X [m]", "Y [m]", "Z [mLAT]", "Mass [t]"]

        return df[cols]

    def set_df_distributed_appurtenances(self, idx):
        """
        :param idx: string. Possible values:
            * 'TP'
            * 'MP'

        :return: data frame containing distributed lumped masses data from database. Z coordinates in mLAT system.
        """
        if idx == "TP":
            # Transition piece
            # Grout is included here
            df_index = (self.tp_sub_assemblies.index.str.contains(idx)) | (
                self.tp_sub_assemblies.index.str.contains("grout")
            )
            df = deepcopy(
                self.tp_sub_assemblies.loc[
                    df_index, ["mass", "x", "y", "z", "height", "volume", "description"]
                ]
            )
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df.height = pd.to_numeric(df.height)
            df = df[df.height.notnull()]
            # Conversion of local z coordinates to elevation in mLAT system
            bottom_tp = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom_tp + df.z * 1e-3
        elif idx == "MP":
            # Monopile
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(
                self.mp_sub_assemblies.loc[
                    df_index, ["mass", "x", "y", "z", "height", "volume", "description"]
                ]
            )
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df.height = pd.to_numeric(df.height)
            df = df[df.height.notnull()]
            # Conversion of local z coordinates to elevation in mLAT system
            bottom = self.pile_toe
            df["Z [mLAT]"] = bottom + df.z * 1e-3
        else:
            raise ValueError(
                "Unknown index or non distributed lumped masses located outside the transition piece."
            )

        return df

    def process_distributed_lumped_masses(self, idx):
        """
        Creates dataframe containing the required properties to model uniformly distributed appurtenances. Note that
        if the preprocessor package does not find any appurtenances it'll return an empty dataframe.

        :param idx: string. Possible values:
            * 'TP'
            * 'MP'

        :return: dataframe.
        """
        df = self.set_df_distributed_appurtenances(idx)
        # Setting units
        df["Mass [t]"] = df.mass * 1e-3
        df["X [m]"] = df.x * 1e-3
        df["Y [m]"] = df.y * 1e-3
        df["Height [m]"] = df.height * 1e-3
        df.rename(columns={"volume": "Volume [m3]"}, inplace=True)

        # Setting DataFrame
        cols = [
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Height [m]",
            "Mass [t]",
            "Volume [m3]",
            "description",
        ]

        return df[cols]

    def process_structure(self, option="full"):
        """
        Sets dataframe containing the required properties to model the tower geometry, including the RNA system.

        :param option: string. Possible values:
            * 'full': process all the data for all subassemblies.
            * 'tower': only process the data for the tower subassembly.
            * 'TP': only process the data for the transition piece subassembly.
            * 'monopile': only process the data for the monopile foundation subassembly.

        :return:
        """
        if option == "full":
            # RNA system
            self.process_rna()
            # Support structure
            self.tower_geometry = self.process_structure_geometry("tw")
            self.transition_piece = self.process_structure_geometry("tp")
            self.monopile = self.process_structure_geometry("mp")
            # Lumped mass appurtenances
            self.tower_lumped_mass = self.process_lumped_masses("TW")
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            # Uniformly distributed appurtenances
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")
        elif option == "tower":
            # RNA system
            self.process_rna()
            # Support structure
            self.tower_geometry = self.process_structure_geometry("tw")
            # Lumped mass appurtenances
            self.tower_lumped_mass = self.process_lumped_masses("TW")
        elif option == "TP":
            # Support structure
            self.transition_piece = self.process_structure_geometry("tp")
            # Lumped mass appurtenances
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            # Uniformly distributed appurtenances
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
        elif option == "monopile":
            # Support structure
            self.monopile = self.process_structure_geometry("mp")
            # Lumped mass appurtenances
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            # Uniformly distributed appurtenances
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")

    @staticmethod
    def can_properties(row: pd.Series) -> pd.Series:
        """Recalculation of can properties based on section properties and can elevations: height [m],
        volume [m3], mass [t], rho [t/m].

        :param row: Original can properties.
        :return: Recalculated can properties.
        """
        density = row["Mass [t]"] / row["Volume [m3]"]
        height = row["Depth from [mLAT]"] - row["Depth to [mLAT]"]
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

    def can_modification(self, df, altitude, position="bottom"):
        """
        Args:
            df (DataFrame) :
            altitude (float):
            position (str):
        """
        if position == "bottom":
            # Last can to be modified
            ind = -1
            _col = " to "
        else:
            # First can to be modified
            ind = 0
            _col = " from "
        # Altitude substitution
        df.loc[df.index[ind], "Depth" + _col + "[mLAT]"] = altitude
        # Interpolating diameter
        elevation = [df.iloc[ind]["Depth from [mLAT]"], df.iloc[ind]["Depth to [mLAT]"]]
        diameters = [df.iloc[ind]["Diameter from [m]"], df.iloc[ind]["Diameter to [m]"]]
        df.loc[df.index[ind], "Diameter" + _col + "[m]"] = np.interp(
            [altitude], elevation, diameters
        )[
            0
        ]  # interpolated D
        # Recalculating other properties to match geometry & Substitution
        cols = ["Height [m]", "Volume [m3]", "Mass [t]", "rho [t/m]"]
        df.loc[df.index[ind], cols] = self.can_properties(df.iloc[ind])
        return df

    def assembly_tp_mp(self):
        """
        Further processing TP structural item to assembly with MP foundation ensuring continuity. Tp skirt is processed
        as well.
        """
        # Assembly TP + MP
        if (self.transition_piece is not None) and (self.transition_piece is not None):
            # Substructure
            mp_head = self.pile_head
            tp = self.transition_piece
            df = deepcopy(tp[tp["Depth from [mLAT]"] > mp_head])
            if df.loc[df.index[0], "Depth to [mLAT]"] != mp_head:
                # Not bolted connection (i.e. Rentel) preprocessing needed
                tp1 = self.can_modification(df, mp_head, position="bottom")
                # Concatenate with MP
                self.substructure = pd.concat([tp1, deepcopy(self.monopile)])
            else:
                # Bolted connection, nothing to do
                self.substructure = pd.concat([df, deepcopy(self.monopile)])
            # TP skirt
            df = deepcopy(tp[tp["Depth to [mLAT]"] < mp_head])
            self.tp_skirt = self.can_modification(df, mp_head, position="top")
        else:
            raise TypeError("TP or MP items need to be processed before!")


def get_monopile_pyles(
    self, water_depth, projectsite, assetlocation, cutoff_point=np.nan
):
    """
    Returns a dataframe with the monopile geometry with the mudline as reference

    :param water_depth: Water depth in mLAT
    :param projectsite:
    :param assetlocation:
    :return:
    """
    # Retrieve the monopile cans
    bbs = self.get_buildingblocks(
        projectsite=projectsite, assetlocation=assetlocation, buildingblock_type="MP"
    )
    # Retrieve the monopile subassembly
    sas = self.get_subassemblies(
        projectsite=projectsite, assetlocation=assetlocation, subassembly_type="MP"
    )
    # Calculate the pile penetration
    toe_depth_lat = sas["data"]["z_position"].iloc[0]
    penetration = -((1e-3 * toe_depth_lat) - water_depth)

    # Create the pile for subsequent response analysis
    pile = pd.DataFrame()

    for i, row in bbs["data"].iterrows():
        if i != 0:
            pile.loc[i, "Depth from [m]"] = (
                penetration - 1e-3 * bbs["data"].loc[i - 1, "z_position"]
            )
            pile.loc[i, "Depth to [m]"] = penetration - 1e-3 * row["z_position"]
            pile.loc[i, "Pile material"] = row["material_name"]
            pile.loc[i, "Pile material submerged unit weight [kN/m3]"] = (
                1e-2 * row["density"] - 10
            )
            pile.loc[i, "Wall thickness [mm]"] = row["wall_thickness"]
            pile.loc[i, "Diameter [m]"] = (
                1e-3 * 0.5 * (row["bottom_outer_diameter"] + row["top_outer_diameter"])
            )
            pile.loc[i, "Youngs modulus [GPa]"] = row["youngs_modulus"]
            pile.loc[i, "Poissons ratio [-]"] = row["poissons_ratio"]

    # Cut off at the mudline
    if not np.math.isnan(cutoff_point):
        pile = pile.loc[pile["Depth to [m]"] > cutoff_point].reset_index(drop=True)
        pile.loc[0, "Depth from [m]"] = cutoff_point

    return pile
