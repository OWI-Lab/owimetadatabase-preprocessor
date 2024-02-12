"Module containing the processing functions for the geometry data."

from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.geometry.structures import SubAssembly
from owimetadatabase_preprocessor.utils import deepcompare


class OWT(object):
    def __init__(
        self,
        api,
        materials: pd.DataFrame,
        subassemblies: pd.DataFrame,
        location: pd.DataFrame,
        tower_base: Union[float, None] = None,
        pile_head: Union[float, None] = None,
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
        if not tower_base or not pile_head:
            self.tower_base = self.sub_assemblies["TW"].absolute_bottom
            self.pile_head = self.sub_assemblies["MP"].absolute_top
        else:
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
        self.water_depth = location["elevation"].values[0]

    def _set_subassemblies(self, subassemblies: pd.DataFrame) -> None:
        """Create a dictionary containing the subassemblies of the OWT."""
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

    def _set_members(self) -> None:
        """Identify and stores in separate data frames each part of the support structure (tower=TW, transition piece=TP,
        monopile=MP).
        """
        for k, v in self.sub_assemblies.items():
            if k == "TW":
                self.tower_sub_assemblies = v.as_df()
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

    def process_rna(self) -> None:
        """Set dataframe containing the required properties to model the RNA system.

        :return:
        """
        rna_index = self.tower_sub_assemblies.index.str.contains("RNA")
        rna = deepcopy(
            self.tower_sub_assemblies.loc[
                rna_index, ["mass", "moment_of_inertia", "x", "y", "z"]
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

    def set_df_appurtenances(self, idx: str) -> pd.DataFrame:
        """Set dataframe containing the required properties to model concentrated masses from database subassemblies.

        :param idx: Index to identify corresponding subassembly with possible values: 'TW', 'TP', 'MP'.
        :return: Data frame containing lumped masses data from database with Z coordinates in mLAT system.
        """
        cols = ["mass", "x", "y", "z"]
        if idx == "TW":
            df_index = self.tower_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tower_sub_assemblies.loc[df_index, cols])
            df["Z [mLAT]"] = self.tower_base + df["z"] * 1e-3
        elif idx == "TP":
            df_index = self.tp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols + ["height"]])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].isnull()]
            bottom = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom + df["z"] * 1e-3
        elif idx == "MP":
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
        cols = ["X [m]", "Y [m]", "Z [mLAT]", "Mass [t]"]
        return df[cols]

    def set_df_distributed_appurtenances(self, idx: str) -> pd.DataFrame:
        """Set dataframe containing the required properties to model distributed lumped masses from database.

        :param idx: Index to identify corresponding subassembly with possible values: 'TW', 'TP', 'MP'.
        :return: Dataframe containing distributed lumped masses data from database. Z coordinates in mLAT system.
        """
        cols = ["mass", "x", "y", "z", "height", "volume", "description"]
        if idx == "TP":
            # Grout is included here
            df_index = (self.tp_sub_assemblies.index.str.contains(idx)) | (
                self.tp_sub_assemblies.index.str.contains("grout")
            )
            df = deepcopy(self.tp_sub_assemblies.loc[df_index, cols])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].notnull()]
            bottom_tp = self.tower_base - self.tp_sub_assemblies.iloc[0]["z"] * 1e-3
            df["Z [mLAT]"] = bottom_tp + df["z"] * 1e-3
        elif idx == "MP":
            df_index = self.mp_sub_assemblies.index.str.contains(idx)
            df = deepcopy(self.mp_sub_assemblies.loc[df_index, cols])
            # Lumped masses have 'None' height whereas distributed masses present not 'None' values
            df["height"] = pd.to_numeric(df["height"])
            df = df[df["height"].notnull()]
            bottom = self.pile_toe
            df["Z [mLAT]"] = bottom + df["z"] * 1e-3
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

    def process_structure(self, option="full") -> None:
        """Set dataframe containing the required properties to model the tower geometry, including the RNA system.

        :param option: Option to process the data for a specific subassembly. Possible values:

            - "full": To process all the data for all subassemblies.
            - "tower": To process only the data for the tower subassembly.
            - "TP": To process only the data for the transition piece subassembly.
            - "monopile": To process only the data for the monopile foundation subassembly.
        :return:
        """
        if option == "full":
            self.process_rna()
            self.tower_geometry = self.process_structure_geometry("tw")
            self.transition_piece = self.process_structure_geometry("tp")
            self.monopile = self.process_structure_geometry("mp")
            self.tower_lumped_mass = self.process_lumped_masses("TW")
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")
        elif option == "tower":
            self.process_rna()
            self.tower_geometry = self.process_structure_geometry("tw")
            self.tower_lumped_mass = self.process_lumped_masses("TW")
        elif option == "TP":
            self.transition_piece = self.process_structure_geometry("tp")
            self.tp_lumped_mass = self.process_lumped_masses("TP")
            self.tp_distributed_mass = self.process_distributed_lumped_masses("TP")
        elif option == "monopile":
            self.monopile = self.process_structure_geometry("mp")
            self.mp_lumped_mass = self.process_lumped_masses("MP")
            self.mp_distributed_mass = self.process_distributed_lumped_masses("MP")

    @staticmethod
    def can_adjust_properties(row: pd.Series) -> pd.Series:
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

    def can_modification(
        self, df: pd.DataFrame, altitude: np.float64, position: str = "bottom"
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
        df.loc[df.index[ind], "Depth" + _col + "[mLAT]"] = altitude
        elevation = [df.iloc[ind]["Depth from [mLAT]"], df.iloc[ind]["Depth to [mLAT]"]]
        diameters = [df.iloc[ind]["Diameter from [m]"], df.iloc[ind]["Diameter to [m]"]]
        df.loc[df.index[ind], "Diameter" + _col + "[m]"] = np.interp(
            [altitude], elevation, diameters
        )[0]
        cols = ["Height [m]", "Volume [m3]", "Mass [t]", "rho [t/m]"]
        df.loc[df.index[ind], cols] = self.can_adjust_properties(df.iloc[ind])
        return df

    def assembly_tp_mp(self) -> None:
        """Process TP structural item to assembly with MP foundation ensuring continuity. TP skirt is processed
        as well.

        :return:
        """
        if (self.transition_piece is not None) and (self.monopile is not None):
            mp_head = self.pile_head
            tp = self.transition_piece
            df = deepcopy(tp[tp["Depth from [mLAT]"] > mp_head])
            if df.loc[df.index[0], "Depth to [mLAT]"] != mp_head:
                # Not bolted connection (i.e. Rentel) preprocessing needed
                tp1 = self.can_modification(df, mp_head, position="bottom")
                self.substructure = pd.concat([tp1, deepcopy(self.monopile)])
            else:
                # Bolted connection, nothing to do
                self.substructure = pd.concat([df, deepcopy(self.monopile)])
            df = deepcopy(tp[tp["Depth to [mLAT]"] < mp_head])
            self.tp_skirt = self.can_modification(df, mp_head, position="top")
        else:
            raise TypeError("TP or MP items need to be processed before!")

    def get_monopile_pyles(
        self,
        projectsite: str,
        assetlocation: str,
        cutoff_point: np.float64 = np.nan,
    ) -> pd.DataFrame:
        """Returns a dataframe with the monopile geometry with the mudline as reference

        :param projectsite: Title of the projectsite.
        :param assetlocation: Title of the turbine.
        :return: Dataframe with the monopile geometry.
        """
        building_blocks = self.api.get_buildingblocks(
            projectsite=projectsite, assetlocation=assetlocation, subassembly_type="MP"
        )
        subassemblies = self.api.get_subassemblies(
            projectsite=projectsite, assetlocation=assetlocation, subassembly_type="MP"
        )
        toe_depth_lat = subassemblies["data"]["z_position"].iloc[0]
        penetration = -((1e-3 * toe_depth_lat) - self.water_depth)
        pile = pd.DataFrame()
        for i, row in building_blocks["data"].iterrows():
            if i != 0:
                pile.loc[i, "Depth from [m]"] = (
                    penetration
                    - 1e-3 * building_blocks["data"].loc[i - 1, "z_position"]
                )
                pile.loc[i, "Depth to [m]"] = penetration - 1e-3 * row["z_position"]
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
        if not np.math.isnan(cutoff_point):
            pile = pile.loc[pile["Depth to [m]"] > cutoff_point].reset_index(drop=True)
            pile.loc[0, "Depth from [m]"] = cutoff_point
        return pile

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return deepcompare(self, other)
        elif isinstance(other, dict):
            return deepcompare(self.__dict__, other)
        else:
            return False


class OWTs(object):
    """Class to process the geometry data of multiple OWTs."""

    def __init__(
        self,
        turbines: List[str],
        owts: List[OWT],
    ) -> None:
        self.owts = {k: v for k, v in zip(turbines, owts)}
        self.api = self.owts[turbines[0]].api
        self.materials = self.owts[turbines[0]].materials
        self.sub_assemblies = {
            k: owt.sub_assemblies for k, owt in zip(turbines, self.owts.values())
        }
        self.tower_sub_assemblies = pd.concat(
            [owt.tower_sub_assemblies for owt in self.owts.values()]
        )
        self.tp_sub_assemblies = pd.concat(
            [owt.tp_sub_assemblies for owt in self.owts.values()]
        )
        self.mp_sub_assemblies = pd.concat(
            [owt.mp_sub_assemblies for owt in self.owts.values()]
        )
        self.tower_base = {
            k: owt.tower_base for k, owt in zip(turbines, self.owts.values())
        }
        self.pile_head = {
            k: owt.pile_head for k, owt in zip(turbines, self.owts.values())
        }
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
        self.water_depth = {
            k: owt.water_depth for k, owt in zip(turbines, self.owts.values())
        }
        self.all_cans = None
        self.all_distributed_mass = None
        self.all_lumped_mass = None
        self.all_turbines = None

    def _concat_list(self, attr_list) -> None:
        """Internal method to concatenate lists of dataframes for attributes.

        :param attr_list: List of attributes to concatenate.
        """
        for attr in attr_list:
            setattr(self, attr, pd.concat(getattr(self, attr)))

    def assembly_turbine(self) -> None:
        """Method to assemble general geometry data of all specified turbines."""
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
        ]
        df_list = []
        for turb in self.owts.keys():
            df_list.append(
                [
                    turb,
                    self.water_depth[turb],
                    self.pile_toe[turb],
                    self.pile_head[turb],
                    self.tower_base[turb],
                    self.owts[turb].monopile["Height [m]"].sum(),
                    (
                        self.owts[turb].monopile["Mass [t]"].sum()
                        + self.owts[turb].mp_distributed_mass["Mass [t]"].sum()
                        + self.owts[turb].mp_lumped_mass["Mass [t]"].sum()
                    ),
                    self.owts[turb].transition_piece["Height [m]"].sum(),
                    (
                        self.owts[turb].transition_piece["Mass [t]"].sum()
                        + self.owts[turb].tp_distributed_mass["Mass [t]"].sum()
                        + self.owts[turb].tp_lumped_mass["Mass [t]"].sum()
                    ),
                    self.owts[turb].tower_geometry["Height [m]"].sum(),
                    (
                        self.owts[turb].tower_geometry["Mass [t]"].sum()
                        + self.owts[turb].tower_lumped_mass["Mass [t]"].sum()
                        + self.owts[turb].rna["Mass [t]"].sum()
                    ),
                ]
            )
        df = pd.DataFrame(df_list, columns=cols)
        self.all_turbines = df.round(2)

    def process_structure(self) -> None:
        """Set dataframes containing the required properties to model the tower geometry, including the RNA system."""
        attr_list = []
        for attr in list(self.__dict__.keys()):
            if getattr(self, attr) is None:
                attr_list.append(attr)
                setattr(self, attr, [])
        attr_list.remove("all_turbines")
        for owt in self.owts.values():
            owt.process_structure()
            owt.assembly_tp_mp()
            for attr in attr_list:
                if attr == "pile_toe":
                    self.pile_toe.append(getattr(owt, attr))
                elif attr == "all_cans":
                    self.all_cans.extend(
                        [owt.tower_geometry, owt.transition_piece, owt.monopile]
                    )
                elif attr == "all_distributed_mass":
                    self.all_distributed_mass.extend(
                        [owt.tp_distributed_mass, owt.mp_distributed_mass]
                    )
                elif attr == "all_lumped_mass":
                    self.all_lumped_mass.extend(
                        [owt.tower_lumped_mass, owt.tp_lumped_mass, owt.mp_lumped_mass]
                    )
                else:
                    attr_val = getattr(self, attr)
                    owt_attr_val = getattr(owt, attr)
                    attr_val.append(owt_attr_val)
        attr_list.remove("pile_toe")
        self.pile_toe = {k: v for k, v in zip(self.owts.keys(), self.pile_toe)}
        self._concat_list(attr_list)
        self.assembly_turbine()

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return deepcompare(self, other)
        elif isinstance(other, dict):
            return deepcompare(self.__dict__, other)
        else:
            return False
