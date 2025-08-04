"""
This module defines the SoilDataProcessor class which provides helper routines
for processing soil data.  It is used to transform coordinates, combine raw
and processed DataFrames, and extract/convert in-situ test detail data.
"""

import warnings
from typing import Dict, List, Tuple, Union

import pandas as pd
from groundhog.general.soilprofile import profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from pyproj import Transformer


class SoilDataProcessor:
    """
    Helper class for processing soil data.
    """

    @staticmethod
    def transform_coord(
        df: pd.DataFrame, longitude: float, latitude: float, target_srid: str
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Transform coordinates from EPSG:4326 to the target SRID.

        The input DataFrame must contain the keys 'easting' and 'northing'.
        The function transforms these to new columns 'easting [m]' and '
        northing [m]'. In addition, the central point (longitude, latitude)
        is also transformed.

        :param df: Input DataFrame with 'easting' and 'northing' columns.
        :param longitude: Longitude of the central point (in decimal degrees).
        :param latitude: Latitude of the central point (in decimal degrees).
        :param target_srid: Target SRID as a string (e.g. "25831").
        :return: A tuple containing:
            - The updated DataFrame with transformed coordinates.
            - The transformed easting of the central point.
            - The transformed northing of the central point.
        """
        transformer = Transformer.from_crs(
            "epsg:4326", f"epsg:{target_srid}", always_xy=True
        )
        try:
            # Transform the easting and northing columns in the DataFrame
            df["easting [m]"], df["northing [m]"] = transformer.transform(
                df["easting"], df["northing"]
            )
        except Exception as err:
            warnings.warn(f"Error transforming DataFrame coordinates: {err}")
        # Transform the reference central point
        point_east, point_north = transformer.transform(longitude, latitude)
        return df, point_east, point_north

    @staticmethod
    def combine_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine two DataFrames (usually raw and processed data) along the
        common column "z [m]".

        If the merge fails, the method returns the 'rawdata' DataFrame as a
        fallback.

        :param dfs: Dictionary of DataFrames with keys "rawdata" and
            "processeddata".
        :return: The merged DataFrame if successful; otherwise, returns the
            "rawdata" DataFrame.
        """
        try:
            combined_df = pd.merge(
                dfs["rawdata"],
                dfs["processeddata"],
                on="z [m]",
                how="inner",
                suffixes=("", "_processed"),
            )
            return combined_df
        except Exception as err:
            warnings.warn(f"Error combining raw and processed data: {err}")
            return dfs.get("rawdata", pd.DataFrame())

    @staticmethod
    def process_insitutest_dfs(
        df: pd.DataFrame, cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process the in-situ test detail DataFrame by extracting specified
        columns.

        Each specified column is assumed to contain nested data (such as a
        dictionary or list) in its first row. The method attempts to convert
        these nested structures into new DataFrames and also applies numerical
        conversion where applicable.

        :param df: The input DataFrame containing in-situ test details.
        :param cols: A list of column names to extract from the DataFrame.
        :return: A dictionary mapping each column name (as key) to its
            processed DataFrame.
        """
        processed_dfs = {}
        for col in cols:
            try:
                # The column data is assumed to be in the first row as a nested
                # dict or list.
                temp_df = pd.DataFrame(df[col].iloc[0]).reset_index(drop=True)
                processed_dfs[col] = temp_df
            except KeyError:
                warnings.warn(
                    f"""
                    Column '{col}' not found. Check the DataFrame structure.

                    Check that you entered correct parameters in your request
                    or contact database administrators.
                    """
                )
                processed_dfs[col] = pd.DataFrame()
            except Exception as e:
                warnings.warn(f"Error processing column '{col}': {e}")
                processed_dfs[col] = pd.DataFrame()

        # Attempt to convert values to numeric where applicable.
        for key in processed_dfs:
            try:
                processed_dfs[key] = processed_dfs[key].apply(
                    lambda x: pd.to_numeric(x)
                )
            except Exception as err:
                warnings.warn(f"Numeric conversion warning for {key}: {err}")
        return processed_dfs

    @staticmethod
    def gather_data_entity(
        df: pd.DataFrame,
    ) -> Dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Gather the data for the closest entity to a certain point in 2D.

        :param df: Pandas dataframe with the data according to the specified search criteria
        :return: Dictionary with the following keys:

            - 'data': Pandas dataframe with the test location data for each location in the specified search area
            - 'id': ID of the closest test location
            - 'title': Title of the closest test location
            - 'offset [m]': Offset in meters from the specified point
        """
        if df.__len__() == 1:
            loc_id = df["id"].iloc[0]
        else:
            df.sort_values("offset [m]", inplace=True)
            loc_id = df[df["offset [m]"] == df["offset [m]"].min()]["id"].iloc[0]
        return {
            "data": df,
            "id": loc_id,
            "title": df["title"].iloc[0],
            "offset [m]": df[df["offset [m]"] == df["offset [m]"].min()][
                "offset [m]"
            ].iloc[0],
        }

    @staticmethod
    def process_cpt(df_sum: pd.DataFrame, df_raw: pd.DataFrame, **kwargs):
        # TODO: add docstring and type hints
        try:
            cpt = PCPTProcessing(title=df_sum["title"].iloc[0])
            if "Push" in df_raw.keys():
                push_key = "Push"
            else:
                push_key = None
            cpt.load_pandas(df_raw, push_key=push_key, **kwargs)
            return cpt
        except Exception as err:
            warnings.warn(f"ERROR: PCPTProcessing object not created - {err}")
            return None

    @staticmethod
    def convert_to_profile(df_sum, df_detail, profile_title, drop_info_cols):
        # TODO: add docstring and type hints
        try:
            soilprofile_df = (
                pd.DataFrame(df_detail["soillayer_set"].iloc[0])
                .sort_values("start_depth")
                .reset_index(drop=True)
            )
            soilprofile_df.rename(
                columns={
                    "start_depth": "Depth from [m]",
                    "end_depth": "Depth to [m]",
                    "soiltype_name": "Soil type",
                    "totalunitweight": "Total unit weight [kN/m3]",
                },
                inplace=True,
            )
            for i, row in soilprofile_df.iterrows():
                try:
                    for key, value in row["soilparameters"].items():
                        soilprofile_df.loc[i, key] = value
                except Exception:
                    pass
            for col in soilprofile_df.columns:
                is_numeric_col = True
                for value in soilprofile_df[col]:
                    if value is None or pd.isna(value) or value == "" or value == "None" or value == "null":
                        continue
                    if not isinstance(value, (int, float)):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            is_numeric_col = False
                            break
                if is_numeric_col:
                    try:
                        soilprofile_df[col] = pd.to_numeric(soilprofile_df[col])
                    except Exception as err:
                        warnings.warn(
                            f"Error converting column '{col}' to numeric: {err}"
                        )
            if profile_title is None:
                profile_title = (
                    f"{df_sum['location_name'].iloc[0]} - {df_sum['title'].iloc[0]}"
                )
            if drop_info_cols:
                soilprofile_df.drop(
                    [
                        "id",
                        "profile",
                        "soilparameters",
                        "soilprofile_name",
                        "soilunit",
                        "description",
                        "soilunit_name",
                    ],
                    axis=1,
                    inplace=True,
                )
            dsp = profile_from_dataframe(soilprofile_df, title=profile_title)
            return dsp
        except KeyError:
            warnings.warn(
                """
                Something is wrong with the output dataframe:
                check that the database gave a non-empty output.

                Check that you entered correct parameters in your request
                or contact database administrators.
                """
            )
            return None
        except Exception as err:
            warnings.warn(f"Error during loading of soil layers and parameters: {err}")
            return None

    @staticmethod
    def fulldata_processing(
        unitdata, row, selected_depths, func_get_details, depthcol, **kwargs
    ):
        # TODO: add docstring and type hints
        _fulldata = func_get_details(location=row["location_name"], **kwargs)["rawdata"]
        _depthranges = selected_depths[
            selected_depths["location_name"] == row["location_name"]
        ]
        for _, _layer in _depthranges.iterrows():
            _unitdata = _fulldata[
                (_fulldata[depthcol] >= _layer["start_depth"])
                & (_fulldata[depthcol] <= _layer["end_depth"])
            ]
            unitdata = pd.concat([unitdata, _unitdata])
        unitdata.reset_index(drop=True, inplace=True)
        unitdata.loc[:, "location_name"] = row["location_name"]
        unitdata.loc[:, "projectsite_name"] = row["projectsite_name"]
        unitdata.loc[:, "test_type_name"] = row["test_type_name"]
        return unitdata

    @staticmethod
    def partialdata_processing(unitdata, row, selected_depths, selected_tests):
        # TODO: add docstring and type hints
        _depthranges = selected_depths[
            selected_depths["location_name"] == row["location_name"]
        ]
        for _, _layer in _depthranges.iterrows():
            if (
                row["depth"] >= _layer["start_depth"]
                and row["depth"] <= _layer["end_depth"]
            ):
                _unitdata = selected_tests[selected_tests["id"] == row["id"]]
                unitdata = pd.concat([unitdata, _unitdata])
            else:
                pass
        unitdata.reset_index(drop=True, inplace=True)

    @staticmethod
    def objects_to_list(selected_obj, func_get_detail, data_type):
        # TODO: add docstring and type hints
        obj = []
        for _, row in selected_obj.iterrows():
            try:
                if data_type == "soilprofile":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "soilprofile": row["title"],
                        "drop_info_cols": False,
                        "profile_title": row["location_name"],
                    }
                elif data_type == "cpt":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "insitutest": row["title"],
                        "testtype": row["test_type_name"],
                    }
                else:
                    raise ValueError(f"Data type {data_type} not supported.")
                _obj = func_get_detail(**params)[data_type]
                _obj.set_position(
                    easting=row["easting"],
                    northing=row["northing"],
                    elevation=row["elevation"],
                )
                obj.append(_obj)
            except Exception:
                warnings.warn(
                    f"Error loading {row['projectsite_name']}-{row['location_name']}-{row['title']}"
                )
        return obj


class SoilprofileProcessor:
    """
    Helper class for processing required inputs from a given dataframe for
    soil-strucutre interaction modeling.

    The class defines a database of keys (LATERAL_SOIL_KEYS) to be used by the
    lateral() method. For each available option (e.g. "apirp2geo", "pisa"), the
    dictionary contains lists of mandatory and, optionally, optional keys.
    If any mandatory keys are missing in the provided DataFrame, an error is
    raised. Otherwise, the DataFrame is filtered to include the mandatory keys
    and any optional keys that are present.
    """

    LATERAL_SSI_KEYS: Dict[str, Dict[str, List[Union[str, Tuple[str, str]]]]] = {
        "apirp2geo": {
            "mandatory": [
                "Depth from [m]",
                "Depth to [m]",
                "Soil type",
                ("Total unit weight", "[kN/m3]"),
                ("Su", "[kPa]"),
                ("Phi", "[deg]"),
                ("epsilon50", "[-]"),
            ],
            "optional": [
                ("Dr", "[-]"),
            ],
        },
        "pisa": {
            "mandatory": [
                "Depth from [m]",
                "Depth to [m]",
                "Soil type",
                ("Total unit weight", "[kN/m3]"),
                ("Gmax", "[kPa]"),
                ("Su", "[kPa]"),
                ("Dr", "[-]"),
            ],
            "optional": [],
        },
    }

    AXIAL_SSI_KEYS: Dict[str, Dict[str, List[Union[str, Tuple[str, str]]]]] = {
        "cpt": {
            "mandatory": [],
            "optional": [],
        }
    }

    @classmethod
    def get_available_options(cls, loading: str = "lateral") -> list[str]:
        """
        Return a list of available lateral soil reaction modeling options.

        :param loading: String specifying the type of loading (default='lateral').
        :return: List of available options for the specified loading type.
        :raises ValueError: If the provided loading type is not supported.
        """
        if loading.lower() == "lateral":
            return list(cls.LATERAL_SSI_KEYS.keys())
        elif loading.lower() == "axial":
            return list(cls.AXIAL_SSI_KEYS.keys())
        else:
            raise ValueError(f"Unsupported loading type '{loading}'.")

    @staticmethod
    def _validate_keys(
        data: pd.DataFrame, required_keys: list, mandatory: bool = True
    ) -> list[str]:
        """
        Validate that all required keys are present in the data and return a list
        of standardized column names corresponding to these keys.
        For keys defined as tuples, ensures that at least one column contains all the
        tuple elements (case-insensitive); for string keys, performs a case-insensitive
        check and renames the column to the standardized key if needed.

        :param data: DataFrame containing the soil profile data.
        :param required_keys: List of required keys that may be strings or tuples.
        :param mandatory: Boolean flag. If True, raise error when a key is missing;
                        if False, skip keys that are not found.
        :return: A list of validated (and standardized) column names.
        :raises ValueError: If any required key is missing and mandatory is True.
        """
        validated_columns = []
        # Map lower-case column names to original names for renaming.
        keys_lower = {col.lower(): col for col in data.columns}

        for key in required_keys:
            if isinstance(key, tuple):
                candidate = []
                for col in data.columns:
                    if all(elem.lower() in col.lower() for elem in key):
                        candidate.append(col)
                if candidate == []:
                    if mandatory:
                        raise ValueError(
                            f"Soil input: '{key}' is missing in the soil data."
                        )
                    else:
                        continue
                validated_columns.extend(candidate)
            else:
                # For a string key, check using lower-case comparison.
                matching_cols = [
                    col for col in data.columns if key.lower() in col.lower()
                ]
                if len(matching_cols) == 0:
                    if mandatory:
                        raise ValueError(
                            f"Soil input: '{key}' is missing in the soil data."
                        )
                    else:
                        continue
                elif len(matching_cols) > 1:
                    raise ValueError(
                        f"'{key}' should be defined by a single column, found: {matching_cols}"
                    )

                original = keys_lower[key.lower()]
                if original != key:
                    data.rename(columns={original: key}, inplace=True)
                validated_columns.append(key)
        return validated_columns

    @classmethod
    def lateral(
        cls,
        df: pd.DataFrame,
        option: str,
        mudline: Union[float, None] = None,
        pw: float = 1.025,
    ) -> pd.DataFrame:
        """Process soil profile data to ensure that the required inputs for lateral
        soil reaction modeling are present based on the specified option.

        The method uses a pre-defined set of keys stored in the LATERAL_SSI_KEYS
        dictionary. Each option defines two categories:
        - mandatory: columns that must be present in the DataFrame.
        - optional: columns that will be included if they are present.

        Available options: {"apirp2geo", "pisa"}.

        If any mandatory key defined for the selected option is missing from
        the DataFrame, a KeyError will be raised. The returned DataFrame will
        include the mandatory keys and any optional keys that exist in the
        input.

        :param df: DataFrame containing the soil profile data.
        :param option: String specifying the option to model the lateral soil
            reaction. The option must be one of the available options (e.g.,
            "apirp2geo" or "pisa").
        :param mudline: float, sea bed level in mLAT coordinates (default=None).
        :param pw: float, sea water density (default=1.025 t/m3)
        :return: Filtered DataFrame containing only the required columns.
        :raises NotImplementedError: If the provided option is not supported.
        :raises KeyError: If one or more mandatory columns are missing.
        """
        available_options = cls.get_available_options(loading="lateral")
        if option not in available_options:
            raise NotImplementedError(f"Option '{option}' not supported.")

        key_db = cls.LATERAL_SSI_KEYS[option]
        # Mandatory keys for the selected option.
        _keys = key_db.get("mandatory", [])
        mandatory_keys = cls._validate_keys(
            data=df, required_keys=_keys, mandatory=True
        )
        # Include optional keys that are present.
        _keys = key_db.get("optional", [])
        optional_keys = cls._validate_keys(
            data=df, required_keys=_keys, mandatory=False
        )
        soilprofile = df[mandatory_keys + optional_keys].copy()
        # Add additional required info
        soilprofile = cls._add_soilinfo(soilprofile, pw, mudline)

        return soilprofile

    @staticmethod
    def _add_soilinfo(
        df: pd.DataFrame, pw: float, mudline: Union[float, None]
    ) -> pd.DataFrame:
        """
        Add additional soil information to the soil profile DataFrame. The
        method calculates the submerged unit weight of the soil and, if provided,
        the mudline depth in mLAT coordinates.

        :param df: DataFrame containing the soil profile data.
        :param pw: float, sea water density (default=1.025 t/m3).
        :param mudline: float, sea bed level in mLAT coordinates (default=None).
        :return: DataFrame with the added columns.
        """
        # Add submerged unit weight
        acc_gravity = 9.81  # acceleration due to gravity (m/s2)
        for col in df.columns:
            if "Total unit weight" in col:
                new_col = col.replace("Total unit weight", "Submerged unit weight")
                df[new_col] = df[col] - pw * acc_gravity

        # Add mudline depth in mLAT coordinates
        if mudline:
            df["Elevation from [mLAT]"] = mudline - df["Depth from [m]"]
            df["Elevation to [mLAT]"] = mudline - df["Depth to [m]"]

        return df
