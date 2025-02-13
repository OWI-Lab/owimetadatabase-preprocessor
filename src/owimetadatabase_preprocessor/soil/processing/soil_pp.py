"""
This module defines the SoilDataProcessor class which provides helper routines 
for processing soil data.  It is used to transform coordinates, combine raw
and processed DataFrames, and extract/convert in-situ test detail data.
"""
import pandas as pd 
import warnings 
from typing import Dict, List, Tuple, Union
from pyproj import Transformer
from groundhog.general.soilprofile import profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing

class SoilDataProcessor:
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
            "epsg:4326", 
            f"epsg:{target_srid}", 
            always_xy=True
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
    def _combine_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    def _process_insitutest_dfs(
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
                processed_dfs[key] = processed_dfs[key].apply(lambda x: pd.to_numeric(x, errors="ignore"))
            except Exception as err:
                warnings.warn(f"Numeric conversion warning for {key}: {err}")
        return processed_dfs

    @staticmethod
    def _gather_data_entity(
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
    def _process_cpt(df_sum: pd.DataFrame, df_raw:pd.DataFrame, **kwargs):
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
    def _convert_to_profile(df_sum, df_detail, profile_title, drop_info_cols):
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
    def _fulldata_processing(
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
    def _partialdata_processing(unitdata, row, selected_depths, selected_tests):
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
    def _objects_to_list(selected_obj, func_get_detail, data_type):
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