import json
import os

import pandas as pd
import pytest
from groundhog.general.soilprofile import profile_from_dataframe

from owimetadatabase_preprocessor.soil import SoilDataProcessor, SoilprofileProcessor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.mark.parametrize(
    "filename, mudline",
    [
        ("api_ok_1.json", None),  # "from" and "to" keys
        ("api_ok_1.json", -26.0),  # mudline provided [mLAT]
        ("api_ok_2.json", None),  # cte keys
        ("api_ok_3.json", None),  # Su #from and #to keys; no "Dr" (optional)
    ],
)
def test_lateral_api2rpgeo(filename: str, mudline: float) -> None:
    """Test lateral method with option 'apirp2geo' for various mudline values."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Execute lateral with option 'apirp2geo' and given mudline
    result = SoilprofileProcessor.lateral(
        df, option="apirp2geo", mudline=mudline, pw=1.025
    )

    # Ensure the returned soil profile is not empty
    assert not result.empty, "Returned soil profile is empty."

    # Check that the returned soil profile has the same number of rows as the input
    assert len(result) == len(
        df
    ), "Number of rows in the returned soil profile is different from the input."

    if mudline is None:
        # Check that no column contains "[mLAT]" since no mudline was provided
        for col in result.columns:
            assert (
                "[mLAT]" not in col
            ), f"Column '{col}' should not contain [mLAT] when mudline is None."
    else:
        # When mudline is provided, check that at least one column contains "[mLAT]"
        assert any(
            "[mLAT]" in col for col in result.columns
        ), "No column contains [mLAT] despite mudline being provided."
        assert (
            result.iloc[0]["Elevation from [mLAT]"] == mudline
        ), "Mudline value is not correctly set."
        sp_depth = result.iloc[-1]["Depth to [m]"]
        assert (
            mudline - result.iloc[-1]["Elevation to [mLAT]"] == sp_depth
        ), "Bottom elevation [mLAT] is not correctly set."

    # Check that the returned soil profile has columns which contains "Submerged unit weight" column
    assert any(
        "Submerged unit weight" in col for col in result.columns
    ), "No column contains 'Submerged unit weight'."


@pytest.mark.parametrize(
    "filename, mudline",
    [
        ("api_bad_1.json", None),  # Missing "Depth" keys since "Z" used instead
        ("api_bad_2.json", None),  # Missing "Soil type" keys
        ("api_bad_3.json", None),  # Missing "epsilon50" keys
    ],
)
def test_fail_1_lateral_api2rpgeo(filename: str, mudline: float) -> None:
    """Test lateral method fails appropriately with invalid input data."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Test that the method raises ValueError due to missing mandatory keys
    with pytest.raises(ValueError) as exc_info:
        _ = SoilprofileProcessor.lateral(
            df, option="apirp2geo", mudline=mudline, pw=1.025
        )

    # Verify the error message mentions the missing mandatory key
    assert "missing in the soil data" in str(
        exc_info.value
    ), "Error message does not mention missing mandatory key."


@pytest.mark.parametrize(
    "filename, mudline",
    [
        ("api_bad_1bis.json", None),  # Double "Soil type" key definition
    ],
)
def test_fail_2_lateral_api2rpgeo(
    filename: str, mudline: float
) -> None:  # Changed function name
    """Test lateral method fails appropriately because incorrect parameter
    definition in the input data."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Test that the method raises ValueError due to incorrect parameter definition
    with pytest.raises(ValueError) as exc_info:
        _ = SoilprofileProcessor.lateral(
            df, option="apirp2geo", mudline=mudline, pw=1.025
        )

    # Verify the error message mentions the incorrect parameter definition
    assert "defined by a single column" in str(
        exc_info.value
    ), "Error message does not mention incorrect parameter definition."


@pytest.mark.parametrize(
    "filename, mudline",
    [
        ("pisa_ok_1.json", None),  # Ok
        ("pisa_ok_2.json", None),  # Dr cte value
        ("pisa_ok_3.json", None),  # Dr cte value; gamma linear
    ],
)
def test_lateral_pisa(filename: str, mudline: float) -> None:
    """Test lateral method with option 'pisa'."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Execute lateral with option 'pisa'
    result = SoilprofileProcessor.lateral(df, option="pisa", mudline=mudline, pw=1.025)

    # Ensure the returned soil profile is not empty
    assert not result.empty, "Returned soil profile is empty."

    # Check that the returned soil profile has the same number of rows as the input
    assert len(result) == len(
        df
    ), "Number of rows in the returned soil profile is different from the input."

    # Check that the returned soil profile has columns which contains "Submerged unit weight" column
    assert any(
        "Submerged unit weight" in col for col in result.columns
    ), "No column contains 'Submerged unit weight'."

    # Check that no apirp2geo columns are present
    assert not any(
        "epsilon50" in col for col in result.columns
    ), "Columns from apirp2geo method are present."


@pytest.mark.parametrize(
    "filename_input, filename_output",
    [
        ("convert_input_pisa.json", "convert_output_pisa.json"),
    ],
)
def test_convert_to_profile(filename_input: str, filename_output: str) -> None:
    """Test convert profile lateral method with option 'pisa'.

    The purpose mostly just to ensure dtype consistency.
    """
    filepath_input = os.path.join(DATA_DIR, filename_input)
    filepath_output = os.path.join(DATA_DIR, filename_output)
    with open(filepath_input, "r") as f:
        data_input = json.load(f)
    with open(filepath_output, "r") as f:
        data_output = json.load(f)

    df_input = pd.DataFrame(data_input)
    df_output = pd.DataFrame(data_output)

    cols_not_numeric = {
        "Soil type",
    }
    cols_numeric = df_output.columns.difference(cols_not_numeric)
    df_output[cols_numeric] = df_output[cols_numeric].apply(
        pd.to_numeric, errors="coerce"
    )

    groundhog_profile = profile_from_dataframe(df_output)

    df_temp = pd.DataFrame(
        {
            "location_name": [df_input["location_name"].iloc[0]],
            "title": [df_input["title"].iloc[0]],
        }
    )

    result = SoilDataProcessor.convert_to_profile(df_temp, df_input, None, True)

    pd.testing.assert_frame_equal(result, groundhog_profile, check_like=True)
