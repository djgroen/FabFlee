import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String

# Import modules from custom packages
import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config

# Define a validation class for the closures.csv file
class ClosuresScheme(pa.DataFrameModel):
    # Define simple constraints for all the columns
    closure_type: Series[pa.String] = pa.Field(
        isin=["location", "country", "links", "camp", "idcamp"])
    name1: Series[pa.String] = pa.Field()
    name2: Series[pa.String] = pa.Field(nullable=True)
    closure_start: Series[pa.Int64] = pa.Field(nullable=True,coerce=True)
    closure_end: Series[pa.Int64] = pa.Field(nullable=True, coerce=True)

    @pa.dataframe_check()
    def closure_type_country(cls, df: pd.DataFrame) -> Series[bool]:
        # Load the content of the "locations" file
        dfl = fg.FabGuard.get_instance().load_file(config.locations)

        # Get a list of countries from the "locations" file
        loc_countries = dfl["country"].tolist()

        # Define a mask to check if the conditions are met
        mask = ((df["closure_type"] == "country")
                & (~df["name1"].isin(loc_countries)
                & (~df["name2"].isin(loc_countries))))

        # Check if any rows meet the condition
        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.closures_type_country_err(
                df.index[mask], config.locations))
        return ~mask