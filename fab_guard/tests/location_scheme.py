import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String

import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config

class LocationsScheme(pa.DataFrameModel):
    # name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')
    name: Series[pa.String] = pa.Field(nullable=False)
    region: Series[pa.String] = pa.Field()
    country: Series[pa.String] = pa.Field()
    lat: Series[pa.Float] = pa.Field()
    lon: Series[pa.Float] = pa.Field()
    location_type: Series[pa.String] = pa.Field(
        isin = ["conflict_zone", "town","camp", "forwarding_hub", "marker", "idpcamp"])
    conflict_date: Series[float] = pa.Field(nullable=True, coerce=True)
    population: Series[float] = pa.Field(ge=0,nullable=True,coerce=True)

    # Define column-level validation check, constraint applies to all values in a column
    @pa.check(name,element_wise=True)
    def names_in_routes(cls, name):
        # Load the content of the routes file in a dataframe
        dfr = fg.FabGuard.get_instance().load_file(config.routes)
        # Convert the content of the '#"name1"' column to a list
        rnames = dfr["name1"].tolist()
        # Convert the content of the "name2"' column to a list
        rnames.extend(dfr["name2"].tolist())
        # Check if the name column is in the either name1 or name2 columns
        return name in rnames


    # Coordinate validation check
    @pa.dataframe_check()
    def coords_are_real(cls,df: pd.DataFrame) -> Series[bool]:
        mask = ((df["lat"] < 180.0) & (df["lat"] > -180.0)
                & (df["lon"] < 180.0) & (df["lon"] > -180.0))

        # Filter the DataFrame to keep only valid rows
        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_coord_err(df.index[mask], config.locations))
        return ~mask
        

    # Coordinate validation check
    @pa.dataframe_check()
    def coords_not_on_null_island(cls,df: pd.DataFrame) -> Series[bool]:
        mask = (abs(df["lat"]) > 0.001 | abs(df["lon"]) > 0.001)

        # Filter the DataFrame to keep only valid rows
        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_coord_err(df.index[mask], config.locations))
        return ~mask


    # Define another data-level validation check
    @pa.dataframe_check()
    def population_gt_0(cls,df: pd.DataFrame)->Series[bool]:
        # Define conditions based on 'location_type' and 'population' columns
        mask = ((df["location_type"] == "camp") & (df["population"] <= 0)
                 | ((df["location_type"] == "town") & (df["population"] <= 0))
                 | ((df["location_type"] == "conflict_zone") & (df["population"] <= 0))
                 | ((df["location_type"] == "marker") & (df["population"] != 0))
                 | ((df["location_type"] == "forwarding_hub") & (df["population"] < 0)))

        # Filter the DataFrame to keep only valid rows
        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_population_err(df.index[mask], config.locations))
        return ~mask

    # Define another data-level validation check
    @pa.dataframe_check(ignore_na=False)
    def conflict_zone_has_conflict_date(cls,df: pd.DataFrame)->Series[bool]:
        # Check if there are missing values in 'conflict_date' when 'location_type' is 'conflict_zone'
        mask = ((df["location_type"] == "conflict_zone") & (pd.isnull(df["conflict_date"])))

        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_conflict_zone_err(df.index[mask], config.locations))
        return ~mask

    @pa.dataframe_check()
    def conflict_zone_country_should_be_0(cls, df: pd.DataFrame) -> Series[bool]:
        # Determine the country value in the first row
        country = df["country"][0]
        # Check if 'country' is not equal to the determined value when 'location_type' is 'conflict_zone'
        mask = ((df["location_type"] == "conflict_zone") & (df["country"]!=country))

        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_country_err(df.index[mask], config.locations))
        return ~mask

