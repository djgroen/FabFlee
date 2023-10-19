import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String

import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config


class LocationsScheme(pa.DataFrameModel):
    name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')
    #name: Series[pa.String] = pa.Field(nullable=False)
    region: Series[pa.String] = pa.Field()
    country: Series[pa.String] = pa.Field()
    lat: Series[pa.Float] = pa.Field()
    lon: Series[pa.Float] = pa.Field()
    location_type: Series[pa.String] = pa.Field(
        isin = ["conflict_zone", "town","camp", "forwarding_hub", "marker", "idpcamp"])
    conflict_date: Series[float] = pa.Field(nullable=True)
    population: Series[float] = pa.Field(ge=0,nullable=True)

    @pa.check(name,element_wise=True)
    def names_in_routes(cls, name):
        dfr = fg.FabGuard.get_instance().load_file(config.routes)
        rnames = dfr['#"name1"'].tolist()
        rnames.extend(dfr["name2"].tolist())
        return name in rnames

    @pa.dataframe_check()
    def population_gt_0(cls,df: pd.DataFrame)->Series[bool]:
        mask = ((df["location_type"] == "camp") & (df["population"] <= 0)
                 | ((df["location_type"] == "town") & (df["population"] <= 0))
                 | ((df["location_type"] == "conflict_zone") & (df["population"] <= 0))
                 | ((df["location_type"] == "marker") & (df["population"] != 0))
                 | ((df["location_type"] == "forwarding_hub") & (df["population"] < 0)))

        # Filter the DataFrame to keep only valid rows
        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_population_err(df.index[mask], config.locations))
        return ~mask

    @pa.dataframe_check(ignore_na=False)
    def conflict_zone_has_conflict_date(cls,df: pd.DataFrame)->Series[bool]:
        mask = ((df["location_type"] == "conflict_zone") & (pd.isnull(df["conflict_date"])))

        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_conflict_zone_err(df.index[mask], config.locations))
        return ~mask

    @pa.dataframe_check()
    def conflict_zone_country_should_be_0(cls, df: pd.DataFrame) -> Series[bool]:
        country = df["country"][0]
        mask = ((df["location_type"] == "conflict_zone") & (df["country"]!=country))

        if mask.any():  # Check if any rows meet the condition
            raise ValueError(Errors.location_country_err(df.index[mask], config.locations))
        return ~mask

