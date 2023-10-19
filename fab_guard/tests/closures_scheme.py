import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String

import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config


class ClosuresScheme(pa.DataFrameModel):
    closure_type: Series[pa.String] = pa.Field(isin=["location", "country", "links"], alias='#closure_type')
    name1: Series[pa.String] = pa.Field()
    name2: Series[pa.String] = pa.Field(nullable=True)
    closure_start: Series[int] = pa.Field(nullable=True)
    closure_end: Series[int] = pa.Field(nullable=True)

    @pa.dataframe_check()
    def closure_type_country(cls, df: pd.DataFrame) -> Series[bool]:
        dfl = fg.FabGuard.get_instance().load_file(config.locations)
        loc_countries = dfl["country"].tolist()
        mask = ((df["#closure_type"] == "country")
                & (~df["name1"].isin(loc_countries)
                | (~df["name2"].isin(loc_countries))))

        if mask.any():  # Check if any rows meet the condition
            raise ValueError(
                f"Invalid data: "
                f"If closure_type is country,\n "
                f"then name1 and name1 should be in location.country.\n"
                f"Invalid rows: {df.index[mask]}"
            )
        return ~mask