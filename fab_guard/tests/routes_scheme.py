import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String

class RoutesScheme(pa.DataFrameModel):
    name1: Series[pa.String] = pa.Field(nullable=False)
    name2: Series[pa.String] = pa.Field(nullable=False)
    distance: Series[int] = pa.Field(ge=0)
    forced_redirection: Series[float] = pa.Field(
        isin=[0, 1, 2], nullable=True, coerce=True)
