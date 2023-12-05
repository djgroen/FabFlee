import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String, DataFrame

import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config


class DemographicScheme(pa.DataFrameModel):
    # name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')

    @pa.dataframe_check
    def all_but_first_column_sum_is_1(cls, df: DataFrame) -> bool:
        # Iterate over the names of all columns except the first one
        errors = []
        for column_name in df.columns[1:]:
            column_sum = df[column_name].sum()
            if column_sum != 1:
                errors.append(f"{column_name},{column_sum}")
        if len(errors) > 0:
            raise ValueError(Errors.sum_of_columns_is_1(errors, config.demograohic_files_pattern))
        return True


    @classmethod
    def with_dynamic_columns_old(cls, df: pd.DataFrame):
        all_other_fields = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 100})
        # Create a new class dynamically
        dynamic_attrs = {'__annotations__': {}}

        # Iterate over the column names, skipping the first one
        for column in df.columns[1:]:
            dynamic_attrs['__annotations__'][column] = Series[pa.Float]
            all_other_fields = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 1})
            dynamic_attrs[column] = all_other_fields

        # Create a new class with the dynamic columns
        # return type('ExtendedDistrAgeScheme', (DistrAgeScheme,), dynamic_attrs)
        return type('ExtendedDemographicScheme', (DemographicScheme,), dynamic_attrs)
