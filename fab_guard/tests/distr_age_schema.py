import os
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import Series, String, DataFrame

import plugins.FabFlee.fab_guard.fab_guard as fg
from plugins.FabFlee.fab_guard.error_messages import Errors
import plugins.FabFlee.fab_guard.config as config


class DistrAgeScheme(pa.DataFrameModel):
    # name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')
    Age: Series[pa.Float] = pa.Field(coerce=True, nullable=True,
                                   in_range={"min_value": 0, "max_value": 120})

    @pa.dataframe_check
    def all_but_first_column_sum_is_100(cls, df: DataFrame) -> bool:
        # Iterate over the names of all columns except the first one
        errors = []
        for column_name in df.columns[1:]:
            column_sum = df[column_name].sum()
            if column_sum != 100:
                errors.append(f"{column_name},{column_sum}")
        if len(errors) > 0:
            raise ValueError(Errors.sum_of_columns_is_100(errors, config.distr_age))
        return True

    def __new__(cls, *args, **kwargs):
        # Dynamically add fields based on a provided DataFrame
        print("Inside")
        print(**kwargs)
        if 'df' in kwargs:
            df = kwargs['df']
            float_field = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 100})
            for column in df.columns[1:]:
                setattr(cls, column, Series[pa.Float](float_field))
        return super(DistrAgeScheme, cls).__new__(cls)

    @classmethod
    def with_dynamic_columns(cls, sample_df: pd.DataFrame):
        class ExtendedDistrAgeScheme(DistrAgeScheme):
            pass

        # Define common constraints for the additional columns
        float_field = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 100})

        # Add fields dynamically to the ExtendedDistrAgeScheme class
        for column in sample_df.columns[1:]:
            setattr(ExtendedDistrAgeScheme, column, Series[pa.Float](float_field))

        return ExtendedDistrAgeScheme

    @classmethod
    def with_dynamic_columns_old(cls, df: pd.DataFrame):
        # Define the common constraints
        age_fieled = pa.Field(coerce=True, nullable=True,
                             in_range={"min_value": 0, "max_value": 120})

        all_other_fields = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 100})
        # Create a new class dynamically
        dynamic_attrs = {'__annotations__': {'Age': Series[pa.Float]}}
        dynamic_attrs['Age'] = age_fieled

        # Retrieve existing annotations and fields from DistrAgeScheme

        # Iterate over the column names, skipping the first one
        for column in df.columns[1:]:
            dynamic_attrs['__annotations__'][column] = Series[pa.Float]
            all_other_fields = pa.Field(coerce=True, in_range={"min_value": 0, "max_value": 100})
            dynamic_attrs[column] = all_other_fields

        # Create a new class with the dynamic columns
        # return type('ExtendedDistrAgeScheme', (DistrAgeScheme,), dynamic_attrs)
        return type('ExtendedDistrAgeScheme', (DistrAgeScheme,), dynamic_attrs)
