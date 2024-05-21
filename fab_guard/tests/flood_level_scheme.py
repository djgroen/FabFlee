import os
import pandas as pd
import pandera as pa
import yaml
from pandera.typing import Series, String
import plugins.FabFlee.fab_guard.fab_guard as fg
import plugins.FabFlee.fab_guard.config as config


def get_sim_period_len():
    sdf = fg.FabGuard.get_instance().load_file(config.sim_period)
    tdf = fg.FabGuard.get_instance().transpose(sdf)
    return tdf["Length"][0]-1

def get_settings_flood_level():
    settings_file = os.path.join(fg.FabGuard.get_instance().input_dir, config.simsettings)
    with open(settings_file, 'r') as val_yaml:
        setting = yaml.load(val_yaml, Loader=yaml.SafeLoader)
        return setting["move_rules"]["max_flood_level"]

def is_increment_of_step(series: pd.Series, step: float, min_value: float, max_value: float) -> pd.Series:
    # Check if each value is an increment of `step` within the range [min_value, max_value]
    return ((series - min_value) % step == 0) & (series >= min_value) & (series <= max_value)


class FloodLevelScheme(pa.DataFrameModel):
    # name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')
    Day: Series[pa.Int] = pa.Field(nullable=False)

    @pa.check("Day")
    def check_day_increment(cls, series: Series[int]) -> Series[bool]:
        min_value = 0  # define your min value
        max_value = get_sim_period_len()  # define your max value
        step = 1  # define the step increment

        # Check if each value is an increment of `step` within the range [min_value, max_value]
        return ((series - min_value) % step == 0) & (series >= min_value) & (series <= max_value)

    @classmethod
    def with_dynamic_columns_old(cls, df: pd.DataFrame):
        class ExtendedFloodLevelScheme(FloodLevelScheme):
            pass
        max_flood_level = get_settings_flood_level()
        length = get_sim_period_len()
        # Define constraints for all flood zone columns
        flood_level_field = pa.Field(coerce=True,
                                     in_range={"min_value": 0, "max_value": length})
        # Create constraint for the first column
        day_level_field = pa.Field(coerce=True,
                                     in_range={"min_value": 0, "max_value": 2})
        print(length)
        # Create the first column
        #setattr(ExtendedFloodLevelScheme, df.columns[0], Series[pa.Int](day_level_field))

        # Add fields dynamically to the ExtendedDistrAgeScheme class
        for column in df.columns[1:]:
            # Create flood zone columns
            setattr(ExtendedFloodLevelScheme, column, Series[pa.Int](flood_level_field))

        return ExtendedFloodLevelScheme

    @classmethod
    def with_dynamic_columns(cls, df: pd.DataFrame):
        # Define the common constraints
        day_max_value = get_sim_period_len()
        flood_zone_max_value = get_settings_flood_level()
        day_fieled = pa.Field(coerce=True, nullable=True,
                              in_range={"min_value": 0, "max_value": day_max_value})

                # Create a new class dynamically
        dynamic_attrs = {'__annotations__': {'Day': Series[pa.Int]}}
        dynamic_attrs['Day'] = day_fieled

        # Retrieve existing annotations and fields from DistrAgeScheme

        # Iterate over the column names, skipping the first one
        for column in df.columns[1:]:
            dynamic_attrs['__annotations__'][column] = Series[pa.Float]
            all_other_fields = pa.Field(coerce=True,
                                        in_range={"min_value": 0, "max_value": flood_zone_max_value})
            dynamic_attrs[column] = all_other_fields

        # Create a new class with the dynamic columns
        # return type('ExtendedDistrAgeScheme', (DistrAgeScheme,), dynamic_attrs)
        return type('ExtendedFloodLevelScheme', (FloodLevelScheme,), dynamic_attrs)