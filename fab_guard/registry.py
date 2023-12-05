from plugins.FabFlee.fab_guard.fab_guard import fgcheck
import plugins.FabFlee.fab_guard.config as config
from plugins.FabFlee.fab_guard.tests import *
import os
import glob
import yaml


def is_dflee(self):
    settings_file = os.path.join(self.input_dir, config.simsettings)
    outer_key = "move_rules"
    inner_key = "max_flood_level"
    with open(settings_file, 'r') as val_yaml:
        setting = yaml.load(val_yaml, Loader=yaml.SafeLoader)
        # check if a simulation is for flooding
        return (outer_key in setting) and (setting[outer_key] is not None) and (inner_key in setting[outer_key])



@fgcheck
def test_all_files(self):
    if is_dflee(self):
        self.register_for_test(location_flood_scheme.LocationsFloodScheme, config.locations)
        self.register_for_test(flood_level_scheme.FloodLevelScheme, config.flood_level)
        test_all_demograohic_file(self)
    else:
        self.register_for_test(location_scheme.LocationsScheme, config.locations)
    self.register_for_test(routes_scheme.RoutesScheme, config.routes)
    self.register_for_test(closures_scheme.ClosuresScheme, config.closures)


def test_all_demograohic_file(self):
    if is_dflee(self):
        full_path_pattern = os.path.join(self.input_dir, config.demograohic_files_pattern)
        matching_files = glob.glob(full_path_pattern)
        for file in matching_files:
            self.register_for_test(demographic_scheme.DemographicScheme, file)

#@fgcheck
#def test_all_files(self):
    #self.register_for_test(distr_age_schema.DistrAgeScheme, config.distr_age)

