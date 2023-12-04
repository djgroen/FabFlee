import string

import plugins.FabFlee.fab_guard.fab_guard as fg

class Errors:
    def location_country_err(invalid_input, file):
        err = f"Invalid data for {file}: If location_type is confict_zone,\n "\
              f"then the coumntry should be the country at position 0 in the file. \n"\
              f"Invalid rows: {invalid_input}"
        return err

    def location_conflict_zone_err(invalid_input, file):
        err = f"Invalid data for file {file}: Only conflict zones have conflict dates \n"\
              f"Invalid rows: {invalid_input}"
        return err

    def location_population_err(invalid_input, file):
        err = f"Invalid data for file {file}: \n"\
              f"For rows where location_type is 'camp, town or conflict_zone', \n" \
              f"population must be greater than 0. \n"\
              f"For rows where location_type is 'marker', population should be 0. \n"\
              f"For rows where location_type is 'forwarding_hub', population should be >=0. \n"\
              f"Invalid rows: {invalid_input}"
        return err

    def closures_type_country_err(invalid_input, file):
        err = f"Invalid data for file {file}: \n" \
                f"If closure_type is country,\n " \
                f"then name1 and name1 should be in location.country.\n"\
                f"Invalid rows: {invalid_input}"
        return err

    def location_coord_err(invalid_input, file):
        err = f"Invalid location coordinates in file {file}: Coordinates point to Null Island or are not in range (-180.0,180,0). \n"\
              f"Invalid rows: {invalid_input}"
        return err

    def sum_of_columns_is_100(invalid_input, file):
        err = f"Invalid sum {file}: The sum of all values in a columns should 100). \n"\
              f"Invalid columns, sum: {invalid_input}"
        return err
