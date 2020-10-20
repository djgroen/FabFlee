import argparse
import pandas as pd
from pprint import pprint
import subprocess
import os
import math


def run_City_Graph_app(cityGraph_location,
                       API_KEY,
                       COUNTRY_CODE,
                       POPULATION_LIMIT,
                       CITIES_LIMIT):

    subprocess.call(['java', '-jar', 'citygraph.jar',
                     '-a', API_KEY,
                     COUNTRY_CODE,
                     str(POPULATION_LIMIT),
                     str(CITIES_LIMIT)
                     ],
                    cwd=cityGraph_location
                    )

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityGraph_location', action="store", required=True)
    parser.add_argument('--API_KEY', action="store", required=True)
    parser.add_argument('--POPULATION_LIMIT',
                        action="store", type=int, required=True)
    parser.add_argument('--CITIES_LIMIT', action="store",
                        type=int, required=True)
    parser.add_argument('--config_location', action="store", required=True)
    parser.add_argument('--config_name', action="store", required=True)
    args = parser.parse_args()

    print("Input Arguments :")
    print("\tcityGraph_location : %s" % (args.cityGraph_location))
    print("\tAPI_KEY : %s" % (args.API_KEY))
    print("\tPOPULATION_LIMIT : %d" % (args.POPULATION_LIMIT))
    print("\tCITIES_LIMIT : %d" % (args.CITIES_LIMIT))
    print("\tconfig_location : %s" % (args.config_location))
    print("\tconfig_name : %s" % (args.config_name))

    cityGraph_location = args.cityGraph_location
    API_KEY = args.API_KEY
    POPULATION_LIMIT = args.POPULATION_LIMIT
    CITIES_LIMIT = args.CITIES_LIMIT
    config_location = args.config_location
    config_name = args.config_name

    CountryCodesList = "https://raw.githubusercontent.com/qusaizakir/CityGraph/master/listofcountrycodes.csv"
    CountryCodesList = pd.read_csv(CountryCodesList)
    # convert country to lowercase
    CountryCodesList['country'] = CountryCodesList['country'].str.lower()
    # find the country code from list
    COUNTRY_CODE = CountryCodesList.loc[
        CountryCodesList.country == config_name, 'iso2'].iloc[0]

    run_City_Graph_app(cityGraph_location, API_KEY,
                       COUNTRY_CODE, POPULATION_LIMIT, CITIES_LIMIT)

    config_location_csv = pd.read_csv(
        os.path.join(config_location,
                     "input_csv",
                     "locations.csv")
    )

    citygraph_location_csv = pd.read_csv(
        os.path.join(cityGraph_location,
                     "%s_locations.csv" % (COUNTRY_CODE))
    )

    citygraph_location_csv = citygraph_location_csv.rename(
        columns={'name': '#name'})

    merged_csv = config_location_csv
    for index, config_location_row in merged_csv.iterrows():
        location_name = config_location_row['#name']
        population_size = config_location_row['population']
        citygraph_loc_row = citygraph_location_csv.loc[
            citygraph_location_csv['#name'] == location_name]
        if citygraph_loc_row is not None and \
                isinstance(citygraph_loc_row, pd.DataFrame) and \
                not citygraph_loc_row.empty:
            new_population_size = citygraph_loc_row['population']
            '''
            if locations.csv has population number == 0 and 
                matches names in BI_locations.csv then 
                    replace with number in BI_location.csv 
            if locations.csv has population number for location > 0  and 
                name matches then 
                    DO NOT replace with BI_locations.csv
            if it doesn't have population count in BI_location (names do not match) then 
                leave it as it is in locations.csv            
            '''

            if (math.isnan(population_size) or population_size == 0):
                merged_csv.at[index, 'population'] = new_population_size
                print("population_size for = %s updated : %f -> %f"
                      % (location_name, population_size, new_population_size))

    merged_csv.to_csv(
        os.path.join(config_location,
                     "input_csv",
                     "locations_pop.csv"),
        index=False
    )
