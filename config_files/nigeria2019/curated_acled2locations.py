import pandas as pd
import warnings
import sys
import os
import calendar as cal
from datetime import datetime

# To run this script, simply run this command:
# pyhton3 curated_acled2locations.py <country_name> <start_date> <end_date> <admin_level>

def drop_rows(inputdata, columnname, dropparameter):
    removedrows = inputdata.index[
        inputdata[columnname] < dropparameter].tolist()
    outputdata = inputdata.drop(removedrows)
    return outputdata

def acled2locations(input_file, country, start_date, end_date, admin_level):
    warnings.filterwarnings('ignore')

    print("Current Path: ", input_file)
    try:
        tempdf = pd.read_csv(input_file)
    except:
        print("Runtime Error: File Cannot be found")


    tempdf.columns = map(str.lower, tempdf.columns)   

    df = tempdf[["event_date", "country", "admin1", "admin2",
                 "location", "latitude", "longitude", "fatalities", "population_best"]]


    df['event_date'] = pd.to_datetime(df['event_date'])

    df = drop_rows(df, 'fatalities', 1)

    df = df[(df['event_date'] >= start_date)]
    df = df[(df['event_date'] < end_date)]



    df = df.sort_values("event_date").drop_duplicates(admin_level)

    df = df.loc[(df.country == country.title())]

    df.rename(columns={'event_date': 'conflict_date'}, inplace=True)



    start_date = datetime.strptime(start_date,"%Y-%m-%d")

    df['conflict_date']=(df['conflict_date']-start_date).dt.days


    # Exporting CSV to locations.csv
    output_df = df[['admin1', 'country',
                    'latitude', 'longitude', 'conflict_date', 'population_best']]
    output_df.rename(columns={'admin1': '#name',
                              'population_best': 'population'}, inplace=True)
    output_df["location_type"] = "conflict_zone"
    #output_df["population"] = "0"
    output_df = output_df[
        ['#name', 'country', 'latitude',
         'longitude', 'location_type', 'conflict_date',
         'population']
    ]
    output_df = output_df.loc[output_df.groupby('#name')['conflict_date'].idxmin()]

    output_df['#name'] = output_df['#name'].str.replace(" ", "_")
    output_df['#name'] = output_df['#name'].str.replace("-", "_")





    output_file = os.path.join(work_dir,"locations.csv")

    try:
        output_df.to_csv(output_file, index=False, mode='x')
    except FileExistsError:
        print("File Already exists, saving as new_locations.csv")
        output_file = os.path.join(work_dir,"new_locations.csv")
        output_df.to_csv(output_file, index=False, mode='x')


if __name__ == '__main__':

    work_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(work_dir,"acled.csv")

    country = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    admin_level = sys.argv[4]


    acled2locations(input_file, country, start_date, end_date, admin_level)