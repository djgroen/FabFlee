import pandas as pd
import warnings
import sys
import os
import calendar as cal
from datetime import datetime


def month_convert(month):
    name_to_number = {name: number for number, name in enumerate(cal.month_name) if number}  # dict month : month_num
    month_num = name_to_number.get(month)  # month number in int form
    return month_num


def date_format(in_date):  # converting date from textbased to dd-mm-yyyy format
    split_date = in_date.split()
    month_num = month_convert(split_date[1])
    out_date = split_date[0] + "-" + str(month_num) + "-" + split_date[2]
    return out_date


def between_date(d1, d2):  # Gets difference between two dates in string format "dd-mm-yy"
    d1list = d1.split("-")
    d2list = d2.split("-")
    date1 = datetime(int(d1list[2]), int(d1list[1]), int(d1list[0]))
    date2 = datetime(int(d2list[2]), int(d2list[1]), int(d2list[0]))

    return abs((date1 - date2).days) + 1  # Maybe add +1


def date_verify(date):
    date_format = "%d-%m-%Y"
    try:
        date_obj = datetime.strptime(date, date_format)
        return True

    except ValueError:
        print("Incorrect data format please input dd-mm-yyyy")
        return False

def drop_rows(inputdata, columnname, dropparameter):
    removedrows = inputdata.index[inputdata[columnname] == dropparameter].tolist()
    outputdata = inputdata.drop(removedrows)
    return outputdata

def filter_table(df,colname,adminlevel):
    if adminlevel == "admin1":  adminlist=df.admin1.unique()
    elif adminlevel == "location": adminlist=df.location.unique()
    else: adminlist=df.admin2.unique()
    newdf = pd.DataFrame(columns=df.columns)

    for admin in adminlist:
        tempdf = df.loc[df[adminlevel] == admin]
        tempdf.sort_values(colname, ascending=True)
        newdf = newdf.append(tempdf.tail(1))
    print(newdf)
    return newdf

def find_csv(country):
    path_to_dir = os.getcwd()
    print(path_to_dir)
    filename = country + "-acled.csv"
    locations = os.path.join("config_files",country,
                             "source_data",filename)
    print(locations)

    return locations

#Takes path to acled csv file, a start date in dd-mm-yyyy format, and a filter (First occurence or highest fatalities)
def main(fab_flee_loc,country, start_date,filter,admin_level):
    warnings.filterwarnings('ignore')
    input_file = os.path.join(fab_flee_loc,"config_files",
                            country,
                            "acled.csv")
    print("Current Path: ",input_file)
    try:
        tempdf = pd.read_csv(input_file)
    except:
        print("Runtime Error: File Cannot be found")
    df = tempdf[["event_date","country","admin1","admin2",
                 "location","latitude","longitude","fatalities"]]
    # event_date is given in incorrect format, so formatting to dd-mm-yyyy required
    event_dates = df["event_date"].tolist()
    formatted_event_dates = [date_format(date) for date in event_dates]
    conflict_dates = [between_date(d, start_date) for d in formatted_event_dates]
    # replacing event_date
    df.loc[:, "event_date"] = conflict_dates
    df.rename(columns={'event_date': 'conflict_date'}, inplace=True)

    df = drop_rows(df, 'fatalities', 0)
    if filter == 'earliest':
        filter = 'conflict_date'

    try:
        df = filter_table(df,filter,admin_level)

    except:
        print("Runtime error: Filter value must be earliest or fatalities")
    # Exporting CSV to locations.csv
    output_df = df[['location', 'admin1', 'country', 'latitude', 'longitude', 'conflict_date']]
    output_df.rename(columns={'location': 'name', 'admin1': 'reigon'}, inplace=True)
    output_df["location_type"] = "conflict_zone"
    output_df["population"] = "null"
    output_file = os.path.join(fab_flee_loc, "config_files",
                              country,
                              "locations.csv")

    try:
        output_df.to_csv(output_file, index=False, mode='x')
    except FileExistsError:
        print("File Already exists, saving as new_locations.csv")
        output_file = os.path.join(fab_flee_loc, "config_files",
                                   country,
                                   "new_locations.csv")
        output_df.to_csv(output_file, index=False, mode='x')



if __name__ == '__main__':

    fabflee = sys.argv[1]
    country = sys.argv[2]
    start_date = sys.argv[3]
    filter = sys.argv[4]
    adminlevel = sys.argv[5]
    main(fabflee,country,start_date,filter,adminlevel)
