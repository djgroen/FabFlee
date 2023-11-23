"""
Input:
- country: Name of the country or dataset (e.g., "nigeria2016").
- start_date: The starting date to consider when calculating conflict periods (e.g., "1-1-2016").
- end_date: The ending date to limit the number of days in the conflicts.csv file (e.g., "31-12-2016").

Description:
This script processes conflict information for a specified country and generates a conflicts.csv file. It reads conflict
data from the "conflict_info.csv" file, which includes location names, their corresponding start dates, and conflict periods. The script
calculates the number of days between the start_date and end_date, then creates a DataFrame with a range of days as columns.
It populates the DataFrame with 1s for days that fall within the conflict periods of each location and 0s for the rest.

Usage:
1. Prepare conflict information data for the desired country and save it as a CSV file ("conflict_info.csv").
2. Ensure the "conflict_info.csv" file includes columns for "name," "start_date," and "conflict_period."
3. Specify the desired parameters:
   - <country>: Name of the country or dataset (e.g., "nigeria2016").
   - <start_date>: The starting date to consider when calculating conflict periods (e.g., "1-1-2016").
   - <end_date>: The ending date to limit the number of days in the conflicts.csv file (e.g., "31-12-2016").

Command:
Run the script using the following command:
"python extract_conflicts.py <country> <start_date> <end_date>"

Example Usage:
"python extract_conflicts.py nigeria2016 1-1-2016 31-12-2016"

Output:
The script generates a "conflicts.csv" file containing conflict data for the specified country and date range.

Note:
The generated conflicts.csv file will have columns representing days and 1s or 0s indicating conflict presence or absence
for each location within the defined date range.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Function to calculate the number of days between two dates in "dd-mm-yyyy" format
def between_date(d1, d2):
    d1list = d1.split("-")
    d2list = d2.split("-")
    date1 = datetime(int(d1list[2]), int(d1list[1]), int(d1list[0]))
    date2 = datetime(int(d2list[2]), int(d2list[1]), int(d2list[0]))

    return abs((date1 - date2).days)

def extract_conflicts(country, start_date, end_date):
    # Get the current directory
    current_dir = os.getcwd()

    # Load conflict info from log.csv
    conflict_info_file = os.path.join(current_dir, country, "conflict_info.csv")
    conflict_info_df = pd.read_csv(conflict_info_file)

    # Calculate the number of days between start_date and end_date
    period = between_date(start_date, end_date)

    conflict_zones = conflict_info_df["name"].tolist()

    # Create a DataFrame to store the conflicts data
    data = {'day': list(range(period + 1))}  # +1 to include end_date
    data.update({zone: [0] * (period + 1) for zone in conflict_zones})
    df = pd.DataFrame(data)

    # Loop through rows and update the conflicts DataFrame
    for index, row in conflict_info_df.iterrows():
        location = row['name']
        date = row['start_date']
        days = row['conflict_period']

        # Convert the start_date to datetime format
        start_datetime = datetime.strptime(date, "%d-%m-%Y")

        # Calculate the index at which to start marking conflicts
        start_index = (start_datetime - datetime.strptime(start_date, "%d-%m-%Y")).days

        # Calculate the end index of conflicts
        end_index = start_index + days

        # Ensure that the end_index does not exceed the period
        if end_index > period:
            end_index = period

        # Update the corresponding columns with 1s for conflict days
        df.loc[start_index:end_index, location] = 1

    # Print dataframe
    print(df.to_string(index=False))

    # Save the conflicts DataFrame to a CSV file
    df.to_csv(os.path.join(current_dir, country, "conflicts.csv"), index=False)

country = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]

extract_conflicts(country, start_date, end_date)
