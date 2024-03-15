'''
Input: ACLED data (acled.csv)
Output: Location Names

Description:
This script processes ACLED conflict data for a specified country and extracts unique location names based on a specified location type (e.g., "admin2" for administrative region level 2). 
It provides a grid-like display of the extracted location names for reference when working with population data.

Usage:
1. Prepare ACLED conflict data for the desired country and save it as a CSV file (acled.csv).
2. Specify the desired location type (e.g., "admin2") for which you want to extract location names.

Command:
Run the script using the following command:
"python 00_extract_location_names.py <country> <location_type>"

- <country>: Name of the country or dataset (e.g., nigeria2016).
- <location_type>: The type of location for which you want to extract names (e.g., "admin2" for administrative region level 2).

Example Usage:
"python 00_extract_ACLED_location_names.py nigeria2016 admin2"
'''

import os
import sys
import pandas as pd


def extract_location_names(country, location_type):

    # Check if the directory already exists
    if not os.path.exists(country):
        # Create the directory if it doesn't exist
        os.makedirs(country)
        print(f"Directory '{country}' created.")
    else:
        print(f"Directory '{country}' already exists.")

    # Get the current directory
    current_dir = os.getcwd()

    acled_file = os.path.join(current_dir, country, "acled.csv")
    
    # Load the ACLED data from acled.csv into a DataFrame
    acled_df = pd.read_csv(acled_file)

    # Sort dataframe by location type
    acled_df = acled_df.sort_values(location_type)

    if location_type in acled_df.columns:
        print(f"Here are locations found in '{location_type}'. Please find a population-table accordingly.")

        # Extract and print the unique location names
        location_names = acled_df[location_type].unique()

        # Calculate the maximum length of location names
        max_length = max(len(name) for name in location_names)

        # Define the column width (you can adjust this as needed)
        column_width = max_length + 4  # Add extra spaces for padding

        # Initialise a counter to keep track of column position
        column_counter = 0

        # Print the location names in a grid
        for name in location_names:
            print(name.ljust(column_width), end=' ')
            column_counter += 1

            # Start a new row after every 3 columns
            if column_counter == 4:
                print()
                column_counter = 0  # Reset the counter for the next row
    else:
        print(f"The column '{location_type}' does not exist in the DataFrame.")


country = sys.argv[1]
location_type = sys.argv[2]


extract_location_names(country, location_type)
