import pandas as pd
import warnings
import sys
import os
from datetime import datetime

def acled2locations(fab_flee_loc, country, start_date, filter_option, admin_level):
    """
    Processes ACLED conflict data for a given country, filters it based on a
    start date and a specified filter option, and saves the output to a CSV file.

    Args:
        fabflee (str): The plugin directory path.
        country (str): The name of the country.
        start_date (str): The start date for filtering in 'dd-mm-yyyy' format.
        filter_option (str): The filter to apply ('earliest' or 'fatalities').
        admin_level (str): The administrative level to filter by.
    """

    # 1. Define input and output file paths
    input_file_path = os.path.join(fab_flee_loc, "config_files", country, "acled.csv")
    output_dir = os.path.join(fab_flee_loc, "config_files", country, "input_csv")

    # Map filter_option to a single-letter abbreviation
    filter_abbreviations = {'earliest': 'e', 'fatalities': 'f'}
    filter_abbr = filter_abbreviations.get(filter_option, filter_option)
    
    # Conditionally set the output file name based on admin level and filter option
    if admin_level == 'location' and filter_option == 'earliest':
        output_file_name = "locations.csv"
    elif admin_level == 'location' and filter_option == 'fatalities':
        output_file_name = "locations_f.csv"
    else:
        # Map filter_option to a single-letter abbreviation for other admin levels
        filter_abbreviations = {'earliest': 'e', 'fatalities': 'f'}
        filter_abbr = filter_abbreviations.get(filter_option, filter_option)
        output_file_name = f"locations_{admin_level}{filter_abbr}.csv" 
        
    output_file_path = os.path.join(output_dir, output_file_name)    
    
    # Ensure the output directory exists by checking first, for older Python versions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Read and initial data preparation
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Runtime Error: File not found at {input_file_path}")
        return

    # 3. Data Transformation
    # Select relevant columns
    df = df[["event_date", "country", "admin1", "admin2", "admin3", "location", "latitude", "longitude", "fatalities"]]
    
    # Validate the provided admin_level
    valid_admin_levels = ['admin1', 'admin2', 'admin3', 'location']
    if admin_level not in valid_admin_levels:
        print(f"Error: Invalid admin_level '{admin_level}'. Must be one of {valid_admin_levels}.")
        return
    
    # Convert 'event_date' to a datetime object and calculate conflict date
    try:
        start_date = pd.to_datetime(start_date, format="%d-%m-%Y")
        df['event_date'] = pd.to_datetime(df['event_date'], format="%Y-%m-%d")
        df['conflict_date'] = (df['event_date'] - start_date).dt.days
    except ValueError as e:
        print(f"Date format error: {e}. Please ensure start date is 'dd-mm-yyyy'.")
        return
    
    # Remove entries with zero fatalities
    df = df[df['fatalities'] > 0].copy()

    # 4. Filter logic based on user choice
    if filter_option == 'earliest':
        # Sort by conflict_date and keep the first occurrence per location
        df.sort_values(by='conflict_date', ascending=True, inplace=True)
        df.drop_duplicates(subset=[admin_level], keep='first', inplace=True)
    elif filter_option == 'fatalities':
        # Sort by fatalities and keep the highest fatality entry per location
        df.sort_values(by='fatalities', ascending=True, inplace=True)
        df.drop_duplicates(subset=[admin_level], keep='first', inplace=True)
    else:
        print("Invalid filter_option value. Must be 'earliest' or 'fatalities'.")
        return

    # 5. Finalize output DataFrame
    output_df = df[['location', 'admin1', 'country', 'latitude', 'longitude', 'conflict_date']].copy()
    output_df.rename(columns={'location': '#name', 'admin1': 'region'}, inplace=True)
    
    # Replace spaces in '#name' column with underscores
    output_df['#name'] = output_df['#name'].astype(str).str.replace(' ', '_')
    
    # Add new columns as per the required output format
    output_df['location_type'] = "conflict_zone"
    output_df['population'] = "0"
    
    # Reorder columns
    output_df = output_df[['#name', 'region', 'country', 'latitude', 'longitude', 'location_type', 'conflict_date', 'population']]
    
    # Sort the final output DataFrame by conflict_date in ascending order
    output_df.sort_values(by='conflict_date', ascending=True, inplace=True)

    # 6. Export to CSV
    output_df.to_csv(output_file_path, index=False)
    print(f"Data successfully saved to {output_file_path}")
    print(output_df.reset_index(drop=True).head().to_string())
    print(f"\nThe file contains {len(output_df)} locations.")

if __name__ == '__main__':

    fabflee = sys.argv[1]
    country = sys.argv[2]
    start_date = sys.argv[3]
    filter_option = sys.argv[4]
    admin_level = sys.argv[5]

    acled2locations(fabflee, country, start_date, filter_option, admin_level)

