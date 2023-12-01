'''
Input: population.html
Output: population.csv

Description:
This script creates a directory named after the specified country (e.g., nigeria2016). 
It extracts city/town names and population data from an HTML file (population.html) and saves them in a CSV file (population.csv).

Usage:
1. Locate the desired country's population data on "https://www.citypopulation.de/" (e.g., https://www.citypopulation.de/en/nigeria/cities/).
2. Save the webpage containing the data as an HTML file (population.html).
3. Place the population.html file in the created directory (e.g., nigeria2016).

Command:
Run the script using the following command:
"python 01_extract_population_csv.py <country> <table_num> <column_num> <threshold>"

- <country>: Name of the country (e.g., nigeria2016).
- <table_num>: Table index (0, 1, etc.) specifying which table to extract data from in the HTML file.
- <column_num>: Column index (0, 1, etc.) indicating the population column to extract from the table.
- <threshold>: Population threshold for filtering rows in the selected table.

Example Usage:
"python 01_extract_population_csv.py nigeria2016 0 7 10000"
'''

import sys
import os
import pandas as pd
import codecs


def make_directory(country):
    # Check if the directory already exists
    if not os.path.exists(country):
        # Create the directory if it doesn't exist
        os.makedirs(country)
        return f"Directory '{country}' created."
    else:
        return f"Directory '{country}' already exists."


def extract_population_csv(country, table_num, column_num, threshold):
    # Create a file directory
    make_directory(country)

    # Specify HTML file name and extract its tables
    html_file = f"{country}/population.html"
    if os.path.isfile(html_file):
        tables = pd.read_html(html_file)
        # Check if the specified table number is within the valid range
        if 0 <= table_num < len(tables):
            pass
            # Continue with further processing of the tables
        else:
            print(f"Table number '{table_num}' is out of range. Valid table numbers are from 0 to {len(tables) - 1}.")
            sys.exit(1)
    else:
        print("The file '{}' is not found.".format(html_file))

    # Specify table with major cities' names and population
    table = tables[table_num]

    # Print the selected table
    print("Original Table:\n", table.to_string(index=False))

    # Specify the desired column number for the "Name" column
    name_column_num = 1  # You can change this to the desired column number for "Name"

    # Make sure the column indices are within the valid range
    if 0 <= name_column_num < len(table.columns) and 0 <= column_num < len(table.columns):
        # Select the "Name" column and the second column using the specified column numbers
        selected_columns = table.iloc[:, [name_column_num, column_num]]

        # Convert the "population" column to numeric, handling non-numeric values as NaN
        selected_columns.iloc[:, 1] = pd.to_numeric(selected_columns.iloc[:, 1], errors='coerce')

        # Drop rows with missing values in the selected columns
        selected_columns = selected_columns.dropna()

        # Convert the "population" column back to integers
        selected_columns.iloc[:, 1] = selected_columns.iloc[:, 1].astype(int)

        # Print the selected columns
        print("Extracted Table:\n", selected_columns.to_string(index=False))
    else:
        print(f"Column number '{column_num}' is out of range. Valid column numbers are from 0 to {len(table.columns) - 1}.")
        sys.exit(1)

    # Rename the selected columns for clarity
    selected_columns.columns = ['name', 'population']

    # Filter rows with the second column greater than threshold
    selected_columns = selected_columns[selected_columns['population'] > threshold]

    # Save the data to a CSV file with UTF-8 encoding
    selected_columns.to_csv(f"{country}/population.csv", index=False, encoding='utf-8')

    print(f"{country}/population.csv created. Please inspect the file for non-standard characters!")


country = sys.argv[1]
table_num = int(sys.argv[2])
column_num = int(sys.argv[3])
threshold = int(sys.argv[4])


extract_population_csv(country, table_num, column_num, threshold)
