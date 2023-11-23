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
"python extract_population.py <country> <table> <population_column> <threshold>"

- <country>: Name of the country (e.g., nigeria2016).
- <table>: Table index (0, 1, etc.) specifying which table to extract data from in the HTML file.
- <population_column>: Column index (0, 1, etc.) indicating the population column to extract from the table.
- <threshold>: Population threshold for filtering rows in the selected table.

Example Usage:
"python extract_population.py nigeria2016 0 7 10000"
'''

import sys
import os
import pandas as pd

# Define the country name
country = sys.argv[1]

# Check if the directory already exists
if not os.path.exists(country):
    # Create the directory if it doesn't exist
    os.makedirs(country)
    print(f"Directory '{country}' created.")
else:
    print(f"Directory '{country}' already exists.")

html_file = f"{country}/population.html"

if os.path.isfile(html_file):
    tables = pd.read_html(html_file)
    # Continue with further processing of the tables
else:
    print(f"The file '{html_file}' is not found.")

# Define the desired table from html file tables
table = tables[int(sys.argv[2])]

# Print the selected columns
print("\nOriginal Table:\n", table.to_string(index=False))

# Specify the desired column number for the "Name" column
name_column_num = 1  # You can change this to the desired column number for "Name"

# Specify the desired column number for the second column
second_column_num = int(sys.argv[3])

# Make sure the column indices are within the valid range
if 0 <= name_column_num < len(table.columns) and 0 <= second_column_num < len(table.columns):
    # Select the "Name" column and the second column using the specified column numbers
    selected_columns = table.iloc[:, [name_column_num, second_column_num]]

    # Convert the "population" column to numeric, handling non-numeric values as NaN
    selected_columns.iloc[:, 1] = pd.to_numeric(selected_columns.iloc[:, 1], errors='coerce')

    # Drop rows with missing values in the selected columns
    selected_columns = selected_columns.dropna()

    # Convert the "population" column back to integers
    selected_columns.iloc[:, 1] = selected_columns.iloc[:, 1].astype(int)

    # Print the selected columns
    print("\n\nExtraxted Table:\n", selected_columns.to_string(index=False))
else:
    print("\n\noops: please select valid column numbers for extraction!\n")
    sys.exit(1)

# Rename the selected columns for clarity
selected_columns.columns = ['name', 'population']

# Define population threshold
threshold = int(sys.argv[4])

# Filter rows with the second column greater than threshold
selected_columns = selected_columns[selected_columns['population'] > threshold]

# Save the data to a CSV file
selected_columns.to_csv(f"{country}/population.csv", index=False)

print(f"{country}/population.csv created. Please inspect the file for non-standard characters!")
