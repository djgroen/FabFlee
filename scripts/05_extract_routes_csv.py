"""
Input: locations.csv
Output: routes.csv

Description:
This script processes geographical location data to generate a set of routes between locations. It reads location data from the 'locations.csv' file, which includes location names and their corresponding latitude and longitude coordinates. For each location, the script finds its two nearest neighbors based on the Euclidean distance and creates routes to these neighbors. The script then generates a 'routes.csv' file, which contains these routes, including the names of the start and end locations and the calculated distances between them.

The script uses a simple Euclidean distance calculation for determining the proximity between locations. It iterates over each location, identifies its two closest neighboring locations, and then records these pairs as individual routes. This approach ensures each location is considered as a starting point for two routes, potentially creating a more interconnected network of routes.

Usage:
1. Prepare location data and save it as a CSV file ('locations.csv').
2. Ensure the 'locations.csv' file includes columns for 'name', 'latitude', and 'longitude'.
3. Run the script without any additional parameters.

Command:
Run the script using the following command:
"python 05_routes_csv.py <country>"

Example Usage:
"python 05_routes_csv.py nigeria2016"
"""

import os 
import sys 
import csv
import requests 
import pandas as pd
import numpy as np

def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 1000

def find_nearest_neighbor(current_index, visited, locations_df):
    nearest_distance = float('inf')
    nearest_index = None
    for i in range(len(locations_df)):
        if not visited[i] and i != current_index:
            distance = calculate_distance(locations_df.iloc[current_index]['latitude'], locations_df.iloc[current_index]['longitude'],
                                          locations_df.iloc[i]['latitude'], locations_df.iloc[i]['longitude'])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = i
    return nearest_index, nearest_distance

def extract_routes_csv(country):

    # Get the current directory
    current_dir = os.getcwd()

    # Get the locations file
    locations_file = os.path.join(current_dir, country, "locations.csv")

    # Load the locations data from locations.csv into a DataFrame
    locations_df = pd.read_csv(locations_file)

    # Nearest Neighbor with Intermediate Stops
    visited = [False] * len(locations_df)
    route = [0]  # Start from the first location (index 0)
    visited[0] = True

    routes = []  # To store the routes and distances

    while not all(visited):
        current_index = route[-1]
        next_index, direct_distance = find_nearest_neighbor(current_index, visited, locations_df)

        # Check for possible intermediate stop
        for i in range(len(locations_df)):
            if not visited[i] and i != current_index and i != next_index:
                intermediate_distance = calculate_distance(locations_df.iloc[current_index]['latitude'], locations_df.iloc[current_index]['longitude'],
                                                           locations_df.iloc[i]['latitude'], locations_df.iloc[i]['longitude']) + \
                                        calculate_distance(locations_df.iloc[i]['latitude'], locations_df.iloc[i]['longitude'],
                                                           locations_df.iloc[next_index]['latitude'], locations_df.iloc[next_index]['longitude'])
                # If the route via the intermediate location is shorter, choose it
                if intermediate_distance < direct_distance:
                    next_index = i
                    direct_distance = intermediate_distance
                    break

        route.append(next_index)
        visited[next_index] = True
        routes.append([locations_df.iloc[current_index]['name'], locations_df.iloc[next_index]['name'], round(direct_distance, 2), 0])

    # Save the routes to a CSV file
    with open(f'{country}/routes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name1', 'name2', 'distance', 'force_redirection'])
        for route in routes:
            writer.writerow(route)

    print(f'{country}/routes.csv created. Please inspect the file for unwanted anomalies!')

# Specify desired country
country = sys.argv[1]

extract_routes_csv(country)
