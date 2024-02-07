"""
Input: locations.csv & routes.csv
Output: An interactive map file (HTML format)

Description:
This script visualizes geographical routes and locations for a specified country. It primarily functions through the `extract_routes_info(country)` method, which processes location and route data for the given country. The script reads two CSV files: 'locations.csv' and 'routes.csv'. The 'locations.csv' file contains location data with fields such as 'name', 'latitude', and 'longitude'. The 'routes.csv' file contains routes data with fields including the starting location (name1), destination location (name2), and the distance between them.

The script uses the `folium` library to create an interactive map. It first plots markers for each location from the 'locations.csv' file. Then, it draws lines representing the routes from the 'routes.csv' file, showing the connections between different locations.

The resulting map offers a visual representation of how various locations are interconnected through different routes. This visualization can be particularly useful for understanding geographic proximity and route planning in the specified country.

Usage:
1. Ensure that 'locations.csv' and 'routes.csv' files are present in the directory named after the specified country.
2. Install the necessary Python libraries, including `folium` and `pandas`.
3. Run the script with the country name as a command line argument.

Command:
Run the script using the following command:
"python extract_routes_info.py <country>"

Example Usage:
"python 06_extract_routes_info.py nigeria2016"

This will generate an interactive map for the 'nigeria2016' dataset, visualizing the locations and routes based on the corresponding CSV files in the 'nigeria2016' directory.
"""

import os 
import sys  
import folium
import pandas as pd
import webbrowser

def extract_routes_info(country):

    # Get the current directory
    current_dir = os.getcwd()

    # Get the locations file
    locations_file = os.path.join(current_dir, country, "locations.csv")

    # Load the locations data from locations.csv into a DataFrame
    locations_df = pd.read_csv(locations_file)

    # Initialize a map and zoom_start to suit your dataset
    map_center = [locations_df['latitude'].mean(), locations_df['longitude'].mean()]
    mymap = folium.Map(location=map_center, zoom_start=6,tiles="cartodb positron") #default tiles="cartodb positron"
    
    #Add map styles as tile layers - select choice of tiles from control in top right corner
    folium.TileLayer('openstreetmap').add_to(mymap)
    folium.TileLayer('cartodb positron').add_to(mymap)
    folium.TileLayer('cartodb dark_matter').add_to(mymap)
    folium.LayerControl().add_to(mymap)

    # Add markers for each location
    for _, row in locations_df.iterrows():
        icon = None  # Initialize an empty icon variable

        if row['location_type'] == 'town':
            icon = folium.Icon(color='darkpurple', icon='home')
        elif row['location_type'] == 'camp':
            icon = folium.Icon(color='green', icon='tent', prefix='fa')
        elif row['location_type'] == 'flood_zone':
            icon = folium.Icon(color='darkblue', icon='water', prefix='fa')
        elif row['location_type'] == 'conflict_zone':
            icon = folium.Icon(color='red', icon='person-rifle', prefix='fa')

        # Use the selected icon or create a custom HTML icon only if no icon is set
        if not icon:
            icon = folium.DivIcon(html=f"""<div style="font-family: courier new; color: black">{row['name']}</div>""")

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['name'],
            icon=icon
        ).add_to(mymap)

    # Draw routes
    routes_df = pd.read_csv(f'{country}/routes.csv')
    for _, row in routes_df.iterrows():
        loc1 = locations_df[locations_df['name'] == row['name1']].iloc[0]
        loc2 = locations_df[locations_df['name'] == row['name2']].iloc[0]
        folium.PolyLine(
            locations=[[loc1['latitude'], loc1['longitude']], [loc2['latitude'], loc2['longitude']]],
            color='grey',
        ).add_to(mymap)

    # Save map to HTML file
    mymap.save(f'{country}/{country}_map.html')

    # Open the map in a web browser
    webbrowser.open(f"{country}/{country}_map.html")

# Specify desired country
country = sys.argv[1]

extract_routes_info(country)
