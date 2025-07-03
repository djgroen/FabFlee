try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

from plugins.FabFlee.FabFlee import *

import plotly.express as px
import flee.InputGeography
from flee.SimulationSettings import SimulationSettings
import numpy as np
import sys

import os
import folium
import pandas as pd
from IPython.display import display, HTML
from folium.plugins import MarkerCluster
import plotly.io as pio
pio.renderers.default = "browser"


@task
def plot_flee_agents_location(results_dir, day, age=None, gender=None, proc="0"):
    """
    Plots the movement of refugees by chosen time and demographic prompts using agents.out.0
    """
    file_path = "{}/{}/agents.out.{}".format(env.local_results, results_dir, proc)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            print(f"The {file_path} file is empty.")
            exit()  # Exit the code
    else:
        print(f"The {file_path} file does not exist.")
        exit()  # Exit the code
    
    # Initialize an empty list to store valid lines
    valid_lines = []

    # Read the CSV file line by line and filter valid lines
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split(',')
            if len(columns) <= 14:  # Adjust the column count as needed 
                valid_lines.append(line)
            else:
                print(f"Skipped line with {len(columns)} columns: {line}")

    print("Number of valid lines:", len(valid_lines))
                
    # Read the valid lines into a DataFrame
    from io import StringIO

    # Specify the relevant column names
    default_columns = ['#time', 'rank-agentid', 'current_location', 'gps_x', 'gps_y']  
    header = valid_lines[0].split(',')

    # Create a new df to store the columns
    relevant_columns = default_columns.copy()

    if 'age' in header:
        relevant_columns.append('age')
    else:
        print("Age is not present in the data")

    if 'gender' in header:
        relevant_columns.append('gender')
    else:
        print("Gender is not present in the data")

    print("Relevant columns before filtering:", relevant_columns)

    # Filter out columns that are not present in the header
    relevant_columns = [col for col in relevant_columns if col in header]

    print("Relevant columns after filtering:", relevant_columns)

    # Read in the csv relevant columns
    df = pd.read_csv(StringIO('\n'.join(valid_lines)), usecols=relevant_columns)

    # Rename the '#time' column to 'time'
    df.rename(columns={'#time': 'time'}, inplace=True)

    # Filter the data based on day, age, and gender
    print("Filtering criteria:")
    print("Day:", day)
    print("Age:", age)
    print("Gender:", gender)

    if 'age' in df.columns and 'gender' in df.columns:
        if age is not None and gender is not None:
            filtered_df = df[(df['time'] == int(day)) & (df['age'] == int(age)) & (df['gender'] == gender)]
        elif age is not None:
            filtered_df = df[(df['time'] == int(day)) & (df['age'] == int(age))]
        elif gender is not None:
            filtered_df = df[(df['time'] == int(day)) & (df['gender'] == gender)]
        else:
            filtered_df = df[df['time'] == int(day)]
    elif 'age' in df.columns:
        if age is not None:
            filtered_df = df[(df['time'] == int(day)) & (df['age'] == int(age))]
        else:
            filtered_df = df[df['time'] == int(day)]
    elif 'gender' in df.columns:
        if gender is not None:
            filtered_df = df[(df['time'] == int(day)) & (df['gender'] == gender)]
        else:
            filtered_df = df[df['time'] == int(day)]
    else:
        filtered_df = df[df['time'] == int(day)]

    # Check if any rows are left after filtering
    if filtered_df.empty:
        print("No matching agents found.")
        return None

    # Calculate the total number of agents per location
    location_counts = filtered_df['current_location'].value_counts()
    
    print(location_counts)

    # Create a Folium map centered at the first agent's coordinates
    m = folium.Map(location=[filtered_df.iloc[0]['gps_x'], filtered_df.iloc[0]['gps_y']], zoom_start=5)
    
    # Add circle markers for each location with radius proportional to the agent count
    for location, count in location_counts.items():
        location_data = filtered_df[filtered_df['current_location'] == location].iloc[0]
        folium.CircleMarker(
            location=(location_data['gps_x'], location_data['gps_y']),
            radius=count * 0.02,  # Adjust the scale factor as needed
            popup=f"{count} agents in {location}",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)

    output_dir = os.path.dirname(file_path)
    output_file_name = f'plot_flee_agents_location_{results_dir}.html'
    output_file_path = os.path.join(output_dir, output_file_name)
    m.save(output_file_path)

    print(f"Map has been saved in {output_dir} as {output_file_name}") 



@task
def plot_flee_links(config):
    """
    Plots the location graph in a configuration directory.
    """
    with_config(config)

    #print(env)

    floc = "{}/input_csv/locations.csv".format(env.job_config_path_local)
    flink = "{}/input_csv/routes.csv".format(env.job_config_path_local)
    fssyml = "{}/input_csv/simsetting.yml".format(env.job_config_path_local)

    lats = []
    lons = []
    names = []

    ig = flee.InputGeography.InputGeography()

    ig.ReadLocationsFromCSV(floc)
    ig.ReadLinksFromCSV(flink)

    ll = ig.MakeLocationList()
    cl = ig.MakeLocationColorsList()
    print(ll)

    for l in ig.links:
        lats = np.append(lats, [ ll[l[0]][0], ll[l[1]][0], None])
        lons = np.append(lons, [ ll[l[0]][1], ll[l[1]][1], None])
        names = np.append(names, ["","",None])
        
    fig = px.line_geo(lat=lats,lon=lons)

    loclons = []
    loclats = []
    locnames = []

    for k in ll:
        locnames = np.append(locnames, [k])
        loclats = np.append(loclats,[ll[k][0]])
        loclons = np.append(loclons,[ll[k][1]])

    fig.add_scattergeo(
            name = "(location)",
            lon = loclons,
            lat = loclats,
            hovertext = locnames,
            marker={"color":cl}
        )

    fig.update_traces(marker_size=12, selector=dict(type='scattergeo'))

    fig.update_geos(fitbounds="locations")
    fig.update_geos(
        visible=False, resolution=50,
        showcountries=True, countrycolor="RebeccaPurple"
    )
    fig.update_layout(
        height=800,
        title="Location graph",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        font=dict(
            family="Courier New, monospace",
            size=32,
            color="Black"
            )
        )

    fig.show()


@task
def plot_flee_agents(results_dir, proc="0", agentid="0"):
    """
    Plots the trajectory of an individual agent in Flee
    """

    agentcsv = "{}/{}/agents.out.{}".format(env.local_results, results_dir, proc)
    lats = []
    lons = []
    names = []
    colors = []

    count = 0

    with open(agentcsv, newline='') as csvfile:
        header = []
        agentreader = csv.reader(csvfile, delimiter=',')
        for row in agentreader:
            if len(row) < 2:
                pass
            elif count == 0:
                header = row
                header_not_read = False
                print(header)
                count += 1
            else:
                if(row[1]) == "{}-{}".format(proc, agentid):
                    if count % 10 == 0:
                        lats = np.append(lats, [float(row[4]), None, float(row[4])])
                        lons = np.append(lons, [float(row[5]), None, float(row[5])])
                        names = np.append(names, [row[0], None, row[0]])
                    else:
                        lats = np.append(lats, [float(row[4])])
                        lons = np.append(lons, [float(row[5])])
                        names = np.append(names, [row[0]])
                    count += 1

    print(colors)
    fig = px.scatter_geo(lat=lats,lon=lons, hover_name=names)

    interval=30
    for i in range(0,int((len(lats)+interval-1)/interval)):
        start = i*interval
        end = (i+1)*interval
        fig.add_scattergeo(
            name = '{}-{}'.format(start, end),
            lon = lons[start:end],
            lat = lats[start:end],
            hoverinfo = 'text',
            text = names[start:end],
            mode = 'lines',
            line = dict(width = 4)
        )

    fig.update_geos(fitbounds="locations")
    fig.update_geos(
        visible=False, resolution=50,
        showcountries=True, countrycolor="RebeccaPurple"
    )
    fig.update_layout(height=800) #, margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(
        title="Agent movement graph",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        legend_title="Agent",
        font=dict(
            family="Courier New, monospace",
            size=32,
            color="Black"
            )
        )

    fig.show()




if __name__ == "__main__":
    pass

