try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

from plugins.FabFlee.FabFlee import *

import plotly.express as px
import flee.InputGeography
import numpy as np
import sys


@task
def plot_flee_links(config):
    """
    Plots the location graph in a configuration directory.
    """
    with_config(config)

    #print(env)

    floc = "{}/input_csv/locations.csv".format(env.job_config_path_local)
    flink = "{}/input_csv/routes.csv".format(env.job_config_path_local)

    lats = []
    lons = []
    names = []

    ig = flee.InputGeography.InputGeography()

    ig.ReadLocationsFromCSV(floc)
    ig.ReadLinksFromCSV(flink)

    ll = ig.MakeLocationList()
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
