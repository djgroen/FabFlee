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
    floc = "/home/csstddg/FabSim3/results/mali_localhost_4/input_csv/locations.csv"
    flink = "/home/csstddg/FabSim3/results/mali_localhost_4/input_csv/routes.csv"

    lats = []
    lons = []
    names = []

    ig = flee.InputGeography.InputGeography()

    ig.ReadLocationsFromCSV(floc)
    ig.ReadLinksFromCSV(flink)

    ll = ig.MakeLocationList()

    for l in ig.links:
        lats = np.append(lats, [ ll[l[0]][0], ll[l[1]][0], None])
        lons = np.append(lons, [ ll[l[0]][1], ll[l[1]][1], None])
        names = np.append(names, ["","",None])
        
    fig = px.line_geo(lat=lats,lon=lons)
    fig.update_geos(fitbounds="locations")
    fig.update_geos(
        visible=False, resolution=50,
        showcountries=True, countrycolor="RebeccaPurple"
    )
    fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

if __name__ == "__main__":
    pass
