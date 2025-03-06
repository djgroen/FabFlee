try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

import glob
import csv
import os
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree, move
# Add local script, blackbox and template path.
add_local_paths("FabFlee")

@task
# Syntax: fabsim localhost add_camp:<config_name>,<camp_name>,<region>,<country>(,<lat>,<lon>)
def add_camp(config, camp_name, region=" ", country=" ", lat=0.0, lon=0.0):
    """ 
    Add an additional new camp to locations.csv. 
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost add_camp:%s\n" % camp_name)

    # 1. Add (or make existing forwarding hub) a new camp to locations.csv
    # If new camp, add country,lat,lon,location_type(camp)
    # If existing camp, change location_type to camp
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/locations.csv" % (get_plugin_path("FabFlee"), config), "r"))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != camp_name:
            continue
        print("Warning: camp %s is already present in locations.csv."
              % (camp_name))
        return

    # 2. Append one line to lines, containing the details of the new camp.
    add_camp = [camp_name, region, country, lat, lon, "camp"]
    with open("%s/config_files/%s/input_csv/locations.csv"
              % (get_plugin_path("FabFlee"), config), "a") as new_csv:
        writer = csv.writer(new_csv)
        writer.writerow(add_camp)
    print(add_camp)

@task
# Syntax: fabsim localhost delete_location:<location_name>
def delete_location(config, location_name):
    """ 
    Deletes not-required camp (or location) from locations.csv. 
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost delete_location:%s\n" % location_name)

    # 1. Delete camp from locations.csv containing the details of the camp.
    # 2. Write the updated CSV file.
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/locations.csv"
                        % (get_plugin_path("FabFlee"), config), "r"))
    lines = [l for l in r]

    writer = csv.writer(open("%s/config_files/%s/input_csv/locations.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))

    for i in range(0, len(lines)):
        if lines[i][0].strip() != location_name:
            writer.writerow(lines[i])
            continue

            print(lines[i])

    # 3. Check whether wanted to delete camp is present in locations.csv
    for i in range(1, len(lines)):
        if lines[i][0] == location_name:
            continue
        print("Warning: camp %s is deleted from locations.csv."
              % (location_name))
        return

@task
# Syntax: fabsim localhost change_distance:name1,name2,distance
def change_distance(config, source, destination, distance):
    """ 
    Change distance between two locations in routes.csv. 
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost change_distance:%s,%s,%s\n"
                     % (source, destination, distance))

    # 1. Read routes.csv and for each location in the dict, find in the csv,
    # and change distance between two locations.
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/routes.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != source:
            continue
        if lines[i][1].strip() != destination:
            continue
        lines[i][2] = distance
        print(lines[i])

    # 2. Write the updated closures.csv.
    writer = csv.writer(open("%s/config_files/%s/input_csv/routes.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost close_camp:camp_name,country(,closure_start,closure_end)
def close_camp(config, camp_name, country, closure_start=0, closure_end=-1):
    """ 
    Close camp located within neighbouring country. 
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost close_camp:%s,%s\n" % (camp_name, country))

    # 1. Change closure_start and closure_end or add a new
    # camp closure to closures.csv.
    # Format: closure type <location>,name1,name2,closure_start,closure_end
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/closures.csv"
                        % (get_plugin_path("FabFlee"), config)))  # Here your csv file
    lines = [l for l in r]
    camp_found = False

    for i in range(1, len(lines)):
        if lines[i][0].strip() != "location":
            continue
        if lines[i][1].strip() != camp_name:
            continue
        if lines[i][2].strip() != country:
            continue
        lines[i][3] = closure_start
        lines[i][4] = closure_end
        camp_found = True
        print(lines[i])

    if not camp_found:
        lines.append(["location", camp_name, country,
                      closure_start, closure_end])
    # print(lines)

    # 2. Write the updated closures.csv in the %s directory.
    writer = csv.writer(open("%s/config_files/%s/input_csv/closures.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost close_border:country1,country2(,closure_start,closure_end)
def close_border(config, country1, country2, closure_start=0, closure_end=-1):
    """
    Close border between conflict country and camps located
    within specific neighbouring country.
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost close_border:%s,%s\n"
                     % (country1, country2))

    # 1. Change closure_start and closure_end or add a new camp
    # closure to closures.csv.
    # Format: closure type <country>,name1,name2,closure_start,closure_end
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/closures.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]
    border_found = False

    for i in range(1, len(lines)):
        if lines[i][0].strip() != "country":
            continue
        if lines[i][1].strip() != country1:
            continue
        if lines[i][2].strip() != country2:
            continue
        lines[i][3] = closure_start
        lines[i][4] = closure_end
        border_found = True
        print(lines[i])

    if not border_found:
        lines.append(["country", country1, country2,
                      closure_start, closure_end])

    """
    local(template("cp %s/config_files/%s/input_csv/*.csv \
        %s/config_files/%s/")
          % (get_plugin_path("FabFlee"), conflict_name,
             get_plugin_path("FabFlee")))
    print(lines)
    """

    # 2. Write the updated closures.csv.
    writer = csv.writer(open("%s/config_files/%s/input_csv/closures.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost change_capacities:camp_name=capacity(,camp_name2=capacity2)
def change_capacities(config, **capacities):
    """
    Change the capacity of a set of camps in the active conflict directory.
    """
    # Note: **capacities will be a Python dict object.

    capacities_string = ""
    for c in capacities.keys():
        capacities_string += "%s=%s" % (c, capacities[c])
    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost change_capacities:%s\n"
                     % capacities_string)

    # 1. Read in locations.csv
    # 2. for each location in the dict, find it in the csv, and modify the
    # population value accordingly.
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/locations.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]

    for camp_name in capacities.keys():
        for i in range(1, len(lines)):
            if lines[i][5].strip() != "camp":
                continue
            if lines[i][0].strip() != camp_name:
                continue

            lines[i][7] = capacities[camp_name]

            print(lines[i])

    # 3. Write the updated CSV file.
    writer = csv.writer(open("%s/config_files/%s/input_csv/locations.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost redirect:location_name1,location_name2
def redirect(config, source, destination):
    """
    Redirect from town or (small/other)camp to (main)camp.
    """

    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost redirect:%s,%s\n" % (source, destination))

    # 1. Read locations.csv and for each location in the dict, find in the csv,
    # and redirect refugees from location in neighbouring country to camp.
    # 2. Change location_type of source location to forwarding_hub.
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/locations.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != source:
            continue
        lines[i][5] = "forwarding_hub"

        print(lines[i])

    # 3. Write the updated CSV file.
    writer = csv.writer(open("%s/config_files/%s/input_csv/locations.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

    # 4. Find the route from source to destination in routes.csv, and enable
    # forced_redirection.
    r = csv.reader(open("%s/config_files/%s/input_csv/routes.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != source:
            continue
        if lines[i][1].strip() != destination:
            continue
        lines[i][3] = "2"
        print(lines[i])

    for i in range(1, len(lines)):
        if lines[i][0].strip() != destination:
            continue
        if lines[i][1].strip() != source:
            continue
        lines[i][3] = "1"
        print(lines[i])

    # 5. Write the updated CSV file.
    writer = csv.writer(open("%s/config_files/%s/input_csv/routes.csv"
                             % (get_plugin_path("FabFlee"), config), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost add_new_link:<name1>,<name2>,<distance>
def add_new_link(config, name1, name2, distance):
    """  
    Add a new link between locations to routes.csv. 
    """
    with open("%s/config_files/%s/commands.log.txt"
              % (get_plugin_path("FabFlee"), config), "a") as myfile:
        myfile.write("fabsim localhost add_new_link:%s,%s,%s\n"
                     % (name1, name2, distance))

    # 1. Read routes.csv and for each location in the dict, find in the csv,
    # and change distance between two locations.
    import csv
    r = csv.reader(open("%s/config_files/%s/input_csv/routes.csv"
                        % (get_plugin_path("FabFlee"), config)))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != name1:
            continue
        if lines[i][1].strip() != name2:
            continue
        lines[i][2] = distance
        print(lines[i])

    # 2. Append one line to lines, containing the details of links.
    add_new_link = [name1, name2, distance]
    with open("%s/config_files/%s/input_csv/routes.csv"
              % (get_plugin_path("FabFlee"), config), "a") as new_csv:
        writer = csv.writer(new_csv)
        writer.writerow(add_new_link)
    print(add_new_link)

@task
# Syntax: fabsim localhost find_capacity:<csv_name>
def find_capacity(config, csv_name):
    """
    Find the highest refugee number within csv file of source data
    for neighbouring camps.
    """

    import csv
    csv_file = open("%s/config_files/%s/source_data/%s"
                    % (get_plugin_path("FabFlee"), config, csv_name)).readlines()
    print(max(((i, int(l.split(',')[1])) for i, l in enumerate(
        csv_file)), key=lambda t: t[1])[1])
