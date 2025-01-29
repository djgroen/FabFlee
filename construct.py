try:
    from fabsim.base.fab import *
except ImportError:
    from base.fab import *

# Import V&V primitives.
try:
    import fabsim.VVP.vvp as vvp
except ImportError:
    import VVP.vvp as vvp

try:
    from FabFlee import *
except ImportError:
    from FabFlee import *

import glob
import csv
import os
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree, move
# Add local script, blackbox and template path.
add_local_paths("FabFlee")

@task
# Syntax: fabsim localhost clear_active_conflict
def clear_active_conflict():
    """ 
    Delete all content in the active conflict directory. 
    """

    local(template("rm -rf %s/conflict_data/active_conflict/"
                   % (get_plugin_path("FabFlee"))))

@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost new_conflict:<config_name>
def new_conflict(config, **args):
    """
    Create a new conflict configuration directory for Flee simulations.
    """

    local(template("mkdir -p %s/config_files/%s"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("mkdir -p %s/config_files/%s/input_csv"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("mkdir -p %s/config_files/%s/source_data"
                   % (get_plugin_path("FabFlee"), config)))

    local(template("cp %s/flee/config_template/run.py \
        %s/config_files/%s")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/run_par.py \
        %s/config_files/%s")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/simsetting.csv \
        %s/config_files/%s")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/sim_period.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/closures.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

    local(template("cp %s/flee/config_template/input_csv/"
                   "registration_corrections.csv "
                   "%s/config_files/%s/input_csv")
          % (env.flee_location, get_plugin_path("FabFlee"), config))

@task
# Syntax: fabsim localhost load_conflict:<conflict_name>
def load_conflict(conflict_name):
    """
    Load source data and flee csv files for a specific conflict from
    conflict data to active conflict directory.
    """
    # copies *.csv files from $FabFlee/conflict_data/<conflict_name> to
    # $FabFlee/conflict_data/active_conflict.

    # 1. Load locations.csv, routes.csv and closures.csv files which
    # correspond to a specific conflict.
    # These CSV will be store in $FabFlee/conflict_data. Each conflict will be
    # stored in a separate folder.

    # 2. Move these CSVs to an "active_conflict" directory.
    # This is located in $FABSIM/conflict_data/active_conflict.
    local(template("mkdir -p %s/conflict_data/active_conflict"
                   % (get_plugin_path("FabFlee"))))

    local(template("cp %s/conflict_data/%s/*.csv \
        %s/conflict_data/active_conflict/")
          % (get_plugin_path("FabFlee"), conflict_name,
             get_plugin_path("FabFlee")))

    local(template("mkdir -p %s/conflict_data/active_conflict/source_data"
                   % (get_plugin_path("FabFlee"))))

    local(template("cp %s/conflict_data/%s/source_data/*.csv \
        %s/conflict_data/active_conflict/source_data/")
          % (get_plugin_path("FabFlee"), conflict_name,
             get_plugin_path("FabFlee")))

    local(template("cp %s/config_files/run.py \
        %s/conflict_data/active_conflict")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee")))

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost load_conflict:%s\n" % conflict_name)

@task
# Syntax: fabsim localhost instantiate:conflict_name
def instantiate(conflict_name):
    """
    Copy modified active conflict directory to config_files
    (i.e. flee_conflict_name) to run instance with Flee.
    """

    # 1. Copy modified active_conflict directory to instantiate runs with
    # specific conflict name
    local(template("mkdir -p %s/config_files/%s"
                   % (get_plugin_path("FabFlee"), conflict_name)))

    local(template("mkdir -p %s/config_files/%s/input_csv"
                   % (get_plugin_path("FabFlee"), conflict_name)))

    local(template("cp %s/conflict_data/active_conflict/*.csv \
        %s/config_files/%s/input_csv")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))

    local(template("cp %s/conflict_data/active_conflict/commands.log.txt \
        %s/config_files/%s/")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))

    local(template("mkdir -p %s/config_files/%s/source_data"
                   % (get_plugin_path("FabFlee"), conflict_name)))

    local(template("cp %s/conflict_data/active_conflict/source_data/*.csv \
        %s/config_files/%s/source_data")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))

    local(template("cp %s/conflict_data/active_conflict/run.py \
        %s/config_files/%s/run.py")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))

    local(template("cp %s/config_files/run_food.py \
        %s/config_files/%s/run_food.py")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))
    # line added to copy run_food.py as well (otherwise executing
    # food_flee doesn't work...)

    # line added to copy simsetting.csv and make sure that
    # flee.SimulationSettings....ReadfromCSV works.
    local(template("cp %s/config_files/simsetting.csv \
        %s/config_files/%s/simsetting.csv")
          % (get_plugin_path("FabFlee"), get_plugin_path("FabFlee"),
             conflict_name))

@task
@load_plugin_env_vars("FabFlee")
# Syntax: fabsim localhost add_population:<config_name>
def add_population(config, PL="100", CL="100", **args):
    # update_environment(args, {"simulation_period": simulation_period})
    """
    Add population and city limits to the Flee simulation configuration.
    """
    with_config(config)
    if env.machine_name != 'localhost':
        print("Error : This task should only executed on your localhost")
        print("Please re-run is again with :")
        print("\t fab localhost add_population:%s" % (config))
        exit()
    env.cityGraph_POPULATION_LIMIT = PL
    env.cityGraph_CITIES_LIMIT = CL
    local("python %s --cityGraph_location %s --API_KEY %s "
          "--POPULATION_LIMIT %s --CITIES_LIMIT %s "
          "--config_location %s --config_name %s"
          % (os.path.join(env.localplugins["FabFlee"],
                          "scripts",
                          "population2locations.py"),
             env.cityGraph_location,
             env.cityGraph_API_KEY,
             env.cityGraph_POPULATION_LIMIT,
             env.cityGraph_CITIES_LIMIT,
             env.job_config_path_local,
             config
             )
          )
    
@task
# Syntax: fabsim localhost add_camp:camp_name,region,country(,lat,lon)
def add_camp(camp_name, region=" ", country=" ", lat=0.0, lon=0.0):
    """ 
    Add an additional new camp to locations.csv. 
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost add_camp:%s\n" % camp_name)

    # 1. Add (or make existing forwarding hub) a new camp to locations.csv
    # If new camp, add country,lat,lon,location_type(camp)
    # If existing camp, change location_type to camp
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv"
                        % (get_plugin_path("FabFlee")), "r"))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != camp_name:
            continue
        print("Warning: camp %s is already present in locations.csv."
              % (camp_name))
        return

    # 2. Append one line to lines, containing the details of the new camp.
    add_camp = [camp_name, region, country, lat, lon, "camp"]
    with open("%s/conflict_data/active_conflict/locations.csv"
              % (get_plugin_path("FabFlee")), "a") as new_csv:
        writer = csv.writer(new_csv)
        writer.writerow(add_camp)
    print(add_camp)

@task
# Syntax: fabsim localhost delete_location:<location_name>
def delete_location(location_name):
    """ 
    Deletes not-required camp (or location) from locations.csv. 
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost delete_location:%s\n" % location_name)

    # 1. Delete camp from locations.csv containing the details of the camp.
    # 2. Write the updated CSV file.
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv"
                        % (get_plugin_path("FabFlee")), "r"))
    lines = [l for l in r]

    writer = csv.writer(open("%s/conflict_data/active_conflict/locations.csv"
                             % (get_plugin_path("FabFlee")), "w"))

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
def change_distance(source, destination, distance):
    """ 
    Change distance between two locations in routes.csv. 
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost change_distance:%s,%s,%s\n"
                     % (source, destination, distance))

    # 1. Read routes.csv and for each location in the dict, find in the csv,
    # and change distance between two locations.
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/routes.csv"
                        % (get_plugin_path("FabFlee"))))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != source:
            continue
        if lines[i][1].strip() != destination:
            continue
        lines[i][2] = distance
        print(lines[i])

    # 2. Write the updated closures.csv in the active_conflict directory.
    writer = csv.writer(open("%s/conflict_data/active_conflict/routes.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost close_camp:camp_name,country(,closure_start,closure_end)
def close_camp(camp_name, country, closure_start=0, closure_end=-1):
    """ 
    Close camp located within neighbouring country. 
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost close_camp:%s,%s\n" % (camp_name, country))

    # 1. Change closure_start and closure_end or add a new
    # camp closure to closures.csv.
    # Format: closure type <location>,name1,name2,closure_start,closure_end
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/closures.csv"
                        % (get_plugin_path("FabFlee"))))  # Here your csv file
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

    # 2. Write the updated closures.csv in the active_conflict directory.
    writer = csv.writer(open("%s/conflict_data/active_conflict/closures.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost close_border:country1,country2(,closure_start,closure_end)
def close_border(country1, country2, closure_start=0, closure_end=-1):
    """
    Close border between conflict country and camps located
    within specific neighbouring country.
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost close_border:%s,%s\n"
                     % (country1, country2))

    # 1. Change closure_start and closure_end or add a new camp
    # closure to closures.csv.
    # Format: closure type <country>,name1,name2,closure_start,closure_end
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/closures.csv"
                        % (get_plugin_path("FabFlee"))))
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
    local(template("cp %s/conflict_data/%s/*.csv \
        %s/conflict_data/active_conflict/")
          % (get_plugin_path("FabFlee"), conflict_name,
             get_plugin_path("FabFlee")))
    print(lines)
    """

    # 2. Write the updated closures.csv in the active_conflict directory.
    writer = csv.writer(open("%s/conflict_data/active_conflict/closures.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost change_capacities:camp_name=capacity(,camp_name2=capacity2)
def change_capacities(**capacities):
    """
    Change the capacity of a set of camps in the active conflict directory.
    """
    # Note: **capacities will be a Python dict object.

    capacities_string = ""
    for c in capacities.keys():
        capacities_string += "%s=%s" % (c, capacities[c])
    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost change_capacities:%s\n"
                     % capacities_string)

    # 1. Read in locations.csv
    # 2. for each location in the dict, find it in the csv, and modify the
    # population value accordingly.
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv"
                        % (get_plugin_path("FabFlee"))))
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
    writer = csv.writer(open("%s/conflict_data/active_conflict/locations.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost redirect:location_name1,location_name2
def redirect(source, destination):
    """
    Redirect from town or (small/other)camp to (main)camp.
    """

    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fabsim localhost redirect:%s,%s\n" % (source, destination))

    # 1. Read locations.csv and for each location in the dict, find in the csv,
    # and redirect refugees from location in neighbouring country to camp.
    # 2. Change location_type of source location to forwarding_hub.
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/locations.csv"
                        % (get_plugin_path("FabFlee"))))
    lines = [l for l in r]

    for i in range(1, len(lines)):
        if lines[i][0].strip() != source:
            continue
        lines[i][5] = "forwarding_hub"

        print(lines[i])

    # 3. Write the updated CSV file.
    writer = csv.writer(open("%s/conflict_data/active_conflict/locations.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

    # 4. Find the route from source to destination in routes.csv, and enable
    # forced_redirection.
    r = csv.reader(open("%s/conflict_data/active_conflict/routes.csv"
                        % (get_plugin_path("FabFlee"))))
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
    writer = csv.writer(open("%s/conflict_data/active_conflict/routes.csv"
                             % (get_plugin_path("FabFlee")), "w"))
    writer.writerows(lines)

@task
# Syntax: fabsim localhost add_new_link:<name1>,<name2>,<distance>
def add_new_link(name1, name2, distance):
    """  
    Add a new link between locations to routes.csv. 
    """
    with open("%s/conflict_data/active_conflict/commands.log.txt"
              % (get_plugin_path("FabFlee")), "a") as myfile:
        myfile.write("fab localhost add_new_link:%s,%s,%s\n"
                     % (name1, name2, distance))

    # 1. Read routes.csv and for each location in the dict, find in the csv,
    # and change distance between two locations.
    import csv
    r = csv.reader(open("%s/conflict_data/active_conflict/routes.csv"
                        % (get_plugin_path("FabFlee"))))
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
    with open("%s/conflict_data/active_conflict/routes.csv"
              % (get_plugin_path("FabFlee")), "a") as new_csv:
        writer = csv.writer(new_csv)
        writer.writerow(add_new_link)
    print(add_new_link)

@task
# Syntax: fabsim localhost find_capacity:<csv_name>
def find_capacity(csv_name):
    """
    Find the highest refugee number within csv file of source data
    for neighbouring camps.
    """

    import csv
    csv_file = open("%s/conflict_data/active_conflict/source_data/%s"
                    % (get_plugin_path("FabFlee"), csv_name)).readlines()
    print(max(((i, int(l.split(',')[1])) for i, l in enumerate(
        csv_file)), key=lambda t: t[1])[1])