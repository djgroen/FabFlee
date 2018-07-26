# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is 151distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to fabFlee.

from base.fab import *

# Add local script, blackbox and template path.
add_local_paths("FabFlee")

@task
def flee(config,**args):
    """Submit a Flee job to the remote queue.
    The job results will be stored with a name pattern as defined in the environment,
    e.g. car-abcd1234-localhost-4
    config : config directory to use for the simulation script, e.g. config=car2014
    Keyword arguments:
            cores : number of compute cores to request
            wall_time : wall-time job limit
            memory : memory per node
    """
    update_environment({"input_directory":"%s/config_files/%s/input_csv" % (env.localroot, config),"validation_data_directory":"%s/config_files/%s/source_data" % (env.localroot, config)})
    #print_local_environment()
    update_environment(args)
    with_config(config)
    execute(put_configs,config)
    job(dict(script='flee', wall_time='0:15:0', memory='2G'),args)


@task
def flees(config,**args):
    # Save relevant arguments to a Python or numpy list.
    print(args)

    # Generate config directories, copying from the config provided, and adding a different generated test.csv in each directory.

    # Run the flee() a number of times.


@task
def plot_output(output_dir="", graphs_dir=""):      # Syntax: fab localhost plot_output:flee_conflict_name_localhost_16(,graphs_dir_name)
  """ Plot generated output results using plot-flee-output.py. """
  # python3 $flee_dir/plot-flee-output.py $fabsim_results/$output_dir
  local("mkdir -p %s/%s/%s" % (env.results_path, output_dir, graphs_dir))
  local("python3 %s/plot-flee-output.py %s/%s %s/%s/%s" % (env.flee_location, env.results_path, output_dir, env.results_path, output_dir, graphs_dir))

@task
def load_conflict(conflict_name):     # Syntax: fab localhost load_conflict:conflict_name
  """ Load source data and flee csv files for a specific conflict from conflict data to active conflict directory. """
  # copies *.csv files from $FABSIM/conflict_data/<conflict_name> to $FABSIM/conflict_data/active_conflict.

  # 1. Load locations.csv, routes.csv and closures.csv files which correspond to a specific conflict.
  # These CSV will be store in $FABSIM/conflict_data. Each conflict will be stored in a separate folder.

  # 2. Move these CSVs to an "active_conflict" directory.
  # This is located in $FABSIM/conflict_data/active_conflict.
  local(template("mkdir -p %s/plugins/FabFlee/conflict_data/active_conflict" % (env.localroot)))
  local(template("cp %s/plugins/FabFlee/conflict_data/%s/*.csv %s/plugins/FabFlee/conflict_data/active_conflict/") % (env.localroot, conflict_name, env.localroot))
  local(template("mkdir -p %s/plugins/FabFlee/conflict_data/active_conflict/source_data" % (env.localroot)))
  local(template("cp %s/plugins/FabFlee/conflict_data/%s/source_data/*.csv %s/plugins/FabFlee/conflict_data/active_conflict/source_data/") % (env.localroot, conflict_name, env.localroot))


  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost load_conflict:%s\n" % conflict_name)


@task
def change_capacities(**capacities):     # Syntax: fab localhost change_capacities:camp_name=capacity(,camp_name2=capacity2)
  """ Change the capacity of a set of camps in the active conflict directory. """
  # Note: **capacities will be a Python dict object.

  capacities_string = ""
  for c in capacities.keys():
    capacities_string += "%s=%s" % (c,capacities[c])
  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost change_capacities:%s\n" % capacities_string)

  # 1. Read in locations.csv
  # 2. for each location in the dict, find it in the csv, and modify the population value accordingly.
  import csv
  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot)))
  lines = [l for l in r]

  for camp_name in capacities.keys():
    for i in range(1,len(lines)):
      if lines[i][5].strip() != "camp":
        continue
      if lines[i][0].strip() != camp_name:
        continue

      lines[i][7] = capacities[camp_name]

      print(lines[i])

  # 3. Write the updated CSV file.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "w"))
  writer.writerows(lines)


@task
def add_camp(camp_name, region=" ", country=" ", lat=0.0, lon=0.0):    # Syntax: fab localhost add_camp:camp_name,region,country(,lat,lon)
  """ Add an additional new camp to locations.csv. """

  with open("%s/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost add_camp:%s\n" % camp_name)

  # 1. Add (or make existing forwarding hub) a new camp to locations.csv
  # If new camp, add country,lat,lon,location_type(camp)
  # If existing camp, change location_type to camp
  import csv
  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "r"))
  lines = [l for l in r]

  for i in range(1,len(lines)):
    if lines[i][0].strip() != camp_name:
      continue
    print("Warning: camp %s is already present in locations.csv." % (camp_name))
    return

  # 2. Append one line to lines, containing the details of the new camp.
  add_camp = [camp_name,region,country,lat,lon,"camp"]
  with open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "a") as new_csv:
    writer = csv.writer(new_csv)
    writer.writerow(add_camp)
  print(add_camp)


@task
def delete_location(location_name):     # Syntax: fab localhost delete_location:location_name
  """ Delete not required camp (or location) from locations.csv. """

  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost delete_location:%s\n" % location_name)

  # 1. Delete camp from locations.csv containing the details of the camp.
  # 2. Write the updated CSV file.
  import csv

  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "r"))
  lines = [l for l in r]

  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "w"))

  for i in range(0,len(lines)):
    if lines[i][0].strip() != location_name:
      writer.writerow(lines[i])
      continue

      print(lines[i])

  # 3. Check whether wanted to delete camp is present in locations.csv
  for i in range(1,len(lines)):
    if lines[i][0] == location_name:
      continue
    print("Warning: camp %s is deleted from locations.csv." % (location_name))
    return


@task
def close_camp(camp_name, country, closure_start=0, closure_end=-1):     # Syntax: fab localhost close_camp:camp_name,country(,closure_start,closure_end)
  """ Close camp located within neighbouring country. """

  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost close_camp:%s\n" % (camp_name,country))

  # 1. Change closure_start and closure_end or add a new camp closure to closures.csv.
  # Format: closure type <location>,name1,name2,closure_start,closure_end
  import csv
  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/closures.csv" % (env.localroot))) # Here your csv file
  lines = [l for l in r]
  camp_found = False

  for i in range(1,len(lines)):
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
    lines.append(["location",camp_name,country,closure_start,closure_end])
  #print(lines)

  # 2. Write the updated closures.csv in the active_conflict directory.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/closures.csv" % (env.localroot), "w"))
  writer.writerows(lines)


@task
def close_border(country1, country2, closure_start=0, closure_end=-1):     # Syntax: fab localhost close_border:country1,country2(,closure_start,closure_end)
  """ Close border between conflict country and camps located within specific neighbouring country. """

  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost close_border:%s,%s\n" % (country1,country2))

  # 1. Change closure_start and closure_end or add a new camp closure to closures.csv.
  # Format: closure type <country>,name1,name2,closure_start,closure_end
  import csv

  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/closures.csv" % (env.localroot)))
  lines = [l for l in r]
  border_found = False

  for i in range(1,len(lines)):
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
    lines.append(["country",country1,country2,closure_start,closure_end])

  #local(template("cp %s/plugins/FabFlee/conflict_data/%s/*.csv %s/plugins/FabFlee/conflict_data/active_conflict/") % (env.localroot, conflict_name, env.localroot))
  #print(lines)

  # 2. Write the updated closures.csv in the active_conflict directory.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/closures.csv" % (env.localroot), "w"))
  writer.writerows(lines)


@task
def redirect(source, destination):    # Syntax: fab localhost redirect:location_name1,location_name2
  """ Redirect from town or (small/other)camp to (main)camp """

  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost redirect:%s,%s\n" % (source,destination))


  # 1. Read locations.csv and for each location in the dict, find in the csv, and redirect refugees from location in neighbouring country to camp.
  # 2. Change location_type of source location to forwarding_hub.
  import csv

  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot)))
  lines = [l for l in r]

  for i in range(1,len(lines)):
    if lines[i][0].strip() != source:
      continue
    lines[i][5] = "forwarding_hub"

    print(lines[i])

  # 3. Write the updated CSV file.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/locations.csv" % (env.localroot), "w"))
  writer.writerows(lines)

  # 4. Find the route from source to destination in routes.csv, and enable forced_redirection.
  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/routes.csv" % (env.localroot)))
  lines = [l for l in r]

  for i in range(1,len(lines)):
    if lines[i][0].strip() != source:
      continue
    if lines[i][1].strip() != destination:
      continue
    lines[i][3] = "2"
    print(lines[i])

  for i in range(1,len(lines)):
    if lines[i][0].strip() != destination:
      continue
    if lines[i][1].strip() != source:
      continue
    lines[i][3] = "1"
    print(lines[i])

  # 5. Write the updated CSV file.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/routes.csv" % (env.localroot), "w"))
  writer.writerows(lines)


@task
def change_distance(source, destination, distance):         #Syntax: fab localhost change_distance:name1,name2,distance
  """ Change distance between two locations in routes.csv """

  with open("%s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt" % (env.localroot), "a") as myfile:
    myfile.write("fab localhost change_distance:%s,%s,%s\n" % (source, destination, distance))

  # 1. Read routes.csv and for each location in the dict, find in the csv, and change distance between two locations.
  import csv

  r = csv.reader(open("%s/plugins/FabFlee/conflict_data/active_conflict/routes.csv" % (env.localroot)))
  lines = [l for l in r]

  for i in range(1,len(lines)):
    if lines[i][0].strip() != source:
      continue
    if lines[i][1].strip() != destination:
      continue
    lines[i][2] = distance
    print(lines[i])

  # 2. Write the updated closures.csv in the active_conflict directory.
  writer = csv.writer(open("%s/plugins/FabFlee/conflict_data/active_conflict/routes.csv" % (env.localroot), "w"))
  writer.writerows(lines)


@task
def clear_active_conflict():     # Syntax: fab localhost clear_active_conflict
  """ Delete all content in the active conflict directory"""

  local(template("rm -rf %s/plugins/FabFlee/conflict_data/active_conflict/" % (env.localroot)))



@task
def instantiate(conflict_name):      # Syntax: fab localhost instantiate:conflict_name
  """ Copy modified active conflict directory to config_files (i.e. flee_conflict_name) to run instance with Flee """

  # 1. Copy modified active_conflict directory to instantiate runs with specific conflict name
  local(template("mkdir -p %s/plugins/FabFlee/config_files/%s" % (env.localroot, conflict_name)))
  local(template("mkdir -p %s/plugins/FabFlee/config_files/%s/input_csv" % (env.localroot, conflict_name)))
  local(template("cp %s/plugins/FabFlee/conflict_data/active_conflict/*.csv %s/plugins/FabFlee/config_files/%s/input_csv") % (env.localroot, env.localroot, conflict_name))
  local(template("cp %s/plugins/FabFlee/conflict_data/active_conflict/commands.log.txt %s/plugins/FabFlee/config_files/%s/") % (env.localroot, env.localroot, conflict_name))
  local(template("mkdir -p %s/plugins/FabFlee/config_files/%s/source_data" % (env.localroot, conflict_name)))
  local(template("cp %s/plugins/FabFlee/conflict_data/active_conflict/source_data/*.csv %s/plugins/FabFlee/config_files/%s/source_data") % (env.localroot, env.localroot, conflict_name))
  local(template("cp %s/plugins/FabFlee/config_files/flee/run.py %s/plugins/FabFlee/config_files/%s/run.py") % (env.localroot, env.localroot, conflict_name))

# Test Functions
from plugins.FabFlee.test_FabFlee import *


@task
def test_variability(config,**args):     # Syntax: fab localhost test_variability:flee_conflict_name,simulation_period=number,replicas=number
  """ Run a number of replicas for a specific conflict """

  number_of_replicas = int(args["replicas"])
  for i in range(0, number_of_replicas):
    update_environment({"label":"%s" % i})
    flee(config, **args)


@task
def test_sensitivity(config,**args):     #Syntax: fab localhost test_sensitivity:flee_conflict_name,simulation_period=number,name=MaxMoveSpeed,values=50-100-200
  """ Run simulation using different speed limits, movechances and awareness levels to test sensitivity
      Example: fab localhost test_sensitivity:flee_ssudan,simulation_period=604,name=MaxMoveSpeed,values=20-40-80
  """
  #SimSettingsCSVs directory: movechance and attractiveness

  import csv

  # 1. Generate simSettingsCSVs for which sensitivity analysis is conducted
  # 1a. extract parameter name
  parameter_name = args["name"]

  # 1b. convert value range to python list
  values = args["values"].split('-')

  # 2. Create a config directory for each simSettingsCSV (for each value in the list), and place csv in it
  for v in values:
    instantiate("%s_%s_%s" % (config, args["name"], v))

    csvfile = open('%s/plugins/FabFlee/config_files/%s_%s_%s/simsetting.csv' % (env.localroot, config, args["name"], v), "wb")
    writer = csv.writer(csvfile)
    writer.writerow([args["name"], v])
    csvfile.close()

    # 3. Run simulation for each config directory that is generated
    instance_config = ("%s_%s_%s" % (config, args["name"], v))
    flee(instance_config, **args)

    # 4. Analyse output and report sensitivity
