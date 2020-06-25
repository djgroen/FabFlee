from flee import food_flee as flee 	#use food_flee instead of flee to account for food modifications but call it flee so nothing else has to change in the code
from flee.datamanager import handle_refugee_data
from flee.datamanager import DataTable #DataTable.subtract_dates()
from flee import InputGeography_food as InputGeography
import numpy as np
import pandas as pd
import flee.postprocessing.analysis as a
import sys

def AddInitialRefugees(e, d, loc):
  """ Add the initial refugees to a location, using the location name"""
  num_refugees = int(d.get_field(loc.name, 0, FullInterpolation=True))
  for i in range(0, num_refugees):
    e.addAgent(location=loc)

def date_to_sim_days(date):
  return DataTable.subtract_dates(date,"2013-12-15")

def start_movechance_log(e, end_time):
  MC_log=pd.DataFrame(index=range(0,end_time),columns=e.locationNames)
  return MC_log

def movechance_log(MC_log, e, t, end_time):
  for i in range(len(e.locationNames)):
    MC_log.loc[t,e.locationNames[i]]=e.locations[i].movechance
  return MC_log


if __name__ == "__main__":

  end_time = 100
  last_physical_day = 100

  if len(sys.argv)<4:
    print("Please run using: python3 run.py <your_csv_directory> <your_refugee_data_directory> <duration in days> <optional: simulation_settings.csv> > <output_directory>/<output_csv_filename>")

  input_csv_directory = sys.argv[1]
  validation_data_directory = sys.argv[2]
  duration = int(sys.argv[3])
  end_time = int(sys.argv[3])
  last_physical_day = int(sys.argv[3])

  if len(sys.argv)==5:
    flee.SimulationSettings.ReadFromCSV(sys.argv[4])

  e = flee.Ecosystem()

  ig = InputGeography.InputGeography()

  ig.ReadLocationsFromCSV("%s/locations.csv" % input_csv_directory)

  ig.ReadLinksFromCSV("%s/routes.csv" % input_csv_directory)

  ig.ReadClosuresFromCSV("%s/closures.csv" % input_csv_directory)

  e,lm = ig.StoreInputGeographyInEcosystem(e)

  #print("Network data loaded")

  d = handle_refugee_data.RefugeeTable(csvformat="generic", data_directory=validation_data_directory, start_date="2013-12-15", data_layout="data_layout.csv")

  output_header_string = "Day,"

  camp_locations      = e.get_camp_names()

  for l in camp_locations:
    #AddInitialRefugees(e,d,lm[l])
    output_header_string += "%s sim,%s data,%s error," % (lm[l].name, lm[l].name, lm[l].name)

  output_header_string += "Total error,refugees in camps (UNHCR),total refugees (simulation),raw UNHCR refugee count,refugees in camps (simulation),refugee_debt"

  print(output_header_string)

  # Set up a mechanism to incorporate temporary decreases in refugees
  refugee_debt = 0
  refugees_raw = 0 #raw (interpolated) data from TOTAL UNHCR refugee count only.

  #Load food info:
  [critict,IPC_all,current_i]=flee.initiate_food() #has to go in the main part of flee before starting time count
  old_line=0
  print("Loaded food info", file=sys.stderr)

  MC_log=start_movechance_log(e,end_time)
  MC=[]

  for t in range(0,end_time):
    #Evaluate needed line for IPC:
    line_IPC=flee.line42day(t,current_i,critict)           #has to go in the time count of flee to choose the values of IPC according to t

    #if t>0:
    ig.AddNewConflictZones(e,t)

    # Determine number of new refugees to insert into the system.
    new_refs = d.get_daily_difference(t, FullInterpolation=True) - refugee_debt
    refugees_raw += d.get_daily_difference(t, FullInterpolation=True)

    if new_refs < 0:
      refugee_debt = -new_refs
      new_refs = 0
    elif refugee_debt > 0:
      refugee_debt = 0

    #Insert refugee agents
    for i in range(0, new_refs):
      e.addAgent(e.pick_conflict_location())

    e.refresh_conflict_weights()
    t_data = t

    #Update (if needed the IPC indexes and movechances)
    if not old_line==line_IPC:
      print("Time = %d. Updating IPC indexes and movechances"%(t), file=sys.stderr)
      e.update_IPC_MC(line_IPC,IPC_all)                 #update all locations in the ecosystem: IPC indexes and movechances (inside t loop)
      #print("After updating IPC and movechance:")
      e.printInfo()
    old_line=line_IPC
 
    MC=meanMC(MC)

    movechance_log(MC_log,e,t,end_time)				#adds a line to the Pandas dataframe where all the movechances are being saved

    e.enact_border_closures(t)
    e.evolve()

    #Calculation of error terms
    errors = []
    abs_errors = []
    loc_data = []

    camps = []
    for i in camp_locations:
      camps += [lm[i]]
      loc_data += [d.get_field(i, t)]

    # calculate retrofitted time.
    refugees_in_camps_sim = 0
    for c in camps:
      refugees_in_camps_sim += c.numAgents

    # calculate errors
    j=0
    for i in camp_locations:
      errors += [a.rel_error(lm[i].numAgents, loc_data[j])]
      abs_errors += [a.abs_error(lm[i].numAgents, loc_data[j])]

      j += 1

    output = "%s" % t

    for i in range(0,len(errors)):
      output += ",%s,%s,%s" % (lm[camp_locations[i]].numAgents, loc_data[i], errors[i])

    if refugees_raw>0:
      #output_string += ",%s,%s,%s,%s" % (float(np.sum(abs_errors))/float(refugees_raw), int(sum(loc_data)), e.numAgents(), refugees_raw)
      output += ",%s,%s,%s,%s,%s,%s" % (float(np.sum(abs_errors))/float(refugees_raw), int(sum(loc_data)), e.numAgents(), refugees_raw, refugees_in_camps_sim, refugee_debt)
    else:
      output += ",0,0,0,0,0,0"
      #output_string += ",0"


    print(output)

  MC_log.to_csv("movechance.csv",index_label="Days")
  print(mean(MC), file=sys.stderr)
