from flee import pflee as flee
from flee import spawning
from flee.datamanager import handle_refugee_data, read_period
from flee.datamanager import DataTable #DataTable.subtract_dates()
from flee import InputGeography
import numpy as np
import flee.postprocessing.analysis as a
import sys
from flee.SimulationSettings import SimulationSettings

from datetime import datetime, timedelta

if __name__ == "__main__":

  # --- weather-coupling flags (parsed & stripped so positional args still work) ---
  import random
  _flags, _pos = {}, []
  for _tok in sys.argv[1:]:
      if _tok.startswith("--") and "=" in _tok:
          _k, _v = _tok[2:].split("=", 1); _flags[_k.strip()] = _v.strip()
      else:
          _pos.append(_tok)
  sys.argv = [sys.argv[0]] + _pos
  def _as_bool(x): return str(x).strip().lower() in ("1","true","yes","on")
  weather_coupling  = _as_bool(_flags.get("weather_coupling", "false"))
  use_precipitation = _as_bool(_flags.get("use_precipitation", "true"))
  use_discharge     = _as_bool(_flags.get("use_discharge", "true"))
  wc_q1 = float(_flags.get("q1", 0.15)); wc_q2 = float(_flags.get("q2", 0.75)); wc_Q = float(_flags.get("Q", 8000.0))
  _seed = _flags.get("seed")
  if _seed is not None:
      np.random.seed(int(_seed)); random.seed(int(_seed))

  start_date,end_time = read_period.read_sim_period("{}/sim_period.csv".format(sys.argv[1]))

  if len(sys.argv)<4:
    print("Please run using: python3 run.py <your_csv_directory> <your_refugee_data_directory> <duration in days> <optional: simsettings.yml> > <output_directory>/<output_csv_filename>")

  input_csv_directory = sys.argv[1]
  validation_data_directory = sys.argv[2]
  if int(sys.argv[3]) > 0:
    end_time = int(sys.argv[3])

  if len(sys.argv)==5:
    flee.SimulationSettings.ReadFromYML(sys.argv[4])
  else:
    flee.SimulationSettings.ReadFromYML("simsetting.yml")

  # Conflict file will be read if modelling conflict-driven displacement. Ignored otherwise.
  flee.SimulationSettings.ConflictInputFile = "%s/conflicts.csv" % input_csv_directory
  # Flood file will be read if modelling flood-driven displacement. Ignored otherwise.
  flee.SimulationSettings.FloodLevelInputFile = "%s/flood_level.csv" % input_csv_directory

  # --- enable single-scale weather coupling (M0-M3) ---
  if weather_coupling:
      from flee import weather_coupling as wc
      wc.configure(input_csv_directory, start_date, end_time,
                   use_precipitation=use_precipitation, use_discharge=use_discharge,
                   q1=wc_q1, q2=wc_q2, Q=wc_Q)
      flee.Link = wc.Link_weather_coupling

  e = flee.Ecosystem()

  ig = InputGeography.InputGeography()

  ig.ReadLocationsFromCSV("%s/locations.csv" % input_csv_directory)

  ig.ReadLinksFromCSV("%s/routes.csv" % input_csv_directory)

  ig.ReadClosuresFromCSV("%s/closures.csv" % input_csv_directory)

  e,lm = ig.StoreInputGeographyInEcosystem(e)

  if SimulationSettings.spawn_rules["read_from_agents_csv_file"] == True:
      ig.ReadAgentsFromCSV(e, "%s/agents.csv" % input_csv_directory)

  d = handle_refugee_data.RefugeeTable(csvformat="generic", data_directory=validation_data_directory, start_date=start_date, data_layout="data_layout.csv", population_scaledown_factor=SimulationSettings.optimisations["PopulationScaleDownFactor"], start_empty=SimulationSettings.spawn_rules["EmptyCampsOnDay0"])

  d.ReadL1Corrections("%s/registration_corrections.csv" % input_csv_directory)

  output_header_string = "Day,Date,"

  camp_locations      = e.get_camp_names()

  for l in camp_locations:
      spawning.add_initial_refugees(e,d,lm[l])
      output_header_string += "%s sim,%s data,%s error," % (lm[l].name, lm[l].name, lm[l].name)

  output_header_string += "Total error,refugees in camps (UNHCR),total refugees (simulation),raw UNHCR refugee count,refugees in camps (simulation),refugee_debt"

  if SimulationSettings.log_levels["idp_totals"] > 0:
      output += ",total IDPs".format(e.numIDPs())

  if e.getRankN(0):
      print(output_header_string)

  # Set up a mechanism to incorporate temporary decreases in refugees
  refugee_debt = 0
  refugees_raw = 0 #raw (interpolated) data from TOTAL UNHCR refugee count only.

  for t in range(0,end_time):
    
    #if t>0:
    if weather_coupling:
      wc.Link_weather_coupling.current_day = t

    ig.AddNewConflictZones(e,t)

    new_refs,refugees_raw,refugee_debt = spawning.spawn_daily_displaced(e,t,d)

    spawning.refresh_spawn_weights(e)

    e.enact_border_closures(t)
    e.evolve()

    if weather_coupling:
      wc.sample_day(t)

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


    date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=t)
    output = "%s,%s" % (t, date.strftime("%Y-%m-%d"))

    for i in range(0,len(errors)):
      output += ",%s,%s,%s" % (lm[camp_locations[i]].numAgents, loc_data[i], errors[i])

    if refugees_raw>0:
      output += ",%s,%s,%s,%s,%s,%s" % (float(np.sum(abs_errors))/float(refugees_raw), int(sum(loc_data)), e.numAgents(), refugees_raw, refugees_in_camps_sim, refugee_debt)
    else:
      output += ",0.0,0,{},0,{},0".format(e.numAgents(), refugees_in_camps_sim)

    if SimulationSettings.log_levels["idp_totals"] > 0:
      output += ",{}".format(e.numIDPs())

    if e.getRankN(t):
        print(output)


  if weather_coupling:
      _cfg = {(False, False): "M0", (True, False): "M1",
              (False, True): "M2", (True, True): "M3"}.get((use_precipitation, use_discharge), "Mx")
      wc.write_diagnostics("weather_diag_%s_seed%s" % (_cfg, _seed if _seed is not None else "NA"))
