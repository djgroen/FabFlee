from flee import coupling  # coupling interface for multiscale models
from flee.datamanager import handle_refugee_data, read_period
from flee.datamanager import DataTable  # DataTable.subtract_dates()
import numpy as np
from flee.postprocessing import analysis as a
import sys
import argparse
import os
from pprint import pprint
import csv
import pandas as pd
work_dir = os.path.dirname(os.path.abspath(__file__))
insert_day0_refugees_in_camps = False


def AddInitialRefugees(e, d, loc):
    """ Add the initial refugees to a location, using the location name"""
    num_refugees = int(d.get_field(loc.name, 0, FullInterpolation=True))
    for i in range(0, num_refugees):
        e.addAgent(location=loc)


def read_coupled_locations(csv_inputfile):
    c = []
    with open(csv_inputfile, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[0].startswith("#"):
                pass
            else:
                c += [row[0]]

    print("Coupling locations:", c, file=sys.stderr)
    return c


def run_micro_macro_model(e, c, submodel, ig, d, camp_locations, end_time):
    # Set up a mechanism to incorporate temporary decreases in refugees
    refugee_debt = 0
    # raw (interpolated) data from TOTAL UNHCR refugee count only.
    refugees_raw = 0

    while c.reuse_coupling():
        for t in range(0, end_time):

            # if t>0:
            ig.AddNewConflictZones(e, t)

            # Determine number of new refugees to insert into the system.
            new_refs = d.get_daily_difference(
                t, FullInterpolation=True, SumFromCamps=False
            ) - refugee_debt

            refugees_raw += d.get_daily_difference(
                t, FullInterpolation=True, SumFromCamps=False)

            if log_exchange_data is True:
                c.logNewAgents(t, new_refs)

            if new_refs < 0:
                refugee_debt = - new_refs
                new_refs = 0
            elif refugee_debt > 0:
                refugee_debt = 0

            # Insert refugee agents

            if submodel == "macro":
                print("t={}, inserting {} new agents".format(
                    t, new_refs), file=sys.stderr)
                # e.add_agents_to_conflict_zones(new_refs)
                for i in range(0, new_refs):
                    e.addAgent(e.pick_conflict_location())
                e.updateNumAgents(log=False)

            # e.printInfo()

            # exchange data with other code.
            # immediately after agent insertion to ensure ghost locations
            # work correctly.
            c.Couple(t)

            e.refresh_conflict_weights()
            t_data = t

            # e.enact_border_closures(t)
            e.evolve()

            # Calculation of error terms
            errors = []
            abs_errors = []
            loc_data = []

            camps = []

            for i in camp_locations:
                camps += [lm[i]]
                loc_data += [d.get_field(i, t)]

            # calculate retrofitted time.
            refugees_in_camps_sim = 0

            for camp in camps:
                refugees_in_camps_sim += camp.numAgents

            if e.getRankN(t):
                output = "{}".format(t)

                j = 0
                for i in camp_locations:
                    errors += [a.rel_error(lm[i].numAgents, loc_data[j])]
                    abs_errors += [a.abs_error(lm[i].numAgents, loc_data[j])]

                    j += 1

                output = "{}".format(t)

                for i in range(0, len(errors)):
                    output += ",{},{},{}".format(
                        lm[camp_locations[i]].numAgents,
                        loc_data[i],
                        errors[i]
                    )

                if refugees_raw > 0:
                    output += ",{},{},{},{},{},{}".format(
                        round(
                            float(np.sum(abs_errors)) / float(refugees_raw),
                            a.ROUND_NDIGITS
                        ),
                        int(sum(loc_data)),
                        e.numAgents(),
                        refugees_raw,
                        refugees_in_camps_sim,
                        refugee_debt
                    )
                else:
                    output += ",0,0,0,0,0,0"

                # print(output)

                with open(out_csv_file, "a+") as f:
                    f.write(output)
                    f.write("\n")
                    f.flush()
                # exit()

        # break while-loop(c.reuse_coupling) if coupling_type == file
        if not hasattr(c, "instance"):
            break


def run_manager(c, submodel, end_time):

    while c.reuse_coupling():
        for t in range(0, end_time):
            c.Couple(t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Required parameters (mandatory)
    parser.add_argument("--data_dir", required=True,
                        action="store", type=str,
                        help="the input data directory")

    parser.add_argument("--submodel", required=True,
                        action="store", type=str,
                        choices=["micro", "macro",
                                 "macro_manager", "micro_manager"],
                        help="the current active coupled model")

    parser.add_argument("--coupling_type", required=True,
                        action="store", type=str,
                        choices=["file", "muscle3"],
                        help="the coupling type to be used")

    parser.add_argument("--instance_index", required=True,
                        action="store", type=int,
                        help="the index of current active macro/micro session")

    parser.add_argument("--num_instances", required=True,
                        action="store", type=int,
                        help="set the number of concurrent launched sessions \
                        of macro and micro models")

    # Optional parameters
    parser.add_argument("--end_time",
                        action="store", type=int)

    parser.add_argument("--log_exchange_data",
                        action="store", type=str, default="True",
                        help="boolean flag to enable/disable logging \
                        exchanged data between macro and micro models")

    parser.add_argument("--weather_coupling",
                        action="store", type=str, default="False",
                        help="boolean flag to enable/disable weather coupling")

    args, unknown = parser.parse_known_args()

    print("args: {}".format(args), file=sys.stderr)

    data_dir = os.path.join(work_dir, args.data_dir)

    coupled_locations = read_coupled_locations(
        os.path.join(data_dir, "coupled_locations.csv")
    )

    start_date, end_time = read_period.read_conflict_period(
        os.path.join(data_dir, "conflict_period.csv")
    )

    submodel = args.submodel.lower()
    coupling_type = args.coupling_type
    instance_index = int(args.instance_index)
    num_instances = int(args.num_instances)

    if args.weather_coupling.lower() == "true":
        weather_coupling = True
    else:
        weather_coupling = False

    if args.end_time is not None:
        end_time = int(args.end_time)
    last_physical_day = end_time

    if args.log_exchange_data.lower() == "true":
        log_exchange_data = True
    else:
        log_exchange_data = False

    SumFromCamps = False  # Set to TRUE for production runs.

    if submodel in ["macro", "macro_manager"]:
        from flee import pflee as flee
        from flee import InputGeography
        submodel_id = 0
        validation_data_directory = os.path.join(data_dir, "source_data-0")
        if submodel == "macro":
            if weather_coupling is True:
                out_csv_file = os.path.join(
                    work_dir, "out", "weather", coupling_type, "macro",
                    "out[{}].csv".format(instance_index)
                )
            else:
                out_csv_file = os.path.join(
                    work_dir, "out", coupling_type, "macro",
                    "out[{}].csv".format(instance_index)
                )

    elif submodel in ["micro", "micro_manager"]:
        from flee import pmicro_flee as flee
        from flee import micro_InputGeography as InputGeography

        submodel_id = 1
        validation_data_directory = os.path.join(data_dir, "source_data-1")
        if submodel == "micro":
            if weather_coupling is True:
                out_csv_file = os.path.join(
                    work_dir, "out", "weather", coupling_type, "micro",
                    "out[{}].csv".format(instance_index)
                )
            else:
                out_csv_file = os.path.join(
                    work_dir, "out", coupling_type, "micro",
                    "out[{}].csv".format(instance_index)
                )

    print("This is submodel {}".format(submodel_id), file=sys.stderr)

    if submodel == "micro" and weather_coupling is True:
        weather_source_files = {}

        weather_source_files["precipitation"] = pd.read_csv(
            os.path.join(data_dir, "weather_data", "precipitation.csv")
        )

        weather_source_files["river_discharge"] = pd.read_csv(
            os.path.join(data_dir, "weather_data", "river_discharge.csv")
        )

        weather_source_files["40yrs_total_precipitation"] = pd.read_csv(
            os.path.join(data_dir, "weather_data", "40yrs_tp.csv")
        )

        weather_source_files["conflict_start_date"], _ = (
            read_period.read_conflict_period(
                os.path.join(data_dir, "conflict_period.csv")
            )
        )

        weather_source_files["output_log"] = os.path.join(
            work_dir, "out", "weather", coupling_type, submodel,
            "weather_log[{}].csv".format(instance_index)
        )
        prec_len = len(weather_source_files["precipitation"].index)

        if prec_len < end_time:
            print("The lenght of precipitation.csv ({}) is lower than "
                  "the simulation end_time ({})!".format(prec_len, end_time)
                  )
            exit()

        flee.Link = flee.Link_weather_coupling
        flee.weather_source_files = weather_source_files

    if weather_coupling is True:
        outputdir = os.path.join(work_dir, "out", "weather")
    else:
        outputdir = os.path.join(work_dir, "out")

    e = flee.Ecosystem()

    c = coupling.CouplingInterface(e, submodel,
                                   instance_index=instance_index,
                                   num_instances=num_instances,
                                   coupling_type=coupling_type,
                                   weather_coupling=weather_coupling,
                                   outputdir=outputdir,
                                   log_exchange_data=log_exchange_data
                                   )

    if coupling_type == "muscle3":
        if hasattr(c, "instance"):
            print("Instance attribute found for coupling class c.",
                  file=sys.stderr)
        else:
            print("WARNING: coupling class c does not have an "
                  "instance attribute.", file=sys.stderr)

    # setting coupling output file name
    if coupling_type == "file":
        if submodel == "micro":
            c.setCouplingChannel("out", "in")
        elif submodel == "macro":
            c.setCouplingChannel("in", "out")

    ig = InputGeography.InputGeography()

    ig.ReadFlareConflictInputCSV(
        os.path.join(data_dir, "conflicts-{}.csv".format(submodel_id))
    )

    ig.ReadLocationsFromCSV(
        os.path.join(data_dir, "locations-{}.csv".format(submodel_id))
    )

    ig.ReadLinksFromCSV(
        os.path.join(data_dir, "routes-{}.csv".format(submodel_id))
    )

    ig.ReadClosuresFromCSV(
        os.path.join(data_dir, "closures-{}.csv".format(submodel_id))
    )

    e, lm = ig.StoreInputGeographyInEcosystem(e)

    print("Val dir: {}. Start date set to {}.".format(
        validation_data_directory, start_date), file=sys.stderr
    )

    d = handle_refugee_data.RefugeeTable(
        csvformat="generic",
        data_directory=validation_data_directory,
        start_date=start_date,
        data_layout="data_layout.csv"
    )

    d.ReadL1Corrections(
        os.path.join(
            data_dir,
            "registration_corrections-{}.csv".format(submodel_id)
        )
    )
    # d.dump(0, 5)

    output_header_string = "Day,"

    camp_locations = e.get_camp_names()

    for i in camp_locations:
        # Add initial refugees to camps is currently not supported in
        # coupled mode, because the totals are not added up correctly across
        # the submodels.
        # All agents therefore have to start in conflict zones.

        if insert_day0_refugees_in_camps:
            AddInitialRefugees(e, d, lm[i])
        output_header_string += "{} sim,{} data,{} error,".format(
            lm[i].name, lm[i].name, lm[i].name
        )

    output_header_string += ",".join([
        "Total error",
        "refugees in camps (UNHCR)",
        "total refugees (simulation)",
        "raw UNHCR refugee count",
        "refugees in camps (simulation)",
        "refugee_debt"
    ])

    for l in coupled_locations:
        c.addCoupledLocation(lm[l], l)

    if submodel in ["macro", "macro_manager"]:
        # Add ghost conflict zones to macro model ("out" mode)
        c.addGhostLocations(ig)
    if submodel in ["micro", "micro_manager"]:
        # Couple all conflict locs in micro model ("in" mode)
        c.addMicroConflictLocations(ig)

    if submodel in ["macro", "micro"]:
        # DO NOT generate output file for the micro/macro mangers
        if e.getRankN(0):
            # output_header_string += "num agents,num agents in camps"
            print(output_header_string)
            with open(out_csv_file, "a+") as f:
                f.write(output_header_string)
                f.write("\n")
                f.flush()

    # end_time = 30
    if submodel in ["macro", "micro"]:
        run_micro_macro_model(
            e, c, submodel, ig, d, camp_locations, end_time
        )
    elif submodel in ["macro_manager", "micro_manager"]:
        run_manager(c, submodel, end_time)

    if log_exchange_data is True and e.getRankN(0):
        c.saveExchangeDataToFile()
        # only for one submodel instance plot the data exchanged
        if instance_index == 0:
            c.plotExchangedData()
            c.sumOutputCSVFiles()
