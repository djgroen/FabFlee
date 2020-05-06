from flee import pflee  # parallel implementation
from flee import coupling  # coupling interface for multiscale models
from datamanager import handle_refugee_data
from datamanager import DataTable  # DataTable.subtract_dates()
from flee import InputGeography
import numpy as np
import outputanalysis.analysis as a
import sys
import argparse


def AddInitialRefugees(e, loc):
    """ Add the initial refugees to a location, using the location name"""
    num_refugees = 10000
    for i in range(0, num_refugees):
        e.addAgent(location=loc)


def date_to_sim_days(date):
    return DataTable.subtract_dates(date, "2010-01-01")

if __name__ == "__main__":

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    # Required parameters (mandatory)
    parser.add_argument('--submodel', required=True,
                        action="store", type=str,
                        choices=['micro', 'macro'])
    parser.add_argument('--coupling_type', required=True,
                        action="store", type=str,
                        choices=['file', 'muscle3'])
    parser.add_argument('--input_csv_directory', required=True,
                        action="store", type=str)

    # Optional parameters
    parser.add_argument('--end_time',
                        action="store", type=int, default="10")
    parser.add_argument('--last_physical_day',
                        action="store", type=int, default='10')

    args, unknown = parser.parse_known_args()
    print(args)

    if args.submodel == 'macro':
        submodel_id = 1
    elif args.submodel == 'micro':
        submodel_id = 0
    coupling_type = args.coupling_type
    end_time = int(args.end_time)
    last_physical_day = int(args.last_physical_day)
    input_csv_directory = args.input_csv_directory

    e = pflee.Ecosystem()
    c = coupling.CouplingInterface(e, coupling_type=coupling_type,
                                   submodel=args.submodel)

    if submodel_id == 1:
        # macro
        c.setCouplingChannel("out", "in")
    elif submodel_id == 0:
        # micro
        c.setCouplingChannel("in", "out")

    ig = InputGeography.InputGeography()

    ig.ReadLocationsFromCSV("%s/locations-%s.csv" %
                            (input_csv_directory, submodel_id))

    ig.ReadLinksFromCSV("%s/routes-%s.csv" %
                        (input_csv_directory, submodel_id))

    ig.ReadClosuresFromCSV("%s/closures-%s.csv" %
                           (input_csv_directory, submodel_id))

    e, lm = ig.StoreInputGeographyInEcosystem(e)

    output_header_string = "Day,"

    coupled_locations = ["N", "E", "S", "W"]
    camp_locations = list(lm.keys())

    if submodel_id == 0:
        AddInitialRefugees(e, lm["A"])

    for l in coupled_locations:
        c.addCoupledLocation(lm[l], l)

    if submodel_id == 0:
        # Add ghost conflict zones to macro model ("out" mode)
        c.addGhostLocations(ig)
    if submodel_id == 1:
        # Couple all conflict locs in micro model ("in" mode)
        c.addMicroConflictLocations(ig)

    for l in e.locations:
        output_header_string += "%s sim," % (l.name)

    if e.getRankN(0):
        output_header_string += "num agents,num agents in camps"
        print(output_header_string)

    # Set up a mechanism to incorporate temporary decreases in refugees
    refugee_debt = 0
    # raw (interpolated) data from TOTAL UNHCR refugee count only.
    refugees_raw = 0
    while c.reuse_couling():
        for t in range(0, end_time):

            # if t>0:
            ig.AddNewConflictZones(e, t)

            # Determine number of new refugees to insert into the system.
            new_refs = 0
            if submodel_id == 0:
                new_refs = 100
            refugees_raw += new_refs

            # Insert refugee agents
            if submodel_id == 0:
                e.add_agents_to_conflict_zones(new_refs)

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

            camps = []
            for i in camp_locations:
                camps += [lm[i]]

            # calculate retrofitted time.
            refugees_in_camps_sim = 0
            for camp in camps:
                refugees_in_camps_sim += camp.numAgents

            if e.getRankN(t):
                output = "%s" % t

                for i in range(0, len(e.locations)):
                    output += ",%s" % (e.locations[i].numAgents)

                if refugees_raw > 0:
                    output += ",%s,%s" % (e.numAgents(), refugees_in_camps_sim)
                else:
                    output += ",0,0"

                print(output)

        if coupling_type == 'file':
            break
