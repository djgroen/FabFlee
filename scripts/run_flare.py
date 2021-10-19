import numpy as np
import sys

from flee import InputGeography
from flare import Ecosystem


def run_flare(config_dir, flare_out_dir,
              simulation_period=100, file_suffix=""):

    ig = InputGeography.InputGeography()

    ig.ReadLocationsFromCSV("%s/locations%s.csv" % (config_dir, file_suffix))

    ig.ReadLinksFromCSV("%s/routes%s.csv" % (config_dir, file_suffix))

    e = Ecosystem.Ecosystem()

    lm = e.StoreInputGeographyInEcosystem(ig)

    # print("Network data loaded")

    print("output file -> %s" % (flare_out_dir))
    file = open("%s" % flare_out_dir, "w")

    output_header_string = "#Day,"

    for l in e.locations:
        output_header_string += " %s," % (l.name)

    output_header_string += "\n"
    file.write(output_header_string)

    for t in range(0, simulation_period):

        e.evolve()

        output = "%s" % t

        for l in e.locations:
            if l.flare:
                output += ",1"
            else:
                output += ",0"

        output += "\n"
        file.write(output)

    file.close()


if __name__ == "__main__":

    end_time = 100

    if len(sys.argv) > 1:
        if (sys.argv[1]).isnumeric():
            end_time = int(sys.argv[1])

    if len(sys.argv) > 2:
        input_dir = sys.argv[2]
    else:
        input_dir = "test_input_csv"

    if len(sys.argv) > 3:
        out_file = sys.argv[3]
    else:
        out_file = "flare-out.csv"

    if len(sys.argv) > 4:
        file_suffix = sys.argv[4]
    else:
        file_suffix = ""

    run_flare(
        config_dir=input_dir, flare_out_dir=out_file,
        simulation_period=end_time, file_suffix=file_suffix
    )
