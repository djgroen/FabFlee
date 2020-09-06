from flee import InputGeography
from flare import Ecosystem
import numpy as np
import sys

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

    ig = InputGeography.InputGeography()

    ig.ReadLocationsFromCSV("%s/locations%s.csv" % (input_dir, file_suffix))

    ig.ReadLinksFromCSV("%s/routes%s.csv" % (input_dir, file_suffix))

    e = Ecosystem.Ecosystem()

    lm = e.StoreInputGeographyInEcosystem(ig)

    #print("Network data loaded")

    print("output file -> %s" % (out_file))
    file = open("%s" % out_file, "w")

    output_header_string = "#Day,"

    for l in e.locations:
        output_header_string += " %s," % (l.name)

    output_header_string += "\n"
    file.write(output_header_string)

    for t in range(0, end_time):

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
