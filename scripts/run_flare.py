from flee import InputGeography
from flare import Ecosystem
import numpy as np
import sys

if __name__ == "__main__":

    end_time = 100

    if len(sys.argv)>1:
        if (sys.argv[1]).isnumeric():
            end_time = int(sys.argv[1])

    if len(sys.argv)>2:
        input_dir = sys.argv[2]
    else:
        input_dir = "test_input_csv"


    if len(sys.argv)>3:
        out_file = sys.argv[3]
    else:
        out_file = "flare-out.csv"


    ig = InputGeography.InputGeography()

    ig.ReadLocationsFromCSV("%s/locations.csv" % input_dir)

    ig.ReadLinksFromCSV("%s/routes.csv" % input_dir)

    e = Ecosystem.Ecosystem()

    lm = e.StoreInputGeographyInEcosystem(ig)

    #print("Network data loaded")

    file = open("%s" % out_file,"w")

    output_header_string = "#Day,"

    for l in e.locations:
        output_header_string += " %s," % (l.name)
    
    output_header_string += "\n"
    file.write(output_header_string)

    for t in range(0,end_time):

        e.evolve()

        output = "%s" % t

        for l in e.locations:
            if l.flare:
                output +=",1"
            else: 
                output +=",0"

        output += "\n"
        file.write(output)

    file.close()
