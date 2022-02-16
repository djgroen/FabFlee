from flee import InputGeography
import numpy as np
import sys

if __name__ == "__main__":
    """
    Usage <this> <end_time> <input> <output>
    """

    end_time = 1887

    if len(sys.argv)>1:
        if (sys.argv[1]).isnumeric():
            end_time = int(sys.argv[1])

    ig = InputGeography.InputGeography()

    ig.ReadLocationsFromCSV(sys.argv[2])

    file = open(sys.argv[3],"w")

    output_header_string = "#Day,"

    for l in ig.locations:
        output_header_string += " %s," % (l[0])
    
    output_header_string += "\n"
    file.write(output_header_string)

    for t in range(0,end_time):

        output = "%s" % t

        for l in ig.locations:
            # print(l)
            if l[4] == "conflict_zone":
                confl_date = int(l[5])
                if confl_date <= t:
                    output +=", 1"
                else: 
                    output +=", 0"
            else: 
                output +=", 0"

        output += "\n"
        file.write(output)

    file.close()
