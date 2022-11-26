import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    """
    Usage <this> <agents_out_dir> <timestep> <output_matrix_csv>
    """

    df = pd.read_csv(sys.argv[1])

    newdf = df[(df['#time'] == int(sys.argv[2]))]

    #print(newdf.to_string()) 
    #print(newdf.head()) 


    #print("original_location", df["original_location"].unique())
    #print("current_location", df["current_location"].unique())

    l_orig = newdf["original_location"].unique()
    l_cur = newdf["current_location"].unique()

    travel_matrix = pd.DataFrame(np.zeros((len(l_orig), len(l_cur)), dtype='int'), l_orig, l_cur)

    for index, row in newdf.iterrows():
        o = row['original_location']
        c = row['current_location']
        cur_value = travel_matrix.at[o,c]

        travel_matrix.at[o,c] = cur_value+1

    print(travel_matrix.to_string())

    travel_matrix.to_csv(sys.argv[3])
