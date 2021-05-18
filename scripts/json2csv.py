import json
import csv
import os
import glob
  
current_dir = os.path.dirname(os.path.abspath(__file__))

all_files = glob.glob("*.json")

for filename in all_files:

    # Opening JSON file and loading the data
    # into the variable data
    with open(filename) as json_file:
        data = json.load(json_file)

    data = data['data']
  
    refugees = data["timeseries"]

    for j in refugees:
        del j["unix_timestamp"]

    # now we will open a file for writing
    data_file = open('{}.csv'.format(os.path.splitext(filename)[0]), 'w')
      
    # create the csv writer object
    csv_writer = csv.writer(data_file)
      
    # Counter variable used for writing 
    # headers to the CSV file
    count = 0
      
    for ref in refugees:
        if count == 0:
      
            # Writing headers of CSV file
            header = []
            csv_writer.writerow(header)
            count += 1
      
        # Writing data of CSV file
        csv_writer.writerow(ref.values())
      
    data_file.close()