import glob, os
import sys
import csv

if __name__ == "__main__":

    input_dir = sys.argv[1]
    os.chdir(input_dir)
    with open('list_of_files.csv', mode='w') as list_of_files:
            file_writer = csv.writer(list_of_files, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for file in glob.glob("*.csv"):
                camp_name = file.rpartition('.')[0]
        
                file_writer.writerow([camp_name[4:].title(), file])
                

    print(file_writer)