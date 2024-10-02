"""
This is the main script for constructing a new Flee simulation based on the information the user has defined in 
input_files/settings.txt. A template of this settings file is available in input_files. 

It calls/uses the following scripts to create the new simulation files:

1) settings.py: This script reads the settings.txt file in input_files. 
2) ACLED_API.py: This script obtains acled data using their api and your chosen simulation configuration in settings.txt. Your api key details must be provided. 
3) 

These files are created: 
1) locations.csv
2)  acled_data.csv - all the acled data from the acled website obtained using api key 

"""

import os 
import settings
import ACLED_API


def flee_network_construction():

    # ----------------------------------------------------
    # Configure network settings
    # ----------------------------------------------------
    
    setting_dict=settings.obtain_network_settings()
 
    settings.create_constructed_network_directory()

    settings.cp_settings_txt()

    # ----------------------------------------------------
    # Get the ACLED API data and convert to dataframe
    # ----------------------------------------------------

    api_key, email = ACLED_API.obtain_api_keys()

    response = ACLED_API.obtain_acled_data(api_key, email, setting_dict)

    acled_file = ACLED_API.save_acled_data_to_csv(response, setting_dict)

    # acled_df = ACLED_API.create_acled_data_frame(acled_file)

    # ----------------------------------------------------
    # Get Location Data
    # ----------------------------------------------------
    
    

    # ----------------------------------------------------



if __name__ == "__main__":
    flee_network_construction()