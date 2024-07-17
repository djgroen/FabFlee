import os 
import settings
import ACLED_API


def flee_network_construction():

    # ----------------------------------------------------
    # Configure network settings
    # ----------------------------------------------------
    
    settings.obtain_network_settings()
 
    settings.create_constructed_network_directory()

    settings.mv_settings_txt()

    # ----------------------------------------------------
    # Get the ACLED API data and convert to dataframe
    # ----------------------------------------------------

    api_key, email = ACLED_API.obtain_api_keys()

    response = ACLED_API.obtain_acled_data(api_key, email)

    acled_file = ACLED_API.save_acled_data_to_csv(response)

    acled_df = ACLED_API.create_acled_data_frame(acled_file)

    # ----------------------------------------------------
    # Get Location Data
    # ----------------------------------------------------
    
    

    # ----------------------------------------------------



if __name__ == "__main__":
    flee_network_construction()