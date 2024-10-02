import requests
import pandas as pd
import os 

def obtain_api_keys():
    """
    Summary:
        Obtain the user's ACLED API key and email from the settings.txt file.
    
    Args:
        None
    
    Returns:
        api_key: str
        email: str
    """
    print("Obtaining API keys from settings.txt")
    
    # Get api key from settings.txt in input_data
    with open("../input_files/settings.txt", "r") as f:
        for line in f:
            if "api_key" in line:
                api_key = line.split("=")[1].strip()
            if "email" in line:
                email = line.split("=")[1].strip()

    return api_key, email


def obtain_acled_data(api_key, email, setting_dict):
    """
    Fetches ACLED data based on settings from the settings dictionary and API credentials.

    Args:
        api_key (str): Your ACLED API key.
        email (str): Your ACLED registered email.
        setting_dict (dict): A dictionary containing the necessary settings.

    Returns:
        requests.Response: The response object containing the ACLED data, 
                            or None if there's an error.

    Raises:
        ValueError: If any required setting is missing from the settings dictionary.

    """  
    
    print("Obtaining ACLED data")

    # Check if required settings are present
    required_settings = ["start_date", "end_date", "country", "event_type"]
    missing_settings = [setting for setting in required_settings if setting not in setting_dict]
    if missing_settings:
        raise ValueError(f"Missing required settings in the settings dictionary: {', '.join(missing_settings)}")

    # Build the URL with settings and API credentials
    url = f"https://api.acleddata.com/acled/read?key={api_key}&email={email}"
    url += f"&start={setting_dict['start_date']}&end={setting_dict['end_date']}"
    url += f"&country={setting_dict['country']}&event_type={setting_dict['event_type']}"
    # url += ".csv"

    # Make the request and handle potential errors
    try:
        response = requests.post(url)
        print("response", response)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


    

def save_acled_data_to_csv(response, setting_dict):
    """
    Summary:
        Save the ACLED data to a csv file.
    
    Args:
        response: requests.models.Response
    
    Returns:
        acled_file: str
    """
    print("Saving the ACLED data to a csv file.")
    
    # Save the ACLED data to a csv file
    acled_file = f"../constructed_networks/conflict/{setting_dict['network_name']}/acled_data.csv"

    with open(acled_file, "wb") as f:
        f.write(response.content)

    return acled_file
    

# def create_acled_data_frame(acled_file):
#     """ 
#     Summary: 
#         Create a pandas dataframe from the ACLED data.

#     Args:
#         acled_file: str

#     Returns:
#         acled_df: pandas.core.frame.DataFrame
#     """

#     print("Creating ACLED dataframe") 

    

#     # Create a pandas dataframe from the ACLED data
#     acled_df = pd.read_csv(acled_file)

#     return acled_df

# def extract_conflict_location_names_from_ACLED(acled_df, location_type):

#     """
#     Summary: 
#         Extracts and prints the unique location names found in the specified column of the ACLED dataframe

#     Args: 
#         acled_df (pandas.core.frame.DataFrame): The ACLED dataframe
#         location_type (str): The column name containing the location names

#     Returns: 
#         None

#     TODO: This was taken from Maziar's script, needs amending for this. Rename function. What is location_type 
#     """

#     # Sort dataframe by location type
#     acled_df = acled_df.sort_values(location_type)

#     if location_type in acled_df.columns:
#         print(f"Here are locations found in '{location_type}'. Please find a population-table accordingly.")

#         # Extract and print the unique location names
#         location_names = acled_df[location_type].unique()

#         # Calculate the maximum length of location names
#         max_length = max(len(name) for name in location_names)

#         # Define the column width (you can adjust this as needed)
#         column_width = max_length + 4  # Add extra spaces for padding

#         # Initialise a counter to keep track of column position
#         column_counter = 0

#         # Print the location names in a grid
#         for name in location_names:
#             print(name.ljust(column_width), end=' ')
#             column_counter += 1

#             # Start a new row after every 3 columns
#             if column_counter == 4:
#                 print()
#                 column_counter = 0  # Reset the counter for the next row
#     else:
#         print(f"The column '{location_type}' does not exist in the DataFrame.")

#main loop 
if __name__ == "__main__":

    api_key, email = obtain_api_keys()

    response = obtain_acled_data(api_key, email, setting_dict)

    acled_file = save_acled_data_to_csv(response, setting_dict)

    create_acled_data_frame(acled_file)