"""
Summary: This script reads the settings.txt file and extracts the network settings from it. 
"""
import os #used in create_constructed_network_directory


def obtain_network_settings():
    """
    Summary:
        Obtain the network settings from the settings.txt file
    
    Args:
        None
    
    Returns:
        settings: dict
    """

    # print("Obtaining network settings from settings.txt")

    settings = {}
    with open("../input_files/settings.txt", "r") as file:
        
        for line in file:
            if "network_name" in line:
                settings["network_name"] = line.split("=")[1].strip()
            if "network_type" in line:
                settings["network_type"] = line.split("=")[1].strip()
            if "country" in line:
                settings["country"] = line.split("=")[1].strip()
            if "start_date" in line:
                settings["start_date"] = line.split("=")[1].strip()
            if "end_date" in line:
                settings["end_date"] = line.split("=")[1].strip()
            if "event_type" in line:
                settings["event_type"] = line.split("=")[1].strip()
            
    # print(settings["event_type"])
    
    return settings
    

def create_constructed_network_directory():
    """
    Summary:
        Create a directory to store the constructed network. 
        The directory name is the network name obtained from the settings.txt file.
        The directory is within the constructed_networks directory, 
        with subdirectories for conflict and disaster networks.
    
    Args:
        network_dir_name
    
    Returns:
        None
    """
    # print("Creating directory to store the constructed network")

    #Get sim settings
    network_type = obtain_network_settings()["network_type"]
    network_dir_name = obtain_network_settings()["network_name"]

    # Create network directory name 
    network_dir_name = f"../constructed_networks/{network_type}/{network_dir_name}"

    # Check if network directory exists and create directory. 
    os.makedirs(network_dir_name, exist_ok=True) #TODO: at end change to false 
  
    return network_dir_name


def mv_settings_txt():
    """
    Summary:
        Move the settings.txt file to the constructed network directory.
    
    Args:
        None
    
    Returns:
        None
    """

    print("Moving settings.txt to the constructed network directory")

    # Get the network directory name
    network_dir_name = create_constructed_network_directory()

    # Copy settings.txt to the network directory
    os.system(f"cp ../input_files/settings.txt {network_dir_name}/settings.txt") 
    
    #TODO: modify cp to mv when testing is done


if __name__ == "__main__":

    print("Obtaining network settings from settings.txt")
    obtain_network_settings()

    print("Creating directory to store the constructed network")
    create_constructed_network_directory()

    print("Moving settings.txt to the constructed network directory")
    mv_settings_txt()

