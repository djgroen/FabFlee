"""
Summary: 
This script reads the settings.txt file and extracts the network settings from it. 
This script is called in Flee_simulation_construction.py
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
    print("Obtaining network settings from settings.txt")

    # Create empty settings dictionary 
    settings = {}

    # Go up one directory (scripts)
    current_directory = os.path.dirname(__file__)

    # Go to 'input_files' directory
    settings_directory = os.path.join(current_directory, '..', 'input_files')

    # Combine with file name
    settings_path = os.path.join(settings_directory, 'settings.txt')

    # Now you can access the file
    try:
        with open(settings_path, 'r') as file:
            # Read the settings file and extract the network settings
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

    except FileNotFoundError:
        print("settings.txt not found.")
    except PermissionError:
        print("Permission denied for settings.txt.")

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
    print("Creating directory to store the constructed network")

    #Get sim settings
    network_type = obtain_network_settings()["network_type"]
    network_dir_name = obtain_network_settings()["network_name"]

    # Create network directory name 
    network_dir_name = f"../constructed_networks/{network_type}/{network_dir_name}"

    # Check if network directory exists and create directory. 
    os.makedirs(network_dir_name, exist_ok=True) #TODO: at end change to false 
  
    return network_dir_name


def cp_settings_txt():
    """
    Summary:
        Move the settings.txt file to the constructed network directory.
    
    Args:
        None
    
    Returns:
        None
    """
    print("Copying settings.txt to the constructed network directory")

    # Get the network directory name
    network_dir_name = create_constructed_network_directory()

    # Copy settings.txt to the network directory
    os.system(f"cp ../input_files/settings.txt {network_dir_name}/settings.txt") 
    

if __name__ == "__main__":
    
    obtain_network_settings()
    
    create_constructed_network_directory()

    cp_settings_txt()

