import yaml
import os
# Load configuration from the YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Load YAML configuration file
def load_yaml_config(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
