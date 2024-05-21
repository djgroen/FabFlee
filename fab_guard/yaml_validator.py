import os
import yaml
import json
from jsonschema import validate, ValidationError, Draft7Validator
import sys

class YAMLValidator:
    def __init__(self, schema_path):
        self.schema = self.load_schema(schema_path)

    def load_schema(self, schema_path):
        """Load the JSON schema from a file."""
        absolute_schema_path = os.path.abspath(schema_path)
        with open(absolute_schema_path, 'r') as schema_file:
            return json.load(schema_file)

    def load_yaml(self, yaml_path):
        """Load YAML content from a file."""
        absolute_yaml_path = os.path.abspath(yaml_path)
        with open(absolute_yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def validate_yaml(self, yaml_content):
        """Validate the loaded YAML content against the schema."""
        try:
            validate(instance=yaml_content, schema=self.schema)
            print("YAML content is valid.")
            return True
        except ValidationError as e:
            print(f"YAML content is invalid: {e}")
            return False

    def detailed_validation(self, yaml_content):
        """Perform detailed validation and report all errors."""
        validator = Draft7Validator(schema=self.schema)
        errors = sorted(validator.iter_errors(yaml_content), key=lambda e: e.path)

        if errors:
            for error in errors:
                print(f"Error: {error.message} at {'/'.join(map(str, error.path))}")
            return False
        else:
            print("YAML content is valid.")
            return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_yaml_file> <path_to_json_schema>")
    else:
        yaml_path = os.path.abspath(sys.argv[1])
        schema_path = os.path.abspath(sys.argv[2])
        validator = YAMLValidator(schema_path)
        yaml_content = validator.load_yaml(yaml_path)

        # Choose one of the validation methods
        #validator.validate_yaml(yaml_content)
        # Or for detailed validation with all errors:
        validator.detailed_validation(yaml_content)
