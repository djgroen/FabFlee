import yaml
from genson import SchemaBuilder
import sys
import json


def generate_schema_from_yaml(yaml_file):
    # Load the YAML content
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Initialize a Schema Builder
    builder = SchemaBuilder()
    builder.add_object(yaml_content)

    # Generate JSON schema
    return builder.to_schema()


def save_schema_to_file(schema, output_file):
    # Save the schema to a JSON file
    with open(output_file, 'w') as file:
        json.dump(schema, file, indent=4)


if __name__ == "__main__":
    # Check for command line arguments for the YAML and output file paths
    if len(sys.argv) != 3:
        print("Usage: python generate_schema.py <path_to_yaml_file> <output_file>")
        sys.exit(1)

    # Get the file paths from the command line
    yaml_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Generate schema
    schema = generate_schema_from_yaml(yaml_file_path)

    # Save the generated schema to a file
    save_schema_to_file(schema, output_file_path)
    print(f"Schema has been saved to {output_file_path}")
