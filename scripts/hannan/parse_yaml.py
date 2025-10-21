import yaml
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <path_to_yaml_file>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        return

    for key, value in data.items():
        if key.startswith('tulu'):
            if isinstance(value, dict) and 'model' in value:
                model_path = value.get('model')
                if model_path:
                    print(f"{key} {model_path}")

if __name__ == "__main__":
    main()
