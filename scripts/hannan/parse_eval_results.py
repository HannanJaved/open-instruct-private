import os
import json
import csv
import argparse
from pathlib import Path

def find_and_parse_results(root_dir, output_csv):
    """
    Finds all 'eval_results_*.json' files in subdirectories of root_dir,
    parses them, and writes the contents to a CSV file, sorted by task.
    """
    header = ['HP', 'task', 'metric', 'value', 'stderr']
    all_rows = []
    
    root_path = Path(root_dir).resolve()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('eval_results_') and filename.endswith('.json'):
                file_path = Path(dirpath) / filename
                relative_path = file_path.parent.resolve().relative_to(root_path)
                subdirectory_name = str(relative_path)
                leaf_subdirectory_name = file_path.parent.name

                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        results = data.get('results', {})
                        for task, metrics in results.items():
                            for metric_key, value in metrics.items():
                                if metric_key == 'alias':
                                    continue
                                
                                metric_name, metric_type = metric_key.split(',')
                                is_stderr = 'stderr' in metric_name
                                
                                # We will write std err in its own column
                                if is_stderr:
                                    continue

                                stderr_key = f"{metric_name}_stderr,{metric_type}"
                                stderr_value = metrics.get(stderr_key, '')

                                all_rows.append([
                                    subdirectory_name,
                                    task,
                                    metric_name,
                                    value,
                                    stderr_value
                                ])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {file_path}")
                    except Exception as e:
                        print(f"An error occurred while processing {file_path}: {e}")

    # Sort the collected rows by task (the second element in each row)
    all_rows.sort(key=lambda row: row[1])

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_rows)

def main():
    parser = argparse.ArgumentParser(description="Parse evaluation result files and export to CSV.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="The root directory to search for eval_results files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="The path to the output CSV file."
    )
    args = parser.parse_args()

    find_and_parse_results(args.root_dir, args.output_csv)
    print(f"Results successfully exported to {args.output_csv}")

if __name__ == "__main__":
    main()
